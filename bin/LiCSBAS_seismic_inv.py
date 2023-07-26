#!/usr/bin/env python3
"""
v1.0.0 Jack McGrath, University of Leeds

Load in inverted displacements from LiCSBAS and fit pre- and post- seismic linear velocities, co-seismic displacements and postseismic relaxations.

Based off Liu et al. (2021), Improving the Resolving Power of InSAR for Earthquakes Using Time Series: A Case Study in Iran

Work flow:
    1) De-outliering of the data
        a) Iteritive process, where a temporal filter is applied to the data (which breaks at definied earthquakes), and outliers are defined as any
        displacement with a residual > outlier_thresh * filter_std. These are then replaced with the filtered value, and the process repeated until all
        displacements are within the threshold value, as large outliers will peturb the filter. The original data is then checked against the deoutliered
        filtered value, and any outliers replaced with the filtered value
        b) Using the RANSAC algorithm, where a temporal filter is applied to the data (which breaks at definied earthquakes), and RANSAC is applied to
        the residuals, and the outliers are replaced with filtered values. The original data is then checked against the deoutliered, filtered value, and
        any outliers replaced with the filtered value

    2) Fitting velocities
        Velocities are currently fit, allowing a long-term trend (pre-seismic linear velocity), a coseismic displacement (as a heaviside function), post-seismic
        relaxation (as a logarithmic decay) and a post-seismic velocity (linear)
        A check can be added for the minimum coseismic displacement, where inverted displacement < threshold is considered beneath detectable limits

Input files:
    eq_list.txt: Text file containing EQ dates, and optionally which parameters to fit to that earthquake ([C]oseismic displacement, logarithmic [R]elaxation, [P]ostseismic linear velocity, e[X]clude EQ)
                    e.g. 20161113 CRP
    mask:        Mask file produced by LiCSBAS15_mask_ts.py
    cum.h5:      Output from LiCSBAS13_sb_inv.py

#%% Change log

v1.0.0 20230714 Jack McGrath, University of Leeds
 - Initial Implementation
"""

import os
import re
import shutil
import sys
import h5py as h5
import time
import warnings
import argparse
import multiprocessing as multi
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import LiCSBAS_io_lib as io_lib
import LiCSBAS_plot_lib as plot_lib
import SCM
from sklearn.linear_model import RANSACRegressor
from scipy.interpolate import CubicSpline
from lmfit.model import *
from scipy import stats

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    '''
    Use a multiple inheritance approach to use features of both classes.
    The ArgumentDefaultsHelpFormatter class adds argument default values to the usage help message
    The RawDescriptionHelpFormatter class keeps the indentation and line breaks in the ___doc___
    '''
    pass

class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg

def init_args():
    global args

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=CustomFormatter)
    parser.add_argument('-f', dest='frame_dir', default="./", help="directory of LiCSBAS output of a particular frame")
    parser.add_argument('-t', dest='ts_dir', default="TS_GEOCml10GACOS", help="folder containing .h5 file")
    parser.add_argument('-d', dest='unw_dir', default='GEOCml10GACOS', help="folder containing unw ifg")
    parser.add_argument('-i', dest='h5_file', default='cum.h5', help='.h5 file containing results of LiCSBAS velocity inversion')
    parser.add_argument('-o', dest='out_dir', default='results/seismic_vels', help='folder in TSA dir outputs are written to')
    parser.add_argument('-r', dest='ref_file', default='130ref.txt', help='txt file containing reference area')
    parser.add_argument('-m', dest='mask_file', default='results/mask', help='mask file to apply to velocities')
    parser.add_argument('-e', dest='eq_list', default=None, help='Text file containing the dates of the earthquakes to be fitted')
    parser.add_argument('-s', dest='outlier_thre', default=3, type=float, help='StdDev threshold used to remove outliers')
    parser.add_argument('--n_para', dest='n_para', default=False, type=int, help='number of parallel processing')
    parser.add_argument('--tau', dest='tau', default=6, help='Post-seismic relaxation time (days)')
    parser.add_argument('--max_its', dest='max_its', default=5, type=int, help='Maximum number of iterations for temporal filter')
    parser.add_argument('--nofilter', dest='deoutlier', default=True, action='store_false', help="Don't do any temporal filtering")
    parser.add_argument('--applymask', dest='apply_mask', default=False, action='store_true', help="Apply mask to cum data before processing")
    parser.add_argument('--RANSAC', dest='ransac', default=False, action='store_true', help="Deoutlier with RANSAC algorithm")
    parser.add_argument('--replace_outliers', dest='replace_outliers', default=False, action='store_true', help='Replace outliers with filter value instead of nan')
    parser.add_argument('--no_vcm', dest='use_weights', default=True, action='store_false', help="Don't calculate VCM for each date - estimate errors with identity matrix (faster)")

    args = parser.parse_args()

def start():
    global start_time
    # intialise and print info on screen
    start_time = time.time()
    ver="1.0"; date=20230707; author="Jack McGrath"
    print("\n{} ver{} {} {}".format(os.path.basename(sys.argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)

def finish():
    #%% Finish
    elapsed_time = time.time() - start_time
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))
    print("\n{} {} finished!".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)
    print('Output file: {}\n'.format(os.path.relpath(outh5file)))

def set_input_output():
    global tsadir, infodir, resultdir, outdir, ifgdir, metadir
    global h5file, reffile, maskfile, eqfile, outh5file
    global q, outlier_thresh, mask_final

    # define input directories
    ifgdir = os.path.abspath(os.path.join(args.frame_dir, args.unw_dir))
    tsadir = os.path.abspath(os.path.join(args.frame_dir, args.ts_dir))
    infodir = os.path.join(tsadir, 'info')
    resultdir = os.path.join(tsadir, 'results')
    outdir = os.path.join(resultdir, 'seismic_vels')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    metadir = os.path.join(outdir, 'results')
    if not os.path.exists(metadir):
        os.mkdir(metadir)

    # define h5 files
    h5file = os.path.join(tsadir, args.h5_file)
    if args.deoutlier:
        if args.replace_outliers:
            outh5file = os.path.join(tsadir, outdir, 'cum_lptoutliers.h5')
        else:
            outh5file = os.path.join(tsadir, outdir, 'cum_nanoutliers.h5')
    else:
        outh5file = os.path.join(tsadir, outdir, 'cum.h5')    

    # If no reffile defined, search for 13ref, then 130ref, in this folder and infodir
    reffile = os.path.join(tsadir, args.ref_file)
    if not os.path.exists(reffile):
        reffile = os.path.join(infodir, args.ref_file)
        if not os.path.exists(reffile):
            if args.ref_file == '13ref.txt':
                # Seach for 13ref.txt
                reffile = os.path.join(tsadir, '130ref.txt')
                if not os.path.exists(reffile):
                    reffile = os.path.join(infodir, '130ref.txt')
                    if not os.path.exists(reffile):
                        print('\nNo reffile 13ref.txt or 130ref.txt found! No referencing occuring')
                        reffile = []
            else:
                print('\nNo reffile {} found! No referencing occuring'.format(args.ref_file))
                reffile = []
        if reffile != []:
            print('\nHad to search for reffile. Using {}'.format(reffile))

    maskfile = os.path.join(tsadir, args.mask_file)
    if not os.path.exists(maskfile):
        print('\nNo maskfile found. Not masking....')
        args.apply_mask = False
        mask_final = False
    else:
        mask_final = True

    eqfile = os.path.abspath(args.eq_list)

    outlier_thresh = args.outlier_thre

    q = multi.get_context('fork')

def load_data():
    global width, length, data, n_im, cum, dates, length, width, n_para, eq_dates, eq_params, n_eq, eq_dt, eq_ix, ord_eq, date_ord, eq_dates, valid, n_valid, ref

    data = h5.File(h5file, 'r')
    dates = [dt.datetime.strptime(str(d), '%Y%m%d').date() for d in np.array(data['imdates'])]

    cum = np.array(data['cum'])

    # read reference
    ref = reference_disp(cum, reffile)
    cum = cum - ref

    n_im, length, width = cum.shape

    # Identify all pixels where the is time series data
    vel = np.array(data['vel'])

    if mask_final:
        mask = io_lib.read_img(maskfile, length, width)
        shutil.copy(maskfile, os.path.join(metadir, 'mask'))
        if args.apply_mask:
            print('Applying Mask to cum data')
            maskx, masky = np.where(mask == 0)
            cum[:, maskx, masky] = np.nan
            vel[maskx, masky] = np.nan

    valid = np.where(~np.isnan(vel))
    n_valid = valid[0].shape[0]

    # multi-processing
    try:
        n_para = len(os.sched_getaffinity(0))
    except:
        n_para = multi.cpu_count()

    if args.n_para:
        n_para = args.n_para if args.n_para <= n_para else n_para

    ## Sort dates
    # Get list of earthquake dates and index
    eq_dates, eq_params = read_eq_list(eqfile)
    n_eq = len(eq_dates)

    # Find which index each earthquake correlates to
    eq_dt = [dt.datetime.strptime(str(eq_date), '%Y%m%d').date() for eq_date in eq_dates]
    eq_ix = []

    for eq_date in eq_dates:
        # Convert all dates to ordinal, and see which acquisitions are before the earthquakes
        eq_ix.append(np.sum([1 for d in dates if d.toordinal() < dt.datetime.strptime(str(eq_date), '%Y%m%d').toordinal()], dtype='int'))
    eq_ix.append(n_im)

    # Make all dates ordinal
    ord_eq = np.array([eq.toordinal() for eq in eq_dt]) - dates[0].toordinal()
    date_ord = np.array([x.toordinal() for x in dates]) - dates[0].toordinal()

def read_eq_list(eq_listfile):
    eqdates = []
    parameters = []
    f = open(eq_listfile)
    line = f.readline()

    while line:
        if line[0].isnumeric():
            if len(line.split()) == 2:
                if line.split()[1] == 'X':
                    line = f.readline()
                else:
                    eqdates.append(str(line.split()[0]))
                    parameters.append(str(line.split()[1]).upper())
                    line = f.readline()
            else:
                eqdates.append(str(line.split()[0]))
                parameters.append('CRP')
                line = f.readline()
        else:
            line = f.readline()
            continue
    sort_ix = np.array(sorted(range(len(eqdates)), key=eqdates.__getitem__))
    eqdates.sort()
    parameters = np.array(parameters)[sort_ix].tolist()

    return eqdates, parameters

def reference_disp(data, reffile):
    global refx1, refx2, refy1, refy2
    if reffile:
        with open(reffile, "r") as f:
            refarea = f.read().split()[0]  # str, x1/x2/y1/y2
        refx1, refx2, refy1, refy2 = [int(s) for s in re.split('[:/]', refarea)]

        # Reference all data to reference area through static offset [This makes the reference pixel only ever have a displacement of 0]
        ref = np.nanmean(data[:, refy1:refy2, refx1:refx2], axis=(2, 1)).reshape(data.shape[0], 1, 1)
        # Check that one of the refs is not all nan. If so, increase ref area
        if np.isnan(ref).any():
            print('NaN Value for reference for {} dates. Increasing ref area kernel'.format(np.sum(np.isnan(ref))))
            while np.isnan(ref).any():
                refx1 -= 1
                refx2 += 1
                refy1 -= 1
                refy2 += 1
                ref = np.nanmean(data[:, refy1:refy2, refx1:refx2], axis=(2, 1)).reshape(data.shape[0], 1, 1)

        print('Reference area ({}:{}, {}:{})'.format(refx1, refx2, refy1, refy2))
    else:
        ref = 0
        refx1, refx2, refy1, refy2 = 0, 0, 0, 0

    return ref

def temporal_filter(cum):
    global ixs_dict, dt_cum, filterdates, filtwidth_yr, cum_lpt, n_its, filt_std
    """
    Apply a low pass temporal filter, and remove outliers.
     Iterate until no outliers left so filter is not distorted
     by outliers, then on the final pass, remove outliers between
     the original data and final iteration filter
    """

    print('Filtering Temporally for outliers > {} std'.format(outlier_thresh))
    print('Using {} parallel processing'.format(n_para))

    dt_cum = date_ord / 365.25  # Set dates in terms of years
    filtwidth_yr = dt_cum[-1] / (n_im - 1) * 3  # Set the filter width based on n * average epoch seperation
    filterdates = np.linspace(0, n_im - 1, n_im, dtype='int').tolist()
    n_its = 1
    ixs_dict = get_filter_dates(dt_cum, filtwidth_yr, filterdates)

    # Find and remove any outliers above filtered std threshold
    if args.ransac:
        cum, filt_std = find_outliers_RANSAC()
    else:
        cum_lpt, outlier = find_outliers()

        # Iterate until all outliers are removed
        n_its = 1

        # Replace all outliers with filtered values
        cum[outlier] = cum_lpt[outlier]
        all_outliers = outlier

        # Run iterations
        while len(outlier) > 0:
            n_its += 1
            if n_its <= args.max_its:
                print('Running Iteration {}/{}'.format(n_its, args.max_its))
                cum_lpt, outlier = find_outliers()
                # Replace all outliers with filtered values
                cum[outlier] = cum_lpt[outlier]
                all_outliers = np.unique(np.concatenate((all_outliers, outlier), axis=1), axis=1)
            else:
                break

        # Reload original data, nan identified outliers
        cum = np.array(data['cum']) - ref
        if args.apply_mask:
            print('Applying Mask to reloaded data')
            mask = io_lib.read_img(maskfile, length, width)
            maskx, masky = np.where(mask == 0)
            cum[:, maskx, masky] = np.nan

        if args.replace_outliers:
            print('Replacing Outliers')
            cum[all_outliers[0], all_outliers[1], all_outliers[2]] = cum_lpt[all_outliers[0], all_outliers[1], all_outliers[2]]
        else:
            print('Nanning Outliers')
            cum[all_outliers[0], all_outliers[1], all_outliers[2]] = np.nan # Nan the outliers. Better data handling, but causing problems

        print('Finding moving stddev')
        filt_std = np.ones(cum.shape) * np.nan
        filterdates = np.linspace(0, n_im - 1, n_im, dtype='int')
        valid = np.where(~np.isnan(data['vel']))
        diff = cum[:, valid[0], valid[1]] - cum_lpt[:, valid[0], valid[1]]

        for i in filterdates:
            if np.mod(i, 10) == 0:
                print("  {0:3}/{1:3}th image...".format(i, n_im), flush=True)

            filt_std[i, valid[0], valid[1]] = np.nanstd(diff[ixs_dict[i], :], axis=0)

def std_filters(i):
    date = filterdates[i]
    filt_std = np.zeros((length, width)) * np.nan
    with warnings.catch_warnings():  # To silence warning by zero division
            warnings.simplefilter('ignore', RuntimeWarning)
            # Just search valid pixels to speed up
            std_window = diff[ixs_dict[date],:,:]
            filt_std[valid[0], valid[1]] = np.nanstd(std_window[:, valid[0], valid[1]], axis=0)

    return filt_std

def find_outliers_RANSAC():
    global diff, cum_lpt, filt_std, cum
    filt_std = np.zeros((n_im, length, width)) * np.nan

    # Run Low-Pass filter on displacement data
    if n_para > 1 and len(filterdates) > 20:
        p = q.Pool(n_para)
        cum_lpt = np.array(p.map(lpt_filter, range(n_im)), dtype=np.float32)
        p.close()
    else:
        cum_lpt = np.zeros(cum.shape) * np.nan
        for ii in range(n_im):
            cum_lpt[ii, :, :] = lpt_filter(filterdates)

    # Find STD
    diff = cum - cum_lpt  # Difference between data and filtered data
    print('Finding std of residuals')
    if n_para > 1 and len(filterdates) > 20:
        p = q.Pool(n_para)
        filt_std = np.array(p.map(std_filters, range(n_im)), dtype=np.float32)
        p.close()
    else:
        filt_std = np.zeros((n_im, length, width)) * np.nan
        for i in range(n_im):
            filt_std[i, :, :] = std_filters(i)

    print('Deoutliering using RANSAC')

    if n_para > 1 and len(filterdates) > 20:
        p = q.Pool(n_para)
        deoutliered = np.array(p.map(run_RANSAC, range(n_valid)), dtype=np.float32)
        p.close()
        for ii in range(n_valid):
            cum[:, valid[0][ii], valid[1][ii]] = deoutliered[ii, :]
    else:
        for ii in range(n_valid):
            cum[:, valid[0][ii], valid[1][ii]] = run_RANSAC(ii)

    # Rerun Lowpass filter on deoutliered data (as massive outliers will have distorted the original filter)
    print('Rerun temporal filter on deoutliered data')
    if n_para > 1 and len(filterdates) > 20:
        p = q.Pool(n_para)
        cum_lpt = np.array(p.map(lpt_filter, range(n_im)), dtype=np.float32)
        p.close()
    else:
        cum_lpt = lpt_filter(filterdates)

    # Reload the original data
    cum = np.array(data['cum']) - ref

    # Find STD
    diff = cum - cum_lpt  # Difference between original and filtered, deoutliered data
    print('Finding std of residuals')
    if n_para > 1 and len(filterdates) > 20:
        p = q.Pool(n_para)
        filt_std = np.array(p.map(std_filters, range(n_im)), dtype=np.float32)
        p.close()
    else:
        filt_std = np.zeros((n_im, length, width)) * np.nan
        for i in range(n_im):
            filt_std[i, :, :] = std_filters(i)

    # Find location of outliers
    outlier = np.where(abs(diff) > (outlier_thresh * filt_std))
    print('\n{} outliers identified ({:.1f}%)\n'.format(len(outlier[0]), (len(outlier[0]) / np.sum(~np.isnan(cum.flatten()))) * 100))

    # x_pix = valid[1][15001]
    # y_pix = valid[0][15001]
    # x_pix = 347
    # y_pix = 492
    # fig=plt.figure(figsize=(12,24))
    # plt.scatter(np.array(dates), cum[:, y_pix, x_pix], s=4, c='b', label='Inliers')
    # plt.scatter(np.array(dates), cum[:, y_pix, x_pix], s=4, c='r', label='Outliers')
    # plt.plot(np.array(dates), cum_lpt[:, y_pix, x_pix], c='g', label='Filters')
    # plt.plot(np.array(dates), cum_lpt[:, y_pix, x_pix] + filt_std[:, y_pix, x_pix], c='r', label='1 STD')
    # plt.plot(np.array(dates), cum_lpt[:, y_pix, x_pix] + outlier_thresh * filt_std[:, y_pix, x_pix], c='b', label='Outlier Thresh')
    # plt.plot(np.array(dates), cum_lpt[:, y_pix, x_pix] - filt_std[:, y_pix, x_pix], c='r')
    # plt.plot(np.array(dates), cum_lpt[:, y_pix, x_pix] - outlier_thresh * filt_std[:, y_pix, x_pix], c='b')
    # plt.legend()

    # Deal with outlier data
    if args.replace_outliers:
        print('Replacing Outliers')
        cum[outlier] = cum_lpt[outlier]
    else:
        print('Nanning Outliers')
        cum[outlier] = np.nan

    # plt.scatter(np.array(dates), cum[:, y_pix, x_pix], s=4, c='b')
    # plt.savefig(os.path.join(outdir, 'finRANSAC{}.png'.format(15000)))
    # plt.close()
    # print(os.path.join(outdir, 'finRANSAC{}.png'.format(15000)))

    return cum, filt_std

def run_RANSAC(ii):
    if np.mod(ii, 5000) == 0:
        print('{}/{} RANSACed....'.format(ii, n_valid))
    # Find non-nan data
    disp = cum[:, valid[0][ii], valid[1][ii]]
    keep = np.where(~np.isnan(disp))[0]
    # Select difference between filtered and original data
    resid = diff[:, valid[0][ii], valid[1][ii]]
    filtered = cum_lpt[:, valid[0][ii], valid[1][ii]]
    # Set RANSAC threshold based off median std
    limits = outlier_thresh * np.nanmedian(filt_std[:, valid[0][ii], valid[1][ii]])
    reg = RANSACRegressor(residual_threshold=limits).fit(date_ord[keep].reshape((-1,1)),resid[keep].reshape((-1,1)))
    inliers = reg.inlier_mask_
    outliers = np.logical_not(reg.inlier_mask_)

    # Interpolate filtered values over outliers
    interp = CubicSpline(date_ord[keep[inliers]],filtered[keep[inliers]])
    filtered_outliers = interp(date_ord[keep[outliers]])

    # yvals = reg.predict(date_ord.reshape((-1,1)))
    # if np.mod(ii, 5000) == 0:
    #     fig=plt.figure(figsize=(12,24))
    #     ax=fig.add_subplot(2,1,1)
    #     ax.scatter(np.array(dates)[inliers], disp[inliers], s=2, label='Inlier {}'.format(ii))
    #     ax.scatter(np.array(dates)[outliers], disp[outliers], s=2, label='Outlier {}'.format(ii))
    #     ax.scatter(np.array(dates)[outliers], filtered_outliers, s=10, c='r', label='Replaced {}'.format(ii))
    #     ax.plot(dates, filtered, c='g',label='Fitted Vel')
    #     ax.plot(dates, filtered + std, c='r',label='Fitted STD')
    #     ax.plot(dates, filtered - std, c='r')
    #     ax.plot(dates, filtered + limits, c='b',label='Outlier Thresh')
    #     ax.plot(dates, filtered - limits, c='b')
    #     ax.scatter(np.array(dates)[outliers], filtered_outliers, s=10, c='r', label='Replaced {}'.format(ii))
    #     ax.legend()
    #     ax=fig.add_subplot(2,1,2)
    #     ax.scatter(np.array(dates)[inliers], resid[inliers], s=2, label='Inlier {}'.format(ii))
    #     ax.scatter(np.array(dates)[outliers], resid[outliers], s=2, label='Outlier {}'.format(ii))
    #     ax.plot(dates, yvals, label='RANSAC')
    #     ax.plot(dates, yvals + std, c='r', label='1x std')
    #     ax.plot(dates, yvals - std, c='r')
    #     ax.plot(dates, yvals + limits, c='b', label='3*std')
    #     ax.plot(dates, yvals - limits, c='b')
    #     plt.savefig(os.path.join(outdir, 'filtRANSAC{}.png'.format(ii)))
    #     plt.close()
    #     print(os.path.join(outdir, 'filtRANSAC{}.png'.format(ii)))

    disp[keep[outliers]] = filtered_outliers

    return disp

def find_outliers():
    filt_std = np.zeros((n_im, length, width)) * np.nan

    if n_para > 1 and len(filterdates) > 20:
        p = q.Pool(n_para)
        cum_lpt = np.array(p.map(lpt_filter, range(n_im)), dtype=np.float32)
        p.close()
    else:
        lpt_filter(filterdates)

    # Find STD
    diff = cum - cum_lpt  # Difference between data and filtered data
    filt_std = np.zeros((n_im, length, width)) * np.nan
    for i in filterdates:
        with warnings.catch_warnings():  # To silence warning by zero division
            warnings.simplefilter('ignore', RuntimeWarning)
            # Just search valid pixels to speed up
            std_window = diff[ixs_dict[i],:,:]
            filt_std[i, valid[0], valid[1]] = np.nanstd(std_window[:, valid[0], valid[1]], axis=0)

    # Find location of outliers
    outlier = np.where(abs(diff) > (outlier_thresh * filt_std))

    print('\n{} outliers identified\n'.format(len(outlier[0])))

    return cum_lpt, outlier

def lpt_filter(i):

    # for i in datelist:
    if np.mod(i + 1, 10) == 0:
        print("  {0:3}/{1:3}th image...".format(i + 1, len(filterdates)), flush=True)

    # Find time difference between filter date and other dates
    time_diff_sq = (dt_cum[i] - dt_cum) ** 2

    # Get data limits (ie only dates within filtwidth_yr)
    ixs = ixs_dict[i]

    weight_factor = np.tile(np.exp(-time_diff_sq[ixs] / 2 / filtwidth_yr ** 2)[:, np.newaxis, np.newaxis], (1, length, width))  # len(ixs), length, width

    # Take into account nan in cum
    weight_factor = weight_factor * (~np.isnan(cum[ixs, :, :]))

    # Normalize weight
    with warnings.catch_warnings():  # To silence warning by zero division
        warnings.simplefilter('ignore', RuntimeWarning)
        weight_factor = weight_factor / np.sum(weight_factor, axis=0)

    # Find Low-Pass Temporal displacements
    lpt = np.nansum(cum[ixs, :, :] * weight_factor, axis=0)
    lpt[np.where(np.isnan(cum[i, :, :]))] = np.nan

    return lpt

def get_filter_dates(dt_cum, filtwidth_yr, filterdates):
    """
    For each date, find the dates needed to filter it. Don't cross over any earthquakes when generating the filter
    """
    ixs_dict = {}
    for i in filterdates:
        # Find date range needed to filter each date
        time_diff_sq = (dt_cum[i] - dt_cum) ** 2

        # Apply data limits to filter
        # Limit within filtwidth_yr * 8
        ixs = time_diff_sq < filtwidth_yr * 8
        # Limit to between Earthquakes (if earthquakes are in the TS)
        if ord_eq.size > 0:
            if date_ord[i] < ord_eq[0]:
                next_eq = ord_eq[0]
                prev_eq = date_ord[0] - 1
            elif date_ord[i] < ord_eq[-1]:
                next_eq = ord_eq[np.where(date_ord[i] < ord_eq)[0][0]]
                prev_eq = ord_eq[np.where(date_ord[i] < ord_eq)[0][0] - 1]
            else:
                prev_eq = ord_eq[-1]
                next_eq = date_ord[-1] + 1
        else:
            prev_eq = -1
            next_eq = date_ord[-1] + 1


        ixs = np.logical_and(ixs, np.logical_and(date_ord > prev_eq, date_ord < next_eq))

        ixs_dict[i] = ixs
    return ixs_dict

def fit_velocities():
    global pcst, Q, model, errors

    # Create VCM of observables (no c)
    Q = np.eye(n_im)
    if args.use_weights:
        sills = calc_semivariogram()
        # Create weight matrix (inverse of VCM, faster than np.linalg.inv)
        np.fill_diagonal(Q, 1 / sills)

    # Define post-seismic constant
    pcst = 1 / args.tau
    n_variables = 2 + n_eq * 3

    _n_para = n_para if n_para < 25 else 25 # Diminishing returns after this, empirically
    print('\nVelocity fitting on {} Cores'.format(_n_para))
    if _n_para > 1 and n_valid > 100:
        p = q.Pool(_n_para)
        results = np.array(p.map(fit_pixel_velocities, range(n_valid)), dtype="object")
        p.close()
        model = np.concatenate(results[:,0]).reshape(n_valid, n_variables)
        errors = np.concatenate(results[:,1]).reshape(n_valid, n_variables + 2)
    else:
        model = np.zeros((n_valid, n_variables))
        errors = np.zeros((n_valid, n_variables + 2))
        for ii in range(n_valid):
            model[ii, :], errors[ii, :] = fit_pixel_velocities(ii)

def calc_semivariogram():
    global XX, YY, mask_pix
    # Get range and aximuth pixel spacing
    param13 =  os.path.join(infodir, '13parameters.txt')
    pixel_spacing_a = float(io_lib.get_param_par(param13, 'pixel_spacing_a'))
    pixel_spacing_r = float(io_lib.get_param_par(param13, 'pixel_spacing_r'))

    # Rounding as otherwise phantom decimals appearing that makes some Lats too long
    Lat = np.arange(0, np.round(length * pixel_spacing_r, 5), pixel_spacing_r)
    Lon = np.arange(0, np.round(width * pixel_spacing_a, 5), pixel_spacing_a)

    XX, YY = np.meshgrid(Lon, Lat)
    XX = XX.flatten()
    YY = YY.flatten()

    mask = io_lib.read_img(maskfile, length, width)
    mask_pix = np.where(mask.flatten() == 0)

    print('\nCalculating semi-variograms of epoch displacements')
    print('Using {} processing'.format(n_para))
    if n_para > 1 and n_valid > 100:
        p = q.Pool(n_para)
        sills = np.array(p.map(calc_epoch_semivariogram, range(n_im)), dtype="object")
        p.close()
    else:
        sills = np.zeros((n_im, 1))
        for ii in range(1, n_im):
            sills[ii] = calc_epoch_semivariogram(ii)

    sills[0] = np.nanmean(sills[1:]) # As first epoch is 0

    return sills

def calc_epoch_semivariogram(ii):
    if ii == 0:
        sill = 0 # Reference image
    else:
        begin_semi = time.time()

        # Find semivariogram of incremental displacements
        epoch = (cum[ii, :, :] - cum[ii - 1, :, :]).flatten()
        # Nan mask pixels
        epoch[mask_pix] = np.nan
        # Mask out any displacement of > lambda, as coseismic or noise
        epoch[abs(epoch) > 55.6] = np.nan

        # Reference to it's own median
        epoch -= np.nanmedian(epoch)

        # Drop all nan data
        xdist = XX[~np.isnan(epoch)]
        ydist = YY[~np.isnan(epoch)]
        epoch = epoch[~np.isnan(epoch)]

        # calc from lmfit
        mod = Model(spherical)
        medians = np.array([])
        bincenters = np.array([])
        stds = np.array([])

        # Find random pairings of pixels to check
        # Number of random checks
        n_pix = int(1e6)

        pix_1 = np.array([])
        pix_2 = np.array([])

        # Going to look at n_pix pairs. Only iterate 5 times. Life is short
        its = 0
        while pix_1.shape[0] < n_pix and its < 5:
            its += 1
            # Create n_pix random selection of data points (Random selection with replacement)
            # Work out too many in case we need to remove duplicates
            pix_1 = np.concatenate([pix_1, np.random.choice(np.arange(epoch.shape[0]), n_pix * 2)])
            pix_2 = np.concatenate([pix_2, np.random.choice(np.arange(epoch.shape[0]), n_pix * 2)])

            # Find where the same pixel is selected twice
            duplicate = np.where(pix_1 == pix_2)[0]
            np.delete(pix_1, duplicate)
            np.delete(pix_2, duplicate)

            # Drop duplicate pairings
            unique_pix = np.unique(np.vstack([pix_1, pix_2]).T, axis=0)
            pix_1 = unique_pix[:, 0]
            pix_2 = unique_pix[:, 1]

        # In case of early ending
        if n_pix > len(pix_1):
            n_pix = len(pix_1)

        # Trim to n_pix, and create integer array
        pix_1 = pix_1[:n_pix].astype('int')
        pix_2 = pix_2[:n_pix].astype('int')

        # Calculate distances between random points
        dists = np.sqrt(((xdist[pix_1] - xdist[pix_2]) ** 2) + ((ydist[pix_1] - ydist[pix_2]) ** 2))
        # Calculate squared difference between random points
        vals = abs((epoch[pix_1] - epoch[pix_2])) ** 2

        medians, binedges = stats.binned_statistic(dists, vals, 'median', bins=100)[:-1]
        stds = stats.binned_statistic(dists, vals, 'std', bins=100)[0]
        bincenters = (binedges[0:-1] + binedges[1:]) / 2

        try:
            mod.set_param_hint('p', value=np.nanmax(medians))  # guess maximum variance
            mod.set_param_hint('n', value=0)  # guess 0
            mod.set_param_hint('r', value=bincenters[len(bincenters)//2])  # guess mid point distance
            sigma = stds + np.power(bincenters / max(bincenters), 2)
            result = mod.fit(medians, d=bincenters, weights=sigma)
        except:
            # Try smaller ranges
            length = len(bincenters)
            try:
                bincenters = bincenters[:int(length * 3 / 4)]
                stds = stds[:int(length * 3 / 4)]
                medians = medians[:int(length * 3 / 4)]
                sigma = stds + np.power(bincenters / max(bincenters), 3)
                result = mod.fit(medians, d=bincenters, weights=sigma)
            except:
                bincenters = bincenters[:int(length / 2)]
                stds = stds[:int(length / 2)]
                medians = medians[:int(length / 2)]
                sigma = stds + np.power(bincenters / max(bincenters), 3)
                result = mod.fit(medians, d=bincenters, weights=sigma)

        # Print Sill (ie variance)
        sill = result.best_values['p']

        if np.mod(ii + 1, 10) == 0:
            print('\t{}/{}\tSill: {:.2f} ({:.2e}\tpairs processed in {:.1f} seconds)'.format(ii + 1, n_im, sill, n_pix, time.time() - begin_semi))

    return sill

def spherical(d, p, n, r):
    """
    Compute spherical variogram model
    @param d: 1D distance array
    @param p: partial sill
    @param n: nugget
    @param r: range
    @return: spherical variogram model
    """
    if r>d.max():
        r=d.max()-1
    return np.where(d > r, p + n, p * (3/2 * d/r - 1/2 * d**3 / r**3) + n)

def fit_pixel_velocities(ii):

    if np.mod(ii, 10000) == 0:
        print('{}/{}'.format(ii,n_valid))

    # Fit Pre- and Post-Seismic Linear velocities, coseismic offset, postseismic relaxation and referencing offset
    disp = cum[:, valid[0][ii], valid[1][ii]]
    noNanPix = ~np.isnan(disp)
    disp = disp[noNanPix]
    # Intercept (reference term), Pre-Seismic Velocity, [[C]oseismic-offset, log [R]elaxation, [P]ost-seismic linear velocity]
    truemodel = np.zeros((2 + n_eq * 3))
    inverr = np.zeros((2 + n_eq * 3))
    invert_ix = [0, 1]

    for ix, param in enumerate(eq_params):
        if 'C' in param:
            invert_ix.append(2 + ix * 3)
        if 'R' in param:
            invert_ix.append(3 + ix * 3)
        if 'P' in param:
            invert_ix.append(4 + ix * 3)
    invert_ix.sort()

    G = np.zeros([n_im, 2 + n_eq * 3])
    G[:, 0] = 1
    # G[:eq_ix[0], 1] = date_ord[:eq_ix[0]]
    G[:, 1] = date_ord # Makes the pre-seismic rate the long-term rate. Postseimic linear is summed with this

    daily_rates = [1]

    for ee in range(0, n_eq):
        # Create Gmatrix for coseismic, a-value, postseismic
        G[eq_ix[ee]:eq_ix[ee + 1], 2 + ee * 3] = 1
        G[eq_ix[ee]:eq_ix[ee + 1], 3 + ee * 3] = np.log(1 + pcst * (date_ord[eq_ix[ee]:eq_ix[ee + 1]] - ord_eq[ee]))
        G[eq_ix[ee]:eq_ix[ee + 1], 4 + ee * 3] = date_ord[eq_ix[ee]:eq_ix[ee + 1]] - ord_eq[ee]
        daily_rates.append(4 + ee * 3)

    # G = G[np.ix_(noNanPix, invert_ix)]
    G = G[noNanPix, :]
    singular = np.where((G == 0).all(axis=0))[0].tolist()
    invert_ix = list(set(invert_ix) - set(singular))
    G = G[:, invert_ix]

    # Weight matrix (inverse of VCM)
    # W = np.linalg.inv(Q) # Too slow. Faster to do 1/sill before this
    W = Q[np.ix_(noNanPix, noNanPix)].copy()

    # Calculate VCM of inverted model parameters
    invVCM= np.linalg.inv(np.dot(np.dot(G.T, W), G))

    model = np.matmul(invVCM, np.matmul(G.T, disp))

    # Invert for modelled displacement
    invvel = np.matmul(G, model)

    # Calculate inversion parameter standard errors and root mean square error
    rms=np.dot(np.dot((invvel-disp).T, W),(invvel-disp))
    inverr[invert_ix]=np.sqrt(np.diag(invVCM) * rms / np.sum(noNanPix))
    rms=np.sqrt(rms / n_im)

    # Find standard deviations of the velocity residuals
    std = np.sqrt((1 / n_im) * np.sum((disp - invvel) ** 2))

    # Find 'True' Parameters
    truemodel[invert_ix] = model

    Gcos = np.zeros((2 * n_eq, 2 + n_eq * 3))
    Gcos[:, 0] = 1
    Gcos[:, 1] = [ord for ord in ord_eq for _ in range(2)]

    for ee in range(n_eq):
        # Fix postseismic
        truemodel[4 + ee * 3] = truemodel[4 + ee * 3] + truemodel[1]
        eq = ee * 2 # eq = index of immediately before eq, eq+1 = index of immediately after

        Gcos[eq + 1, 2 + ee * 3] = 1 # Coseismic (for post eq1)
        Gcos[eq + 1, 3 + ee * 3] = np.log(1 + pcst * (ord_eq[ee] - ord_eq[ee])) # Avalue (for post eq1)
        Gcos[eq + 1, 4 + ee * 3] = ord_eq[ee] - ord_eq[ee] # Postseismic (for post eq1)
        if ee < (n_eq - 1):
            Gcos[eq + 2, 2 + ee * 3] = 1 # Coseismic (for pre-eq2)
            Gcos[eq + 2, 3 + ee * 3] = np.log(1 + pcst * (ord_eq[ee + 1] - ord_eq[ee])) # Avalue (for pre eq2)
            Gcos[eq + 2, 4 + ee * 3] = ord_eq[ee + 1] - ord_eq[ee] # Postseismic (for pre eq2)

    coseismic = np.matmul(Gcos[:, invert_ix], model)
    truemodel[2:2+n_eq*3:3] = coseismic[1:n_eq*2:2] - coseismic[0:n_eq*2:2]

    # Convert mm/day to mm/yr
    for dd in daily_rates:
        truemodel[dd] *= 365.25
        inverr[dd] *= 365.25

    inverr = np.append(inverr, [rms, std])

    return truemodel, inverr

def plot_timeseries(dates, disp, invvel, ii, x, y):
    plt.scatter(dates, disp, s=2, label='{}'.format(ii))
    plt.plot(dates, invvel, label='{}'.format(ii))
    plt.title('({},{})'.format(x,y))
    plt.legend()
    for ii in range(0, n_eq):
        plt.axvline(x=eq_dt[ii], color="grey", linestyle="--")
    plt.savefig(os.path.join(outdir, '{}.png'.format(ii)))

def set_file_names():

    names = []
    titles = []

    for ext in ['', '_err']:
        names = names + ['intercept{}'.format(ext), 'pre_vel{}'.format(ext)]
        for n in range(n_eq):
            names = names + ['coseismic{}{}'.format(eq_dates[n], ext), 'a_value{}{}'.format(eq_dates[n], ext), 'post_vel{}{}'.format(eq_dates[n], ext)]
    n_vel = len(names) / 2

    names = names + ['rms', 'vstd']

    for ext in ['', ' Error']:
        titles = titles + ['Velocity Intercept{} (mm/yr)'.format(ext), 'Preseismic Linear Velocity{} (mm/yr)'.format(ext)]
        for n in range(n_eq):
            titles = titles + ['Coseismic Displacement{} {} (mm)'.format(ext, eq_dates[n]), 'Postseismic Relaxation A-value{} {} (mm)'.format(ext, eq_dates[n]), 'Postseismic Linear Velocity Component{} {} (mm/yr)'.format(ext, eq_dates[n])]
    titles = titles + ['RMSE (mm/yr)', 'Velocity Std (mm/yr)']

    return names, titles, n_vel

def write_outputs():

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # Copy already produced files
    metafiles = ['coh_avg', 'n_unw', 'vstd', 'maxTlen', 'n_gap', 'stc', 'n_ifg_noloop', 'n_loop_err', 'resid_rms', 'slc.mli', 'hgt']

    for meta in metafiles:
        if os.path.exists(os.path.join(resultdir, meta)):
            shutil.copy(os.path.join(resultdir, meta), os.path.join(metadir, meta))
        if os.path.exists(os.path.join(resultdir, meta + '.png')):
            shutil.copy(os.path.join(resultdir, meta + '.png'), os.path.join(metadir, meta + '.png'))

    results = np.hstack([model, errors])

    names, titles, n_vel = set_file_names()

    print('Writing Outputs to file and png')

    if mask_final:
        mask = io_lib.read_img(maskfile, length, width)
        mask_pix = np.where(mask == 0)

    gridResults = np.zeros((len(names), length, width), dtype=np.float32) * np.nan

    for ix, name in enumerate(names):
        filename = os.path.join(outdir, name)
        pngname = '{}.png'.format(filename)
        gridResults[ix, valid[0], valid[1]] = results[:, ix]
        gridResults[ix, :, :].tofile(filename)

        vmax = np.nanpercentile(gridResults[ix, :, :], 95)
        if ix >= n_vel:
            vmin = 0
            cmap = 'viridis_r'
        else:
            vmin = np.nanpercentile(gridResults[ix, :, :], 5)
            vmin = -np.nanmax([abs(vmin), abs(vmax)])
            vmax = np.nanmax([abs(vmin), abs(vmax)])
            cmap = SCM.roma.reversed()
        plot_lib.make_im_png(gridResults[ix, :, :], pngname, cmap, titles[ix], vmin, vmax)

        if mask_final:
            maskpngname = '{}.mskd.png'.format(filename)
            mask_data = gridResults[ix, :, :]
            mask_data[mask_pix[0], mask_pix[1]] = np.nan

            vmax = np.nanpercentile(mask_data, 95)
            if ix < n_vel:
                vmin = np.nanpercentile(mask_data, 5)
                vmin = -np.nanmax([abs(vmin), abs(vmax)])
                vmax = np.nanmax([abs(vmin), abs(vmax)])
            plot_lib.make_im_png(mask_data, maskpngname, cmap, titles[ix], vmin, vmax)

    write_h5(gridResults, data)

def write_h5(gridResults, data):
    # Currently can only handle writing 1 earthquake to h5

    print('\nWriting to HDF5 file...')
    compress = 'gzip'
    if os.path.exists(outh5file):
        os.remove(outh5file)
    cumh5 = h5.File(outh5file, 'w')

    # Copy data from original cum.h5 that hasn't changed
    cumh5.create_dataset('imdates', data=data['imdates'])
    cumh5.create_dataset('corner_lat', data=data['corner_lat'])
    cumh5.create_dataset('corner_lon', data=data['corner_lon'])
    cumh5.create_dataset('post_lat', data=data['post_lat'])
    cumh5.create_dataset('post_lon', data=data['post_lon'])
    cumh5.create_dataset('gap', data=data['gap'])
    if reffile:
        cumh5.create_dataset('refarea', data='{}:{}/{}:{}'.format(refx1, refx2, refy1, refy2))
    else:
        cumh5.create_dataset('refarea', data=data['refarea'])

    # Add previously calculated indicies to the cum.h5
    indices = ['coh_avg', 'hgt', 'n_loop_err', 'n_unw', 'slc.mli',
               'maxTlen', 'n_gap', 'n_ifg_noloop', 'resid_rms']

    for index in indices:
        file = os.path.join(resultdir, index)
        if os.path.exists(file):
            datafile = io_lib.read_img(file, length, width)
            cumh5.create_dataset(index, data=datafile, compression=compress)
        else:
            print('  {} not exist in results dir. Skip'.format(index))

    LOSvecs = ['E.geo', 'N.geo', 'U.geo']
    for LOSvec in LOSvecs:
        file = os.path.join(ifgdir, LOSvec)
        if os.path.exists(file):
            datafile = io_lib.read_img(file, length, width)
            cumh5.create_dataset(LOSvec, data=datafile, compression=compress)
        else:
            print('  {} not exist in GEOCml dir. Skip'.format(LOSvec))

    # Add new data to h5
    cumh5.create_dataset('eqdates', data=[np.int32(eqd) for eqd in eq_dates])
    cumh5.create_dataset('eqparams', data=[str(param) for param in eq_params])
    cumh5.create_dataset('cum', data=cum, compression=compress)
    cumh5.create_dataset('vel', data=data['vel'])
    cumh5.create_dataset('vintercept', data=gridResults[0], compression=compress)
    cumh5.create_dataset('prevel', data=gridResults[1], compression=compress)
    for nn in range(n_eq):
        cumh5.create_dataset('{} coseismic'.format(eq_dates[nn]), data=gridResults[2 + nn * 3], compression=compress)
        cumh5.create_dataset('{} avalue'.format(eq_dates[nn]), data=gridResults[3 + nn * 3], compression=compress)
        cumh5.create_dataset('{} postvel'.format(eq_dates[nn]), data=gridResults[4 + nn * 3], compression=compress)


    cumh5.close()

def even_split(a, n):
    """ Divide a list, a, in to n even parts"""
    n = min(n, len(a)) # to avoid empty lists
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def main():
    start()
    init_args()
    set_input_output()
    load_data()

    if args.deoutlier:
        # Remove outliers from image displacements to improve velocity fitting
        temporal_filter(cum)

    # Fit velocities
    fit_velocities()

    # Write Outputs
    write_outputs()


    finish()

if __name__ == "__main__":
    main()
