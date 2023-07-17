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
    parser.add_argument('-r', dest='ref_file', default='130ref.txt', help='txt file containing reference area')
    parser.add_argument('-m', dest='mask_file', default='mask', help='mask file to apply to velocities')
    parser.add_argument('-e', dest='eq_list', default=None, help='Text file containing the dates of the earthquakes to be fitted')
    parser.add_argument('-s', dest='outlier_thre', default=3, type=float, help='StdDev threshold used to remove outliers')
    parser.add_argument('-c', dest='detect_thre', default=1, type=float, help="Coseismic detection threshold (n * vstd)")
    parser.add_argument('--n_para', dest='n_para', default=False, type=int, help='number of parallel processing')
    parser.add_argument('--tau', dest='tau', default=6, help='Post-seismic relaxation time (days)')
    parser.add_argument('--max_its', dest='max_its', default=5, type=int, help='Maximum number of iterations for temporal filter')
    parser.add_argument('--nofilter', dest='deoutlier', default=True, action='store_false', help="Don't do any temporal filtering")
    parser.add_argument('--applymask', dest='apply_mask', default=False, action='store_true', help="Apply mask to cum data before processing")
    parser.add_argument('--RANSAC', dest='ransac', default=False, action='store_true', help="Deoutlier with RANSAC algorithm")

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
    print('Output directory: {}\n'.format(os.path.relpath(outdir)))

def set_input_output():
    global tsadir, infodir, resultdir, outdir, ifgdir
    global h5file, reffile, maskfile, eqfile, outh5file
    global q, outlier_thresh, mask_final

    # define input directories
    ifgdir = os.path.abspath(os.path.join(args.frame_dir, args.unw_dir))
    tsadir = os.path.abspath(os.path.join(args.frame_dir, args.ts_dir))
    infodir = os.path.join(tsadir, 'info')
    resultdir = os.path.join(tsadir, 'results')
    outdir = os.path.join(resultdir, 'seismic_vels')

    # define input files
    h5file = os.path.join(tsadir, args.h5_file)
    outh5file = os.path.join(tsadir, outdir, 'cum.h5')

    # If no reffile defined, sarch for 130ref, then 13ref, in this folder and infodir
    reffile = os.path.join(tsadir, args.ref_file)
    if not os.path.exists(reffile):
        reffile = os.path.join(infodir, args.ref_file)
        if not os.path.exists(reffile):
            if args.ref_file == '130ref.txt':
                # Seach for 13ref.txt
                reffile = os.path.join(tsadir, '13ref.txt')
                if not os.path.exists(reffile):
                    reffile = os.path.join(infodir, '13ref.txt')
                    if not os.path.exists(reffile):
                        print('\nNo reffile 130ref.txt or 13ref.txt found! No referencing occuring')
                        reffile = []
            else:
                print('\nNo reffile {} found! No referencing occuring'.format(args.ref_file))
                reffile = []

    maskfile = os.path.join(resultdir, 'mask')
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
    global width, length, data, n_im, cum, dates, length, width, n_para, eq_dates, n_eq, eq_dt, eq_ix, ord_eq, date_ord, eq_dates, valid, n_valid, ref

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
        shutil.copy(maskfile, os.path.join(outdir, 'mask'))
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

    if n_para > 1:
        print('Using {} parallel processing'.format(n_para))
    else:
        print('Using no parallel processing')

    ## Sort dates
    # Get list of earthquake dates and index
    eq_dates = io_lib.read_ifg_list(eqfile)
    eq_dates.sort()
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

        # Reload original data, replace identified outliers with final filtered values
        cum = np.array(data['cum']) - ref
        if args.apply_mask:
            print('Applying Mask to reloaded data')
            mask = io_lib.read_img(maskfile, length, width)
            maskx, masky = np.where(mask == 0)
            cum[:, maskx, masky] = np.nan

        cum[all_outliers[0], all_outliers[1], all_outliers[2]] = cum_lpt[all_outliers[0], all_outliers[1], all_outliers[2]]

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
        cum_lpt = lpt_filter(filterdates)

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
    print('\n{} outliers identified\n'.format(len(outlier[0])))

    x_pix = valid[1][15001]
    y_pix = valid[0][15001]
    x_pix = 347
    y_pix = 492
    fig=plt.figure(figsize=(12,24))
    plt.scatter(np.array(dates), cum[:, y_pix, x_pix], s=4, c='b', label='Inliers')
    plt.scatter(np.array(dates), cum[:, y_pix, x_pix], s=4, c='r', label='Outliers')
    plt.plot(np.array(dates), cum_lpt[:, y_pix, x_pix], c='g', label='Filters')
    plt.plot(np.array(dates), cum_lpt[:, y_pix, x_pix] + filt_std[:, y_pix, x_pix], c='r', label='1 STD')
    plt.plot(np.array(dates), cum_lpt[:, y_pix, x_pix] + outlier_thresh * filt_std[:, y_pix, x_pix], c='b', label='Outlier Thresh')
    plt.plot(np.array(dates), cum_lpt[:, y_pix, x_pix] - filt_std[:, y_pix, x_pix], c='r')
    plt.plot(np.array(dates), cum_lpt[:, y_pix, x_pix] - outlier_thresh * filt_std[:, y_pix, x_pix], c='b')
    plt.legend()

    # Replace outliers with filter data
    cum[outlier] = cum_lpt[outlier]

    plt.scatter(np.array(dates), cum[:, y_pix, x_pix], s=4, c='b')
    plt.savefig(os.path.join(outdir, 'finRANSAC{}.png'.format(15000)))
    plt.close()
    print(os.path.join(outdir, 'finRANSAC{}.png'.format(15000)))

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
    std = np.nanmedian(filt_std[:, valid[0][ii], valid[1][ii]])
    limits = outlier_thresh*np.nanmedian(filt_std[:, valid[0][ii], valid[1][ii]])
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
    global pcst, results, n_para

    # Define post-seismic constant
    pcst = 1 / args.tau
    if n_para > 1 and n_valid > 100:
        # pool = multi.Pool(processes=n_para)
        # results = pool.map(fit_pixel_velocities, even_split(np.arange(0, n_valid, 1).tolist(), n_para))
        p = q.Pool(n_para)
        results = np.array(p.map(fit_pixel_velocities, range(n_valid)), dtype=np.float32)
        print(results.shape)
        p.close()
    else:
        results = np.zeros((n_valid, 3 + n_eq * 3))
        for ii in range(n_valid):
            results[ii, :] = fit_pixel_velocities(ii)

def fit_pixel_velocities(ii):
    # Fit Pre- and Post-Seismic Linear velocities, coseismic offset, postseismic relaxation and referencing offset
    disp = cum[:, valid[0][ii], valid[1][ii]]
    # Intercept (reference term), Pre-Seismic Velocity, [offset, log-param, post-seismic velocity]
    G = np.zeros([n_im, 2 + n_eq * 3])
    G[:, 0] = 1
    # G[:eq_ix[0], 1] = date_ord[:eq_ix[0]]
    G[:, 1] = date_ord # Makes the pre-seismic rate the long-term rate. Postseimic linear is summed with this

    daily_rates = [1]

    for ee in range(0, n_eq):
        G[eq_ix[ee]:eq_ix[ee + 1], 2 + ee * 3] = 1
        G[eq_ix[ee]:eq_ix[ee + 1], 3 + ee * 3] = np.log(1 + pcst * (date_ord[eq_ix[ee]:eq_ix[ee + 1]] - ord_eq[ee]))
        # G[eq_ix[ee]:eq_ix[ee + 1], 4 + ee * 3] = date_ord[eq_ix[ee]:eq_ix[ee + 1]]
        G[eq_ix[ee]:eq_ix[ee + 1], 4 + ee * 3] = date_ord[eq_ix[ee]:eq_ix[ee + 1]] - ord_eq[ee] # This means that intercept of the postseismic is the coseismic + avalue?
        daily_rates.append(4 + ee * 3)

    # Add variance matrix for the data
    Q = np.eye(n_im)

    # Weight matrix (inverse of VCM)
    W = np.linalg.inv(Q)

    # Calculate model variance
    var = np.linalg.inv(np.dot(np.dot(G.T, W), G))
    inverr = np.diag(var).copy()

    x = np.matmul(var, np.matmul(G.T, disp))

    # Invert for modelled displacement
    invvel = np.matmul(G, x)

    # if valid[0][ii] > 335 and valid[0][ii] < 345  and valid[1][ii] > 335 and valid[1][ii] < 345:
    #     plt.scatter(dates, disp, s=2, c='k')
    #     plt.plot(dates, invvel, c='g',label='Before')
    #     plt.title('({}/{})'.format(valid[1][ii], valid[0][ii]))
    #     plt.savefig(os.path.join(outdir, '{}.png'.format(ii)))
    #     plt.close()
    #     print(os.path.join(outdir, '{}.png'.format(ii)))


    # Find velocity standard deviation # INFUTURE, USE BOOTSTRAPPING
    std = np.sqrt((1 / n_im) * np.sum((disp - invvel) ** 2))

    # Check that coseismic displacement is at detectable limit (< std) -> Look to also comparing against STD of filtered values either side of the eq
    recalculate = False

    for ee in range(n_eq):
        # Sod it - only doing 1 earthquake at the moment. Can't work out 2 yet
        Gpre = G[eq_ix[ee] - 1, :]
        Gpre[1] = ord_eq[ee]
        pre_disp = np.matmul(Gpre, x)
        Gpost = G[eq_ix[ee] + 1, :]
        Gpost[3] = np.log(1 + pcst * 0)
        Gpost[4] = 0
        post_disp = np.matmul(Gpost, x)
        coseismic = post_disp - pre_disp
        if abs(coseismic) < args.detect_thre * std:
            recalculate = True
            if np.mod(ii, 1000) == 0 and n_para == 1:
                print('Plotting recalculation')
                print(invvel[-1])
                print(ord_eq[ee] - date_ord[-1])
                print((ord_eq[ee] - date_ord[-1]) * x[4])
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
                plt.scatter(dates, disp, s=2, c='k')
                plt.plot(dates, invvel, c='g',label='Before')
                plt.plot(dates, invvel + std, c='r',label='Before STD')
                plt.plot(dates, invvel - std, c='r')
                title = '({},{}) {} \nSTD: {:.2f} Pre: {:.2f} Post: {:.2f} Coseis: {:.2f}\n {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(valid[0][ii], valid[1][ii], date_ord[eq_ix[ee]], std, pre_disp, post_disp, coseismic, x[0], x[1]*365.25, x[2], x[3],(x[4])*365.25)

            # No coseismic deformation -> Check the effect of referencing now, we set the reference to 0, so there will never be a displacement there.....
            # Allow no coseismic displacement or post-seismic relaxation, but allow a change in the linear velocity afterwards
            G[eq_ix[ee]:eq_ix[ee + 1], 2 + ee * 3] = 1 # Allow coseismic displacement
            G[eq_ix[ee]:eq_ix[ee + 1], 3 + ee * 3] = 0 # Allow no postseismic relaxation
            G[eq_ix[ee]:eq_ix[ee + 1], 4 + ee * 3] = date_ord[eq_ix[ee]:eq_ix[ee + 1]] - ord_eq[ee] # Allow a change in linear velocity -> should we allow this?

        # else:
        #     # Change value in result to be true coseismic displacement
        #     x[2 + ee * 3] = coseismic

    if recalculate:
        # Check for Singular matrix
        if (G == 0).all(axis=0).any():
            # Find singular values and remove from the inversion
            singular = np.where((G == 0).all(axis=0))[0]
            good = np.where((G != 0).any(axis=0))[0]
            G = G[:, good]
        else:
            good = np.arange(G.shape[1])

        # Recalculate values
        new = np.matmul(np.linalg.inv(np.dot(G.T, G)), np.matmul(G.T, disp))
        invvel = np.matmul(G, new)
        std = np.sqrt((1 / n_im) * np.sum(((disp - invvel) ** 2)))

        if singular.any():
            x[singular] = 0
            x[good] = new
        else:
            x = new

        if np.mod(ii, 1000) == 0 and n_para == 1:
            plt.plot(dates, invvel, c='b',label='After')
            plt.plot(dates, invvel + std, c='m',label='After STD')
            plt.plot(dates, invvel - std, c='m')
            plt.legend()
            plt.title(title + '\n{:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(x[0], x[1]*365.25, x[2], x[3] ,(x[4])*365.25))
            for ee in range(0, n_eq):
                plt.axvline(x=eq_dt[ee], color="grey", linestyle="--")
            plt.savefig(os.path.join(outdir, '{}.png'.format(ii)))
            plt.close()
            print(os.path.join(outdir, '{}.png'.format(ii)))

    # Convert mm/day to mm/yr
    for dd in daily_rates:
        x[dd] *= 365.25
        inverr[dd] *= 365.25

    # # If using long term rate, calculate the 'true' postseismic linear
    # x[4] = x[1] + x[4]
    x = np.append(x, std)

    return x, inverr

def plot_timeseries(dates, disp, invvel, ii, x, y):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    plt.scatter(dates, disp, s=2, label='{}'.format(ii))
    plt.plot(dates, invvel, label='{}'.format(ii))
    plt.title('({},{})'.format(x,y))
    plt.legend()
    for ii in range(0, n_eq):
        plt.axvline(x=eq_dt[ii], color="grey", linestyle="--")
    plt.savefig(os.path.join(outdir, '{}.png'.format(ii)))

def write_outputs():

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    names = ['intercept', 'pre_vel']
    titles = ['Intercept of Velocity (mm/yr)', 'Preseismic Velocity (mm/yr)']
    for n in range(n_eq):
        eq_names = ['coseismic{}'.format(eq_dates[n]), 'a_value{}'.format(eq_dates[n]), 'post_vel{}'.format(eq_dates[n])]
        eq_titles = ['Coseismic Displacement {} (mm)'.format(eq_dates[n]), 'Postseismic A-value {} (mm)'.format(eq_dates[n]), 'Postseismic velocity {} (mm/yr)'.format(eq_dates[n])]
        names = names + eq_names
        titles = titles + eq_titles

    names.append('vstd')
    titles.append('Velocity Std (mm/yr)')

    print('Writing Outputs to file and png')

    gridResults = np.zeros((len(names), length, width), dtype=np.float32) * np.nan

    for n in range(len(names)):
        filename = os.path.join(outdir, names[n])
        pngname = '{}.png'.format(filename)
        gridResults[n, valid[0], valid[1]] = results[:, n]
        gridResults[n, :, :].tofile(filename)

        vmax = np.nanpercentile(gridResults[n, :, :], 95)
        if 'vstd' in names[n]:
            vmin = 0
            cmap = 'viridis_r'
        else:
            vmin = np.nanpercentile(gridResults[n, :, :], 5)
            vmin = -np.nanmax([abs(vmin), abs(vmax)])
            vmax = np.nanmax([abs(vmin), abs(vmax)])
            cmap = SCM.roma.reversed()
        plot_lib.make_im_png(gridResults[n, :, :], pngname, cmap, titles[n], vmin, vmax)

    if mask_final:
        print('Creating masked png images')
        gridMasked = gridResults.copy()
        mask = io_lib.read_img(maskfile, length, width)
        mask_pix = np.where(mask == 0)
        gridMasked[:, mask_pix[0], mask_pix[1]] = np.nan

        for n in range(len(names)):
            filename = os.path.join(outdir, names[n])
            pngname = '{}.mskd.png'.format(filename)

            vmax = np.nanpercentile(gridMasked[n, :, :], 95)
            if 'vstd' in names[n]:
                vmin = 0
                cmap = 'viridis_r'
            else:
                vmin = np.nanpercentile(gridMasked[n, :, :], 5)
                vmin = -np.nanmax([abs(vmin), abs(vmax)])
                vmax = np.nanmax([abs(vmin), abs(vmax)])
                cmap = SCM.roma.reversed()

            plot_lib.make_im_png(gridMasked[n, :, :], pngname, cmap, titles[n], vmin, vmax)

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
    cumh5.create_dataset('cum', data=cum, compression=compress)
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
