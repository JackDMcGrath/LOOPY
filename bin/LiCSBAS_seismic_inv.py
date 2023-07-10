#!/usr/bin/env python3
"""
v1.0.0 Jack McGrath, University of Leeds

Load in inverted displacements from LiCSBAS and fit linear, co-seismic and postseismic fits to them.

Based off Liu et al. (2021), Improving the Resolving Power of InSAR for Earthquakes Using Time Series: A Case Study in Iran
"""

import os
import re
import sys
import h5py
import time
import warnings
import argparse
import multiprocessing as multi
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import LiCSBAS_io_lib as io_lib
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext

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
    parser.add_argument('-i', dest='h5_file', default='cum.h5', help='.h5 file containing results of LiCSBAS velocity inversion')
    parser.add_argument('-r', dest='ref_file', default='130ref.txt', help='txt file containing reference area')
    parser.add_argument('-m', dest='apply_mask', default=None, help='mask file to apply to velocities')
    parser.add_argument('-e', dest='eq_list', default=None, help='Text file containing the dates of the earthquakes to be fitted')
    parser.add_argument('-s', dest='outlier_thre', default=3, type=float, help='StdDev threshold used to remove outliers')
    parser.add_argument('--n_para', dest='n_para', default=False, help='number of parallel processing')
    parser.add_argument('--tau', dest='tau', default=6, help='Post-seismic relazation time (days)')
    parser.add_argument('--max_its', dest='max_its', default=5, help='Maximum number of iterations for temporal filter')

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
    print('Output directory: {}\n'.format(os.path.relpath(tsadir)))

def set_input_output():
    global tsadir, infodir, resultdir, h5file, reffile, maskfile, eqfile, outlier_thresh

    # define input directories
    tsadir = os.path.abspath(os.path.join(args.frame_dir, args.ts_dir))
    infodir = os.path.join(tsadir, 'info')
    resultdir = os.path.join(tsadir, 'results')

    # define input files
    h5file = os.path.join(tsadir, args.h5_file)
    reffile = os.path.join(tsadir, args.ref_file)
    if args.apply_mask:
        maskfile = os.path.join(resultdir, 'mask')
    eqfile = os.path.abspath(args.eq_list)

    outlier_thresh = args.outlier_thre

def load_data():
    global width, length, data, n_im, cum, dates, length, width, refx1, refx2, refy1, refy2, n_para, eq_dates, n_eq, eq_dt, eq_ix, ord_eq, date_ord, eq_dates

    data = h5py.File(h5file, 'r')
    cum = np.array(data['cum'])
    dates = np.array(data['imdates'])
    dates = [dt.datetime.strptime(str(d), '%Y%m%d').date() for d in dates]
    n_im, length, width = cum.shape

    # read reference
    with open(reffile, "r") as f:
        refarea = f.read().split()[0]  # str, x1/x2/y1/y2
    refx1, refx2, refy1, refy2 = [int(s) for s in re.split('[:/]', refarea)]

    cum = reference_disp(cum, refx1, refx2, refy1, refy2)

    # multi-processing
    if not args.n_para:
        try:
            n_para = min(len(os.sched_getaffinity(0)), 8) # maximum use 8 cores
        except:
            n_para = multi.cpu_count()
    else:
        n_para = args.n_para

    ## Sort dates
    # Get list of earthquake dates and index
    eq_dates = io_lib.read_ifg_list(eqfile)
    eq_dates.sort()
    n_eq = len(eq_dates)

    # Find which index each earthquake correlates to
    eq_dt = [dt.datetime.strptime(str(eq_date), '%Y%m%d').date() for eq_date in eq_dates]
    eq_ix = []
    for eq_date in eq_dates:
        eq_ix.append(np.sum([1 for d in dates if str(d) < eq_date], dtype='int'))
    eq_ix.append(n_im)

    # Make all dates ordinal
    ord_eq = np.array([eq.toordinal() for eq in eq_dt]) - dates[0].toordinal()
    date_ord = np.array([x.toordinal() for x in dates]) - dates[0].toordinal()

def reference_disp(cum, refx1, refx2, refy1, refy2):

    # Reference all data to reference area through static offset
    ref = np.nanmean(cum[:, refy1:refy2, refx1:refx2], axis=(2, 1)).reshape(cum.shape[0], 1, 1)
    # Check that one of the refs is not all nan. If so, increase ref area
    if np.isnan(ref).any():
        print('NaN Value for reference for {} dates. Increasing ref area kernel'.format(np.sum(np.isnan(ref))))
        while np.isnan(ref).any():
            refx1 -= 1
            refx2 += 1
            refy1 -= 1
            refy2 += 1
            ref = np.nanmean(cum[:, refy1:refy2, refx1:refx2], axis=(2, 1)).reshape(cum.shape[0], 1, 1)

    print('Reference area ({}:{}, {}:{})'.format(refx1, refx2, refy1, refy2))

    cum = cum - ref

    return cum

def temporal_filter():
    global ixs_dict, dt_cum, filterdates, filtwidth_yr, cum_lpt, n_its, filt_std
    """
    Apply a low pass temporal filter, and remove outliers.
     Iterate until no outliers left so filter is not distorted
     by outliers, then on the final pass, remove outliers between
     the original data and final iteration filter
    """

    print('Filtering Temporally for outliers > {} std'.format(outlier_thresh))
    dt_cum = date_ord / 365.25  # Set dates in terms of years
    n_im, length, width = cum.shape  # Get shape information of data
    filtwidth_yr = dt_cum[-1] / (n_im - 1) * 3  # Set the filter width based on n * average epoch seperation
    cum_lpt = np.zeros((n_im, length, width), dtype=np.float32)
    filterdates = np.linspace(0, n_im - 1, n_im, dtype='int').tolist()
    n_its = 1
    ixs_dict = get_filter_dates(dt_cum, filtwidth_yr, filterdates)

    # Find and remove any outliers above filtered std threshold
    cum_lpt, outlier = find_outliers()

    # Iterate until all outliers are removed
    n_its = 1

    # Replace all outliers with filtered values
    cum[outlier] = cum_lpt[outlier]
    all_outliers = outlier.copy()

    ## Here would be the place to add in iterations (as a while loop of while len(outlier) > 0)
    while len(outlier) > 0:
        n_its += 1
        print('Running Iteration {}/{}'.format(n_its, args.max_its))
        cum_lpt, outlier = find_outliers()
        # Replace all outliers with filtered values
        cum[outlier] = cum_lpt[outlier]
        all_outliers = np.unique(np.concatenate((all_outliers, outlier), axis=1), axis=1)

    # Reload original data, replace identified outliers with final filtered values
    cum = np.array(data['cum'])
    cum[all_outliers] = cum_lpt[all_outliers]

    print('Finding moving stddev')
    filt_std = np.ones(cum.shape) * np.nan
    filterdates = np.linspace(0, n_im - 1, n_im, dtype='int')
    valid = np.where(~np.isnan(data['vel']))
    diff = cum[:, valid[0], valid[1]] - cum_lpt[:, valid[0], valid[1]]

    for i in filterdates:
        if np.mod(i, 10) == 0:
            print("  {0:3}/{1:3}th image...".format(i, n_im), flush=True)

        filt_std[i, valid[0], valid[1]] = np.nanstd(diff[ixs_dict[i], :], axis=0)

def find_outliers():
    print('Outlier removal iteration {}'.format(n_its))
    if n_para > 1 and len(filterdates) > 20:
        pool = multi.Pool(processes=n_para)
        cum_lpt = pool.map(lpt_filter, even_split(filterdates, n_para))
    else:
        cum_lpt = lpt_filter(filterdates)

    # Find STD
    diff = cum - cum_lpt  # Difference between data and filtered data
    for i in filterdates:
        with warnings.catch_warnings():  # To silence warning by zero division
            warnings.simplefilter('ignore', RuntimeWarning)
            filt_std[i, :, :] = np.nanstd(diff[ixs_dict[i], :, :], axis=0)

    # Find location of outliers
    outlier = np.where(abs(diff) > outlier_thresh * filt_std)

    print('\t{} outliers identified'.format(len(outlier)))

    return cum_lpt, outlier

def lpt_filter():
    for i in filterdates:
        if np.mod(i, 10) == 0:
            print("  {0:3}/{1:3}th image...".format(i, len(filterdates)), flush=True)

        # Find time difference between filter date and other dates
        time_diff_sq = (dt_cum[i] - dt_cum) ** 2

        # Get data limits
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

        cum_lpt[i, :, :] = lpt

    return cum_lpt

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
    global pcst, valid, n_valid

    # Identify all pixels where the is time series data
    vel = data['vel']
    if args.apply_mask:
        mask = io_lib.read_img(maskfile, length, width)
        vel[np.where(mask == 0)] = np.nan
    valid = np.where(~np.isnan(vel))
    n_valid = valid[0].shape[0]

    # Preallocate shape (n_pixels x 6 (5 inversion params, and std))
    results = np.zeros(n_valid, 6) * np.nan

    # Define post-seismic constant
    pcst = 1 / args.tau

    if n_para > 1 and n_valid > 100:
        pool = multi.Pool(processes=n_para)
        results = pool.map(fit_pixel_velocities, even_split(np.arange(0, n_valid, 1).tolist(), n_para))
    else:
        fit_pixel_velocities(np.arange(0, n_valid, 1).tolist())



def fit_pixel_velocities(ix):
    # Fit Pre- and Post-Seismic Linear velocities, coseismic offset, postseismic relaxation and referencing offset
    results = np.zeros((len(ix), 6)) * np.nan

    for pix in ix:
        disp = cum[:, valid[0][pix], valid[1][pix]]
        # Intercept (reference term), Pre-Seismic Velocity, [offset, log-param, post-seismic velocity]
        G = np.zeros([n_im, 2 + n_eq * 3])
        G[:, 0] = 1
        G[:eq_ix[0], 1] = date_ord[:eq_ix[0]]
        for i in range(0, n_eq):
            G[eq_ix[i]:eq_ix[i + 1], 2 + i * 3] = 1
            G[eq_ix[i]:eq_ix[i + 1], 3 + i * 3] = np.log(1 + pcst * (date_ord[eq_ix[i]:eq_ix[i + 1]] - ord_eq[i]))
            G[eq_ix[i]:eq_ix[i + 1], 4 + i * 3] = date_ord[eq_ix[i]:eq_ix[i + 1]]

        x = np.matmul(np.linalg.inv(np.dot(G.T, G)), np.matmul(G.T, disp))

        # Plot the inverted velocity time series
        invvel = np.matmul(G, x)
        x[5] = (1 / n_im) * ((disp - invvel) ** 2)

        if np.mod(pix, 10) == 0:
            print('{}/{} Velocity STD: {}'.format(pix, n_valid, x[5]))
            print('    Initial Velocity and InSAR Offset: {:.2f} mm/yr, {:.2f} mm'.format(x[1] * 365.25, x[0]))
            for i in range(0, n_eq):
                print('    Co-seismic offset for {}: {:.0f} mm'.format(eq_dates[i], x[2 + i * 3]))
                print('    Post-seismic A-value and velocity: {:.2f}, {:.2f} mm/yr\n'.format(x[3 + i * 3], x[4 + i * 3] * 365.25))

        results[pix, :] = x

    return results


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

    # Remove outliers from image displacements to improve velocity fitting
    temporal_filter()

    # Fit velocities
    fit_velocities()


    finish()

if __name__ == "__main__":
    main()
