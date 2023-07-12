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
import LiCSBAS_plot_lib as plot_lib
import SCM

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
    parser.add_argument('--n_para', dest='n_para', default=False, type=int, help='number of parallel processing')
    parser.add_argument('--tau', dest='tau', default=6, help='Post-seismic relaxation time (days)')
    parser.add_argument('--max_its', dest='max_its', default=5, type=int, help='Maximum number of iterations for temporal filter')
    parser.add_argument('--nofilter', dest='deoutlier', default=True, action='store_false', help="Don't do any temporal filtering")

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
    global tsadir, infodir, resultdir, outdir, h5file, reffile, maskfile, eqfile, outlier_thresh, q

    # define input directories
    tsadir = os.path.abspath(os.path.join(args.frame_dir, args.ts_dir))
    infodir = os.path.join(tsadir, 'info')
    resultdir = os.path.join(tsadir, 'results')
    outdir = os.path.join(resultdir, 'seismic_vels')

    # define input files
    h5file = os.path.join(tsadir, args.h5_file)
    reffile = os.path.join(tsadir, args.ref_file)
    if not os.path.exists(reffile):
        reffile = os.path.join(infodir, args.ref_file)

    if args.apply_mask:
        maskfile = os.path.join(resultdir, 'mask')
    eqfile = os.path.abspath(args.eq_list)

    outlier_thresh = args.outlier_thre

    q = multi.get_context('fork')

def load_data():
    global width, length, data, n_im, cum, dates, length, width, refx1, refx2, refy1, refy2, n_para, eq_dates, n_eq, eq_dt, eq_ix, ord_eq, date_ord, eq_dates, valid, n_valid

    data = h5py.File(h5file, 'r')
    dates = [dt.datetime.strptime(str(d), '%Y%m%d').date() for d in np.array(data['imdates'])]

    # read reference
    with open(reffile, "r") as f:
        refarea = f.read().split()[0]  # str, x1/x2/y1/y2
    refx1, refx2, refy1, refy2 = [int(s) for s in re.split('[:/]', refarea)]

    cum = reference_disp(np.array(data['cum']), refx1, refx2, refy1, refy2)

    n_im, length, width = cum.shape

    # Identify all pixels where the is time series data
    vel = np.array(data['vel'])

    if args.apply_mask:
        print('Applying Mask')
        mask = io_lib.read_img(maskfile, length, width)
        maskx, masky = np.where(mask == 0)
        cum[:, maskx, masky] = np.nan
        vel[maskx, masky] = np.nan

    valid = np.where(~np.isnan(vel))
    n_valid = valid[0].shape[0]

    # multi-processing
    try:
        n_para = min(len(os.sched_getaffinity(0)), 8) # maximum use 8 cores
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
        eq_ix.append(np.sum([1 for d in dates if str(d) < eq_date], dtype='int'))
    eq_ix.append(n_im)

    # Make all dates ordinal
    ord_eq = np.array([eq.toordinal() for eq in eq_dt]) - dates[0].toordinal()
    date_ord = np.array([x.toordinal() for x in dates]) - dates[0].toordinal()

def reference_disp(data, refx1, refx2, refy1, refy2):

    # Reference all data to reference area through static offset
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

    data = data - ref

    return data

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
    cum_lpt, outlier = find_outliers()

    # Iterate until all outliers are removed
    n_its = 1

    # Replace all outliers with filtered values
    cum[outlier] = cum_lpt[outlier]
    all_outliers = outlier

    ## Here would be the place to add in iterations (as a while loop of while len(outlier) > 0)
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
    cum = np.array(data['cum'])
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
    global pcst, results

    # Define post-seismic constant
    pcst = 1 / args.tau

    if n_para > 1 and n_valid > 100:
        # pool = multi.Pool(processes=n_para)
        # results = pool.map(fit_pixel_velocities, even_split(np.arange(0, n_valid, 1).tolist(), n_para))
        p = q.Pool(n_para)
        results = np.array(p.map(fit_pixel_velocities, range(n_valid)), dtype=np.float32)
        p.close()
    else:
        results = fit_pixel_velocities(np.arange(0, n_valid, 1).tolist())

def fit_pixel_velocities(ii):
    # Fit Pre- and Post-Seismic Linear velocities, coseismic offset, postseismic relaxation and referencing offset

    disp = cum[:, valid[0][ii], valid[1][ii]]
    # Intercept (reference term), Pre-Seismic Velocity, [offset, log-param, post-seismic velocity]
    G = np.zeros([n_im, 2 + n_eq * 3])
    G[:, 0] = 1
    G[:eq_ix[0], 1] = date_ord[:eq_ix[0]]

    daily_rates = [1]

    for ee in range(0, n_eq):
        G[eq_ix[ee]:eq_ix[ee + 1], 2 + ee * 3] = 1
        G[eq_ix[ee]:eq_ix[ee + 1], 3 + ee * 3] = np.log(1 + pcst * (date_ord[eq_ix[ee]:eq_ix[ee + 1]] - ord_eq[ee]))
        G[eq_ix[ee]:eq_ix[ee + 1], 4 + ee * 3] = date_ord[eq_ix[ee]:eq_ix[ee + 1]]
        daily_rates.append(4 + ee * 3)

    x = np.matmul(np.linalg.inv(np.dot(G.T, G)), np.matmul(G.T, disp))

    # Plot the inverted velocity time series
    invvel = np.matmul(G, x)
    x = np.append(x, np.sqrt((1 / n_im) * np.sum(((disp - invvel) ** 2))))

    # Convert mm/day to mm/yr
    for dd in daily_rates:
        x[dd] *= 365.25

    if np.mod(ii, 5000) == 0:
        print('{}/{} Velocity STD: {}'.format(i, n_valid, x[5]))
        print('    InSAR Offset and Initial Velocity: {:.2f} mm/yr, {:.2f} mm'.format(x[0], x[1]))
        for n in range(0, n_eq):
            print('    Co-seismic offset for {}: {:.0f} mm'.format(eq_dates[n], x[2 + n * 3]))
            print('    Post-seismic A-value and velocity: {:.2f}, {:.2f} mm/yr\n'.format(x[3 + n * 3], x[4 + n * 3]))

        plt.scatter(dates, disp, s=2, c='blue', label='Displacement')
        plt.plot(dates, invvel, label='Const Linear w/ co- + post-seismic')
        plt.legend()
        plt.savefig(os.path.join(outdir, ii + '.png'))

    return x

def write_outputs():

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    names = ['intercept', 'pre_vel']
    titles = ['Intercept of Velocity (mm/yr)', 'Preseismic Velocity (mm/yr)']
    for n in range(n_eq):
        eq_names = ['coseismic{}'.format(n), 'a_value{}'.format(n), 'post_vel{}'.format(n)]
        eq_titles = ['Coseismic Displacement {} (mm)'.format(n), 'Postseismc A-value {}'.format(n), 'Postseismic velocity {} (mm/yr)'.format(n)]
        names = names + eq_names
        titles = titles + eq_titles

    names.append('vstd')
    titles.append('Velocity Std (mm/yr)')

    print('Writing Outputs to file')

    cmap_vel = SCM.roma.reversed()
    data = np.zeros((len(names), length, width), dtype=np.float32) * np.nan

    for n in range(len(names) - 1):
        print(titles[n])
        filename = os.path.join(outdir, names[n])
        pngname = '{}.png'.format(filename)
        data[n, valid[0], valid[1]] = results[:, n]
        data[n, :, :].tofile(filename)

        vmin = np.nanpercentile(data[n, :, :], 5)
        vmax = np.nanpercentile(data[n, :, :], 95)
        vlim = np.nanmax([abs(vmin), abs(vmax)])
        print(vmin, vmax, vlim)

        plot_lib.make_im_png(data[n, :, :], pngname, cmap_vel, titles[n], -vlim, vlim)

    print(titles[-1])
    filename = os.path.join(resultdir, names[-1])
    pngname = '{}.png'.format(filename)
    data[-1, valid[0], valid[1]] = results[:, -1]
    data[-1, :, :].tofile(filename)

    vmax = np.nanpercentile(data[-1, :, :], 95)
    print(vmax)

    plot_lib.make_im_png(data[-1, :, :], pngname, 'viridis', titles[-1], 0, vmax)

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
