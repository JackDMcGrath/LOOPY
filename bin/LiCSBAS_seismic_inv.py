#!/usr/bin/env python3
"""
v1.0.0 Jack McGrath, University of Leeds

Load in inverted displacements, and fit linear, co-seismic and postseismic fits to them.

Based of Fei's codes

"""

import os
import sys
import h5py
import warnings
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import LiCSBAS_io_lib as io_lib

def init_args():
    global args

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=CustomFormatter)
    parser.add_argument('-f', dest='frame_dir', default="./", help="directory of LiCSBAS output of a particular frame")
    parser.add_argument('-t', dest='ts_dir', default="TS_GEOCml10GACOS", help="folder containing .h5 file")
    parser.add_argument('-i', dest='h5_file', default='cum.h5', help='.h5 file containing results of LiCSBAS velocity inversion')
    parser.add_argument('-r', dest='ref_file', default='130ref.txt', help='txt file containing reference area')
    parser.add_argument('-m', dest='mask_file', default=None, help='mask file to apply to velocities')
    parser.add_argument('-e', dest='eq_list', default=None, help='Text file containing the dates of the earthquakes to be fitted')
    parser.add_argument('-s', dest='outlier_thre', default=10, type=float, help='StdDev threshold used to remove outliers')
    parser.add_argument('--n_para', dest='n_para', default=False, help='number of parallel processing')

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
    maskfile = os.path.join(resultdir, 'mask')
    eqfile = os.path.abspath(args.eq_list)

    outlier_thresh = args.outlier_thre

def load_data():
    global width, length, n_im, cum, dates, length, width, refx1, refx2, refy1, refy2, n_para, eq_dates, n_eq, eq_dt, eq_ix, ord_eq, date_ord

    data = h5py.File(filename, 'r')
    cum = np.array(data['cum'])
    dates = np.array(data['imdates'])
    dates = [dt.datetime.strptime(str(d), '%Y%m%d').date() for d in dates]
    n_im, length, width = cum.shape

    # read reference
    with open(reffile, "r") as f:
        refarea = f.read().split()[0]  # str, x1/x2/y1/y2
    refx1, refx2, refy1, refy2 = [int(s) for s in re.split('[:/]', refarea)]

    cum = reference_disp()

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
        eq_ix.append(np.sum([1 for d in dates if d < eq_date], dtype='int'))
    eq_ix.append(n_im)

    # Make all dates ordinal
    ord_eq = np.array([eq.toordinal() for eq in eq_dt]) - dates[0].toordinal()
    date_ord = np.array([x.toordinal() for x in dates]) - dates[0].toordinal()

def reference_disp():
    # Reference all data to reference area through static offset
    ref = np.nanmean(cum[:, refy1:refy2, refx1:refx2], axis=(2, 1)).reshape(cum.shape[0], 1, 1)
    # Check that one of the refs is not all nan. If so, increase ref area
    if np.isnan(ref).any():
        print('NaN Value for reference for {} dates. Increasing ref area kernel'.format(np.sum(np.isnan(ref))))
        while np.isnan(ref).any():
            ref_x = [ref_x[0] - 1, ref_x[1] + 1]
            ref_y = [ref_y[0] - 1, ref_y[1] + 1]
            ref = np.nanmean(cum[:, ref_y[0]:ref_y[1], ref_x[0]:ref_x[1]], axis=(2, 1)).reshape(cum.shape[0], 1, 1)
    
    print('Reference area ({}:{}, {}:{})'.format(ref_x[0], ref_x[1], ref_y[0], ref_y[1]))
    
    cum = cum - ref
 
    return cum
    
def even_split(a, n):
    """ Divide a list, a, in to n even parts"""
    n = min(n, len(a)) # to avoid empty lists
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def integer_correction():
    global TSmask
    TSmask = io_lib.read_img(os.path.join(resultdir, 'mask'), length, width)
    print('Correcting....')
    # parallel processing
    if n_para > 1 and len(res_list) > 20:
        pool = multi.Pool(processes=n_para)
        pool.map(correcting_by_integer, even_split(res_list, n_para))
    else:
        correcting_by_integer(res_list)

def filter_outliers():
    # %% Temporal Filter
    # Apply low pass temporal filter and remove outliers. Iterate until no outliers
    # left so filter is not distorted by outliers, then on the final pass, find
    # outliers between original data and final iteration filter
    print('Filtering Temporally for outliers > {} std'.format(outlier_thresh))

    dt_cum = date_ord / 365.25  # Set dates in terms of years
    filtwidth_yr = dt_cum[-1] / (n_im - 1) * 3  # Set the filter width based on n * average epoch seperation
    cum_filt1 = np.zeros((n_im, length, width), dtype=np.float32)
    cum_filt2 = np.zeros((n_im, length, width), dtype=np.float32)
    filterdates = np.linspace(0, n_im - 1, n_im, dtype='int')
    valid = np.where(~np.isnan(data['vel']))
    filt_std1 = np.ones(cum.shape) * np.nan
    filt_std2 = np.ones(cum.shape) * np.nan

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

    # Filter once to find outliers
    for i in filterdates:
        if np.mod(i, 10) == 0:
            print("  {0:3}/{1:3}th image...".format(i, len(filterdates)), flush=True)

        # High Pass Temporal Filter
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

        cum_lpt = np.nansum(cum[ixs, :, :] * weight_factor, axis=0)
        cum_lpt[np.where(np.isnan(cum[i, :, :]))] = np.nan

        # Filtered Values
        cum_filt1[i, :, :] = cum_lpt

    # Find STD
    diff = cum - cum_filt1  # Difference between data and filtered data
    for i in filterdates:
        with warnings.catch_warnings():  # To silence warning by zero division
            warnings.simplefilter('ignore', RuntimeWarning)
            filt_std1[i, :, :] = np.nanstd(diff[ixs_dict[i], :, :], axis=0)

    # Find location of outliers
    outlier = np.where(abs(diff) > outlier_thresh * filt_std1)

    # Replace all outliers with filtered values
    cum_replaced = cum.copy()
    cum_replaced[outlier] = cum_filt1[outlier]

    # Rerun filtering with new values
    for i in filterdates:
        if np.mod(i, 10) == 0:
            print("  {0:3}/{1:3}th image...".format(i, len(filterdates)), flush=True)

        # High Pass Temporal Filter
        time_diff_sq = (dt_cum[i] - dt_cum) ** 2

        # Get data limits
        ixs = ixs_dict[i]

        weight_factor = np.tile(np.exp(-time_diff_sq[ixs] / 2 / filtwidth_yr ** 2)[:, np.newaxis, np.newaxis], (1, length, width))  # len(ixs), length, width

        # Take into account nan in cum
        weight_factor = weight_factor * (~np.isnan(cum_replaced[ixs, :, :]))

        # Normalize weight
        with warnings.catch_warnings():  # To silence warning by zero division
            warnings.simplefilter('ignore', RuntimeWarning)
            weight_factor = weight_factor / np.sum(weight_factor, axis=0)

        cum_lpt = np.nansum(cum_replaced[ixs, :, :] * weight_factor, axis=0)
        cum_lpt[np.where(np.isnan(cum_replaced[i, :, :]))] = np.nan

        # Filtered Values
        cum_filt2[i, :, :] = cum_lpt

    # Find STD
    diff = cum_replaced - cum_filt2  # Difference between data and filtered data
    for i in filterdates:
        with warnings.catch_warnings():  # To silence warning by zero division
            warnings.simplefilter('ignore', RuntimeWarning)
            filt_std2[i, :, :] = np.nanstd(diff[ixs_dict[i], :, :], axis=0)

    # Find location of outliers
    outlier = np.where(abs(diff) > outlier_thresh * filt_std2)

    # Replace all outliers with filtered values
    cum_replaced2 = cum_replaced.copy()
    cum_replaced2[outlier] = cum_filt2[outlier]

    # %% Set cum to be the data, with pixels outside the 2std limit replaced by filtered values
    print('Running final Check')
    cum = cum_replaced2.copy()

    # %% Find a moving stddev of the data, relative to the filter values, across same time window
    print('Finding moving stddev')
    filt_std = np.ones(cum.shape)
    filterdates = np.linspace(0, n_im - 1, n_im, dtype='int')
    valid = np.where(~np.isnan(data['vel']))
    diff = cum[:, valid[0], valid[1]] - cum_filt2[:, valid[0], valid[1]]


    for i in filterdates:
        if np.mod(i, 10) == 0:
            print("  {0:3}/{1:3}th image...".format(i, n_im), flush=True)

        filt_std[i, valid[0], valid[1]] = np.nanstd(diff[ixs_dict[i], :], axis=0)


def main():
    start()
    init_args()
    set_input_output()
    load_data()

    filter_outliers()

    integer_correction()

    finish()

if __name__ == "__main__":
    main()












# # %% Calculate the fit for all pixels
# if plot_coseis:
#     valid = np.where(~np.isnan(data['vel']))
#     n_valid = len(valid[0])
#     results = np.ones((length, width, 6)) * np.nan
#     slip_ix = [x for x in range(0, 2 + 2 * n_eq, 2)]

#     count = 0
#     order = np.random.permutation(range(0, n_valid))
#     for i in order:
#         count += 1

#     # #TODO: Deal with nan values in disp
#         disp = cum[-n_im:, valid[0][i], valid[1][i]]
#         G_im = n_im - np.sum(np.isnan(disp))
#         keep_val = ~np.isnan(disp)
#         disp = np.delete(disp, np.isnan(disp))
#         # Calculate using multiple linear trends and co-seismics
#         # Gradient, Y-intercept, post-Gradients, offsets
#         G = np.ones([G_im, 2 + 2 * n_eq])
#         G[:, range(0, 2 + n_eq * 2, 2)] = np.repeat(date_ord[keep_val], n_eq + 1).reshape(G_im, n_eq + 1)
#         G[eq_ix[0]:, 0] = 0
#         G[eq_ix[0]:, 1] = 0
#         for eq in range(0, n_eq):
#             if eq != n_eq - 1:
#                 G[:eq_ix[eq], 2 + eq * 2:4 + eq * 2] = 0
#                 G[eq_ix[eq + 1]:, 2 + eq * 2:4 + eq * 2] = 0
#             else:
#                 G[:eq_ix[eq], 2 + eq * 2:4 + eq * 2] = 0

#         x = np.matmul(np.linalg.inv(np.dot(G.T, G)), np.matmul(G.T, disp))

#         results[valid[0][i], valid[1][i], slip_ix] = x[slip_ix] * 365.25
#         results[valid[0][i], valid[1][i], 1] = x[1]

#         # Calculate co-seismic offsets for each earthquake and store
#         for eq in range(0, n_eq):
#             results[valid[0][i], valid[1][i], eq * 2 + 3] = np.matmul([ord_eq[eq], 1], x[eq * 2 + 2:eq * 2 + 4]) - np.matmul([ord_eq[eq], 1], x[eq * 2: eq * 2 + 2])

#     # #TODO: Check std of diff between trend and EQ- if smaller than STD remove EQ diff = disp - np.matmul(G, x)

#         if count % 5000 == 0:
#             print(count, '/', n_valid)
#             # print(co[valid[0][i], valid[1][i]])
#             # print(valid[0][i], valid[1][i])
#         if count % 10000 == 0:
#             # %
#             plotting = results[:, :, 3].flatten()
#             plotting = np.delete(plotting, np.isnan(plotting))
#             phigh = np.percentile(plotting, 95)
#             plow = np.percentile(plotting, 5)
#             plt.imshow(results[:, :, 3], interpolation='nearest', cmap='RdYlBu', vmin=-max(abs(plow), abs(phigh)), vmax=max(abs(plow), abs(phigh)))
#             plt.title('Christchurch offsets')
#             plt.colorbar()
#             plt.show()

#             plotting = results[:, :, 5].flatten()
#             plotting = np.delete(plotting, np.isnan(plotting))
#             phigh = np.percentile(plotting, 95)
#             plow = np.percentile(plotting, 5)
#             plt.imshow(results[:, :, 5], interpolation='nearest', cmap='RdYlBu', vmin=-max(abs(plow), abs(phigh)), vmax=max(abs(plow), abs(phigh)))
#             plt.title('Kaikoura Offsets')
#             plt.colorbar()
#             plt.show()
# # %%

# disp = cum[-n_im:, ploty, plotx]
# G_im = n_im - np.sum(np.isnan(disp))
# keep_val = ~np.isnan(disp)
# date_plot = date_all.copy()
# for i in np.where(np.isnan(disp))[0]:
#     del date_plot[i]
# disp = np.delete(disp, np.isnan(disp))
# # #TODO: eq_ix needs adapting when having to drop for nans

# # Calculate using multiple linear trends and co-seismics
# # Gradient, Y-intercept, post-Gradients, offsets
# G = np.ones([G_im, 2 + 2 * n_eq])
# G[:, range(0, 2 + n_eq * 2, 2)] = np.repeat(date_ord[keep_val], n_eq + 1).reshape(G_im, n_eq + 1)
# G[eq_ix[0]:, 0] = 0
# G[eq_ix[0]:, 1] = 0
# for eq in range(0, n_eq):
#     if eq != n_eq - 1:
#         G[:eq_ix[eq], 2 + eq * 2:4 + eq * 2] = 0
#         G[eq_ix[eq + 1]:, 2 + eq * 2:4 + eq * 2] = 0
#     else:
#         G[:eq_ix[eq], 2 + eq * 2:4 + eq * 2] = 0

# x = np.matmul(np.linalg.inv(np.dot(G.T, G)), np.matmul(G.T, disp))

# plt.scatter(date_all, cum[-n_im:, ploty, plotx], s=2, c='red')
# plt.plot(date_plot, np.matmul(G, x), label='Multi Linear w/ coseismic')
# for eq in range(0, n_eq):
#     plt.axvline(x=eq_dt[eq], color="grey", linestyle="--")
# # plt.title('Pix = ({},{}), Reference Area = [{}:{}, {}:{}]'.format(valid[0][i], valid[1][i], ref_x[0], ref_x[1], ref_y[0], ref_y[1]))
# plt.show()

# #sys.exit()
# # %%
# # Gradient, y-intercept
# G = np.ones([n_im, 2])
# G[:, 0] = date_ord

# # %% Inversions
# # m, c = np.linalg.lstsq(G, disp.T, rcond=None)[0]
# m, c = np.matmul(np.linalg.inv(np.dot(G.T, G)), np.matmul(G.T, disp))  # Faster

# lin = m * date_ord + c

# print('Simple Linear Velocity: {:.2f} mm/yr'.format(m * 365.25))
# plot_trend = 0

# if enddate > eq_date:
#     # % One Linear Trend and Co-seismic offsets
#     # Gradient, y-intercept, co-seismic
#     G = np.ones([n_im, 2 + n_eq])
#     G[:, 0] = date_ord
#     for i in range(0, n_eq):
#         G[:eq_ix[i], 2 + i] = 0

#     x = np.matmul(np.linalg.inv(np.dot(G.T, G)), np.matmul(G.T, disp))
#     heavy = np.matmul(G, x)

#     print('Constant Velocity with co-seismic offsets')
#     print('    Constant Velocity and y-intercept: {:.2f} mm/yr, {:.2f} mm'.format(x[0] * 365.25, x[1]))
#     for i in range(0, n_eq):
#         print('    Co-seismic offset for {}: {:.0f} mm'.format(eq_dates[i], x[2 + i]))
#     plot_trend += 1

#     # %% Multiple Linear Trends and Co-seismic offsets
#     # Gradient, Y-intercept, post-Gradients, offsets
#     G = np.ones([n_im, 2 + 2 * n_eq])
#     G[:, range(0, 2 + n_eq * 2, 2)] = np.repeat(date_ord, n_eq + 1).reshape(n_im, n_eq + 1)
#     G[eq_ix[0]:, 0] = 0
#     G[eq_ix[0]:, 1] = 0
#     for i in range(0, n_eq):
#         if i != n_eq - 1:
#             G[:eq_ix[i], 2 + i * 2:4 + i * 2] = 0
#             G[eq_ix[i + 1]:, 2 + i * 2:4 + i * 2] = 0
#         else:
#             G[:eq_ix[i], 2 + i * 2:4 + i * 2] = 0

#     x = np.matmul(np.linalg.inv(np.dot(G.T, G)), np.matmul(G.T, disp))
#     split = np.matmul(G, x)

#     print('Independent Gradients with co-seismic offsets')
#     print('    Initial Velocity and y-intercept: {:.2f} mm/yr, {:.2f} mm'.format(x[0] * 365.25, x[1]))
#     # print('    Co-seismic offset and post-seismic linear vel for {}: {:.0f} mm, {:.2f} mm/yr'.format(eq_dates[0], x[3 + 2 * 0], x[2 + 2 * 0] * 365.25))
#     for i in range(0, n_eq):
#         offset = np.matmul([ord_eq[i], 1], x[i * 2 + 2:i * 2 + 4]) - np.matmul([ord_eq[i], 1], x[i * 2: i * 2 + 2])
#         print('    Co-seismic offset and post-seismic linear vel for {}: {:.0f} mm, {:.2f} mm/yr'.format(eq_dates[i], offset, x[2 + 2 * i] * 365.25))
#     plot_trend += 1

#     # %% Constant Linear with Co- and post-seismic

#     pcst = 0.178

#     # Gradient, intercept, offset, log-param, post-velocity?
#     G = np.zeros([n_im, 2 + n_eq * 3])
#     G[:, 0] = date_ord
#     G[:, 1] = 1
#     for i in range(0, n_eq):
#         G[eq_ix[i]:eq_ix[i + 1], 2 + i * 3] = 1
#         G[eq_ix[i]:eq_ix[i + 1], 3 + i * 3] = np.log(1 + pcst * (date_ord[eq_ix[i]:eq_ix[i + 1]] - ord_eq[i]))
#         G[eq_ix[i]:eq_ix[i + 1], 4 + i * 3] = date_ord[eq_ix[i]:eq_ix[i + 1]]

#     x = np.matmul(np.linalg.inv(np.dot(G.T, G)), np.matmul(G.T, disp))
#     invvel = np.matmul(G, x)

#     print('Constant Velocity with co- and post-seismic effects')
#     print('    Initial Velocity and y-intercept: {:.2f} mm/yr, {:.2f} mm'.format(x[0] * 365.25, x[1]))
#     for i in range(0, n_eq):
#         pre_G = G[eq_ix[i] - 1]
#         pre_G[0] = ord_eq[i]
#         if i != 0:
#             pre_G[4 + i * 3] = ord_eq[i]
#         post_G = G[eq_ix[i]]
#         post_G[[0, 4 + i * 3]] = ord_eq[i]

#         offset = np.matmul(post_G, x) - np.matmul(pre_G, x)
#         print('    Co-seismic offset and log param for {}: {:.0f} mm, {:.2f}'.format(eq_dates[i], offset, x[3 + i * 3]))

#     plot_trend += 1

# # %%
# #plt.scatter(date_all, disp_all, s=2, c='red')
# plt.scatter(date_all, disp, s=2, c='blue')  # , label='Displacement')
# #plt.plot(date_all, lin, label='Linear Fit')
# if enddate > eq_date:
# #    if plot_trend > 0:
# #        plt.plot(date_all, heavy, label='Linear w/ coseismic')
# #    if plot_trend > 1:
# #        plt.plot(date_all, split, label='Multi Linear w/ coseismic')
#     if plot_trend > 2:
#         plt.plot(date_all, invvel, label='Const Linear w/ co- + post-seismic')
# plt.legend()
# for i in range(0, n_eq):
#     plt.axvline(x=eq_dt[i], color="grey", linestyle="--")
# #plt.title('Pix = ({}, {}), Reference Area = [{}:{}, {}:{}]'.format(pix_x, pix_y, ref_x[0], ref_x[1], ref_y[0], ref_y[1]))
# plt.show()
