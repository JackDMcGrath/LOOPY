#!/usr/bin/env python3
"""
========
Overview
========
Based on QiCSBAS. Takes the aggressively masked nullified velocity field, interpolates the velocity field, and finds the residual

===============
Input & output files
===============

=====
Usage
=====
LiCSBAS132_3D_correction.py [-h] [-f FRAME_DIR] [-c COMP_CC_DIR] [-g UNW_DIR]
                                   [-r CORRECT_DIR] [-t TS_DIR] [--thresh THRESH] [--suffix SUFFIX]
"""

from scipy import stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import copy
import glob
import argparse
import sys
import time
import re
from pathlib import Path
#import cmcrameri as cm
from matplotlib import cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib
import LiCSBAS_inv_lib as inv_lib
import shutil
import multiprocessing as multi
from scipy import interpolate
import SCM
from scipy.ndimage import binary_opening, binary_closing
from scipy.interpolate import NearestNDInterpolator
from skimage.morphology import disk

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    '''
    Use a multiple inheritance approach to use features of both classes.
    The ArgumentDefaultsHelpFormatter class adds argument default values to the usage help message
    The RawDescriptionHelpFormatter class keeps the indentation and line breaks in the ___doc___
    '''
    pass

def init_args():
    global args

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=CustomFormatter)
    parser.add_argument('-f', dest='frame_dir', default="./", help="directory of LiCSBAS output of a particular frame")
    parser.add_argument('-d', dest='unw_dir', default="GEOCml10GACOS", help="folder containing unw input to be corrected")
    parser.add_argument('-o', dest='out_dir', default="GEOCml10GACOS_corrected/masked", help="folder for corrected/masked unw")
    parser.add_argument('-t', dest='ts_dir', default="TS_GEOCml10GACOS", help="folder containing time series and residuals")
    parser.add_argument('-s', dest='correction_thresh', type=float, help="RMS residual per ifg (in 2pi) for correction, override info/131resid_2pi.txt")
    parser.add_argument('-g', dest='target_thresh', default='thresh', choices=['mode', 'median', 'mean', 'thresh'], help="RMS residual per ifg (in 2pi) for accepting the correction, read from info/131resid_2pi.txt, or follow correction_thresh if given")
    parser.add_argument('-l', dest='ifg_list', default=None, type=str, help="text file containing a list of ifgs to be corrected")
    parser.add_argument('-n', dest='n_para', type=int, help="number of processes for parallel processing")
    parser.add_argument('--suffix', default="", type=str, help="suffix of the input 131resid_2pi*.txt and outputs")
    parser.add_argument('--apply_mask',  action='store_true', help="use masking of uncertian corrections (assign through -m)")
    parser.add_argument('-m', dest='mask_thresh', type=float, default=0.2, help="RMS residual per ifg (in 2pi) for correction")
    parser.add_argument('--no_depeak', default=False, action='store_true', help="don't offset by residual mode before calculation (recommend depeak)")
    parser.add_argument('--interp', default='Linear', choices=['Linear', 'Cubic'], help="Interpolation Method of masked velocities")
    parser.add_argument('--nonan', default=False, action='store_true', help="Set uncertain corrections to 0 rather than nan when masking")
    parser.add_argument('--filter', dest='filter', default=False, action='store_true', help="Run binary filtering of the correction" )
    args = parser.parse_args()

    if '/' in args.out_dir:
        if args.apply_mask:
            args.out_dir = args.unw_dir + '_masked'
        else:
            args.out_dir = args.unw_dir + '_corrected'

def load_vel(ifgd, length, width):
    vel_mm = io_lib.read_img(os.path.join(resdir, ifgd + '.ifg'), length, width)
    if np.isnan(np.nanmedian(vel_mm[refy1:refy2, refx1:refx2])):
        vel_mm = vel_mm - np.nanmedian(vel_mm)
    else:
        vel_mm = vel_mm - np.nanmedian(vel_mm[refy1:refy2, refx1:refx2])

    vel_mm[np.where(TSmask == 0)] = np.nan
    pngfile = os.path.join(resdir, ifgd + '.masked.png')
    title = 'Masked {} IFG'.format(ifgd)
    plot_lib.make_im_png(vel_mm, pngfile, SCM.roma.reversed(), title, np.nanmin(vel_mm), np.nanmax(vel_mm))
    X = np.arange(width)
    Y = np.arange(length)
    X, Y = np.meshgrid(X, Y)
    ix_non_nan = np.where(~np.isnan(vel_mm.flatten()))[0]
    interp_to = np.where(TSmask == 0)

    if args.interp == 'Linear':
        interp = interpolate.LinearNDInterpolator(list(zip(X.flatten()[ix_non_nan], Y.flatten()[ix_non_nan])), vel_mm.flatten()[ix_non_nan])
    elif args.interp == 'Cubic':
        interp = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten()[ix_non_nan], Y.flatten()[ix_non_nan])), vel_mm.flatten()[ix_non_nan])

    vel_interp = interp(X[interp_to], Y[interp_to])
    vel_mm[interp_to] = vel_interp

    pngfile = os.path.join(resdir, ifgd + '.interp.png')
    title = 'Masked + Interpolated {} IFG'.format(ifgd)
    plot_lib.make_im_png(vel_mm, pngfile, SCM.roma.reversed(), title, np.nanmin(vel_mm), np.nanmax(vel_mm))
    vel_rad = vel_mm / coef_r2m

    ifg = io_lib.read_img(os.path.join(args.unw_dir, ifgd, ifgd  + '.unw'), length, width)
    if np.isnan(np.nanmedian(ifg[refy1:refy2, refx1:refx2])):
        ifg = ifg - np.nanmedian(ifg)
    else:
        ifg = ifg - np.nanmedian(ifg[refy1:refy2, refx1:refx2])

    res_num_2pi = (ifg - vel_rad) / 2 / np.pi
    if not args.no_depeak:
        counts, bins = np.histogram(res_num_2pi, np.arange(-2.5, 2.6, 0.1))
        peak = bins[counts.argmax()] + 0.05
        res_num_2pi = res_num_2pi - peak
    res_rms = np.sqrt(np.nanmean(res_num_2pi ** 2))

    return res_num_2pi, res_rms

def load_res(res_file, length, width):
    res_mm = np.fromfile(res_file, dtype=np.float32).reshape((length, width))
    res_rad = res_mm / coef_r2m
    res_num_2pi = res_rad / 2 / np.pi
    if not args.no_depeak:
        counts, bins = np.histogram(res_num_2pi, np.arange(-2.5, 2.6, 0.1))
        peak = bins[counts.argmax()] + 0.05
        res_num_2pi = res_num_2pi - peak
    res_rms = np.sqrt(np.nanmean(res_num_2pi ** 2))
    del res_mm, res_rad
    return res_num_2pi, res_rms

def correcting_by_integer(reslist):
    for i in reslist:
        pair = os.path.basename(i).split('.')[0][-17:]
        unwfile = os.path.join(unwdir, pair, pair + '.unw')
        unw = np.fromfile(unwfile, dtype=np.float32).reshape((length, width))
        # define output dir
        correct_pair_dir = os.path.join(correct_dir, pair)
        Path(correct_pair_dir).mkdir(parents=True, exist_ok=True)

        cycle = 3

        if i in corr_list:
            # print(pair, 'is being corrected')
            # calc component mode
            #res_num_2pi, res_rms = load_res(i, length, width)
            res_num_2pi, res_rms = load_vel(i, length, width)
            res_integer = np.round(res_num_2pi)

            # Set Areas where theres is not a correction to 0 (preserving size of overall field)
            zeromask = np.where(np.logical_and(~np.isnan(unw), np.isnan(res_integer)))
            res_integer[zeromask] = 0

            if args.filter:
                res_integer[np.isnan(res_integer)] = 0
                res_integer = binary_filter(res_integer, pair)
                res_integer[np.isnan(unw)] = np.nan

            rms_res_integer_corrected = np.sqrt(np.nanmean((res_num_2pi - res_integer) ** 2))

            # correcting by component mode
            unw_corrected = unw - res_integer * 2 * np.pi

            # turn uncertain correction into masking
            # mask1 = np.logical_and(abs(res_num_2pi) > 0.2, abs(res_num_2pi) < 0.8)
            # mask2 = np.logical_and(abs(res_num_2pi) > 1.2, abs(res_num_2pi) < 1.8)
            # mask = np.logical_or(mask1, mask2)
            mask = np.zeros(unw.shape).astype(bool)
            mask[np.where(~np.isnan(res_num_2pi))] = np.logical_and(np.mod(abs(res_num_2pi[np.where(~np.isnan(res_num_2pi))]), 1) > 0.2, np.mod(abs(res_num_2pi[np.where(~np.isnan(res_num_2pi))]),1) < 0.8)
            res_mask = copy.copy(res_integer)
            if args.nonan:
                res_mask[mask] = 0
            else:
                res_mask[mask] = np.nan
            unw_masked = unw - res_mask * 2 * np.pi
            rms_res_mask_corrected = np.sqrt(np.nanmean((res_num_2pi - res_mask) ** 2))

            # plotting
            png_path = os.path.join(integer_png_dir, '{}.png'.format(pair))
            plot_correction_by_integer(pair, unw, unw_corrected, unw_masked, res_mask, res_num_2pi, res_integer, res_rms,
                                       rms_res_integer_corrected, rms_res_mask_corrected, png_path)

            # save the corrected unw
            if not args.apply_mask:
                unw_corrected.flatten().tofile(os.path.join(correct_pair_dir, pair + '.unw'))
                plot_lib.make_im_png(np.angle(np.exp(1j*unw_corrected/cycle)*cycle), os.path.join(correct_pair_dir, pair + '.unw.png'), SCM.roma, pair + '.unw', vmin=-np.pi, vmax=np.pi, cbar=False)
            else:
                unw_masked.flatten().tofile(os.path.join(correct_pair_dir, pair + '.unw'))
                plot_lib.make_im_png(np.angle(np.exp(1j*unw_masked/cycle)*cycle), os.path.join(correct_pair_dir, pair + '.unw.png'), SCM.roma, pair + '.unw', vmin=-np.pi, vmax=np.pi, cbar=False)
            del mask, res_mask, unw_masked, rms_res_mask_corrected
        else:
            print(pair, 'not in corr_list')
            unw.flatten().tofile(os.path.join(correct_pair_dir, pair + '.unw'))
            plot_lib.make_im_png(np.angle(np.exp(1j*unw/cycle)*cycle), os.path.join(correct_pair_dir, pair + '.unw.png'), SCM.roma, pair + '.unw', vmin=-np.pi, vmax=np.pi, cbar=False)

def binary_filter(correction, pair):

    corr_val, corr_count = np.unique(correction, return_counts=True)
    corr_order = np.argsort(corr_count)
        
    # Find full region where correction could be needed, and remove random noise
    corr_grid = np.zeros((length, width))
    corr_grid [np.where(correction != 0)] = 1
    corr_grid = binary_closing(corr_grid, structure=disk(radius=2)).astype('int')  # Fill in any holes
    corr_grid = binary_opening(corr_grid, structure=disk(radius=1)).astype(np.float32)  # Remove wild spikes
    plt.imshow(corr_grid)
    plt.savefig(os.path.join(integer_png_dir, '{}_corr_grid.png'.format(pair)))

    # Make variable to store correction
    corrFilt = np.zeros((length, width))
    corrFilt[np.where(np.isnan(correction))] = np.nan
    corrFilt[np.where(corr_grid == 1)] = np.nan

    # Filter each correction value to reduce noise
    for corr in corr_order:
        if corr_val[corr] == 0 or np.isnan(corr_val[corr]):
            continue

        grid = np.zeros((length, width))
        grid[np.where(correction == corr_val[corr])] = 1  # Find all areas that have a correction
        grid = binary_closing(grid, structure=disk(radius=2)).astype('int')  # Fill in any holes
        grid = binary_opening(grid, structure=disk(radius=1)).astype('int')  # Remove wild spikes
        corrFilt[np.where(np.logical_and(grid == 1, correction == corr_val[corr]))] = corr_val[corr]

    plt.imshow(corrFilt, vmin=-2, vmax=2, cmap=cm.RdBu)
    plt.savefig(os.path.join(integer_png_dir, '{}_corrFilt1.png'.format(pair)))
    # Interpolate filtered corrections to identified correction region
    mask = np.where(~np.isnan(corrFilt))  # < this is the good data
    interp = NearestNDInterpolator(np.transpose(mask), corrFilt[mask])  # Create interpolator
    interp_to = np.where(corr_grid == 1)  # Find where to interpolate to
    corrFilt[interp_to] = interp(*interp_to)  # Apply corrected data
    plt.imshow(corrFilt, vmin=-2, vmax=2, cmap=cm.RdBu)
    plt.savefig(os.path.join(integer_png_dir, '{}_corrFilt2.png'.format(pair)))

    return corrFilt

def start():
    global start_time
    # intialise and print info on screen
    start_time = time.time()
    ver="1.0"; date=20221020; author="Qi Ou and Jack McGrath"
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
    print('Output directory: {}\n'.format(os.path.relpath(correct_dir)))

def set_input_output():
    global unwdir, tsadir, resdir, infodir, netdir, correct_dir, integer_png_dir, resultdir

    # define input directories
    unwdir = os.path.abspath(os.path.join(args.frame_dir, args.unw_dir))
    tsadir = os.path.abspath(os.path.join(args.frame_dir, args.ts_dir))
    resdir = os.path.join(tsadir, '130resid')
    infodir = os.path.join(tsadir, 'info')
    resultdir = os.path.join(tsadir, 'results')

    # define output directories
    netdir = os.path.join(tsadir, 'network')

    # perform correction
    correct_dir = os.path.abspath(os.path.join(args.frame_dir, args.out_dir))
    if os.path.exists(correct_dir): shutil.rmtree(correct_dir)
    Path(correct_dir).mkdir(parents=True, exist_ok=True)

    integer_png_dir = os.path.join(resdir, 'integer_correction/')
    if os.path.exists(integer_png_dir): shutil.rmtree(integer_png_dir)
    Path(integer_png_dir).mkdir(parents=True, exist_ok=True)

def get_para():
    global width, length, coef_r2m, refx1, refx2, refy1, refy2, n_para, res_list, corr_list

    # read ifg size and satellite frequency
    mlipar = os.path.join(args.unw_dir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))
    radar_frequency = float(io_lib.get_param_par(mlipar, 'radar_frequency'))  # 5405000000.0 Hz for C-band
    speed_of_light = 299792458  # m/s
    wavelength = speed_of_light/radar_frequency
    coef_r2m = -wavelength/4/np.pi*1000

    # read reference for plotting purpose
    reffile = os.path.join(infodir, '120ref.txt')
    with open(reffile, "r") as f:
        refarea = f.read().split()[0]  # str, x1/x2/y1/y2
    refx1, refx2, refy1, refy2 = [int(s) for s in re.split('[:/]', refarea)]

    # multi-processing
    if not args.n_para:
        try:
            n_para = min(len(os.sched_getaffinity(0)), 8) # maximum use 8 cores
        except:
            n_para = multi.cpu_count()
    else:
        n_para = args.n_para

    # check interferograms exist
    res_list = glob.glob(os.path.join(resdir, '*.res'))
    res_list = [os.path.basename(res[:-4]) for res in res_list]
    if args.ifg_list:
        corr_list = io_lib.read_ifg_list(args.ifg_list)
        print('Corr_list is args.ifg_list')
    else:
        corr_list = res_list
        print('Corr_list is res_list')
    if len(res_list) == 0:
        sys.exit('No ifgs for correcting...\nCheck if there are *res files in the directory {}'.format(resdir))

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

def plot_correction_by_integer(pair, unw, unw_corrected, unw_masked, res_mask, res_num_2pi, res_integer, res_rms, rms_res_integer_corrected, rms_res_mask_corrected, png_path):
    fig, ax = plt.subplots(2, 3, figsize=(9, 5))
    fig.suptitle(pair)
    for x in ax[:, :].flatten():
        x.axes.xaxis.set_ticklabels([])
        x.axes.yaxis.set_ticklabels([])
    unw_vmin = np.nanpercentile(unw, 0.5)
    unw_vmax = np.nanpercentile(unw, 99.5)
    im_unw = ax[0, 0].imshow(unw, vmin=unw_vmin, vmax=unw_vmax, cmap=cm.RdBu, interpolation='nearest')
    im_unw = ax[0, 1].imshow(unw_corrected, vmin=unw_vmin, vmax=unw_vmax, cmap=cm.RdBu, interpolation='nearest')
    im_unw = ax[0, 2].imshow(unw_masked, vmin=unw_vmin, vmax=unw_vmax, cmap=cm.RdBu, interpolation='nearest')
    im_res = ax[1, 0].imshow(res_num_2pi, vmin=-2, vmax=2, cmap=cm.RdBu, interpolation='nearest')
    im_res = ax[1, 1].imshow(res_integer, vmin=-2, vmax=2, cmap=cm.RdBu, interpolation='nearest')
    im_res = ax[1, 2].imshow(res_mask, vmin=-2, vmax=2, cmap=cm.RdBu, interpolation='nearest')
    ax[1, 0].plot([refx1, refx1, refx2, refx2, refx1], [refy1, refy2, refy2, refy1, refy1], c='r')
    ax[0, 0].set_title("Unw (rad)")
    ax[0, 1].set_title("Correct by Integer")
    ax[0, 2].set_title("Correct with Mask")
    ax[1, 0].set_title("Residual/2pi (RMS={:.2f})".format(res_rms))
    ax[1, 1].set_title("Nearest Integer ({:.2f})".format(rms_res_integer_corrected))
    ax[1, 2].set_title("Masked Integer ({:.2f})".format(rms_res_mask_corrected))
    fig.colorbar(im_unw, ax=ax[0, :], location='right', shrink=0.8)
    fig.colorbar(im_res, ax=ax[1, :], location='right', shrink=0.8)
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

def move_metafiles():
    metafiles = glob.glob(os.path.join(unwdir, '*.*'))
    files = ['baselines', 'hgt']
    for file in files:
        if os.path.exists(os.path.join(unwdir, file)):
            metafiles.append(os.path.join(unwdir, file))

    print('Soft Linking Metadata....')
    for file in metafiles:
        if not os.path.exists(os.path.join(correct_dir, os.path.basename(file))):
            os.symlink(file, os.path.join(correct_dir, os.path.basename(file)))

    print('Soft Linking Coherences....')
    for ifg in res_list:
        pair = os.path.basename(ifg)
        ccfile = os.path.join(unwdir, pair, pair + '.cc')
        if not os.path.exists(os.path.join(correct_dir, pair, pair + '.cc')):
            os.symlink(ccfile, os.path.join(correct_dir, pair, pair + '.cc'))

def main():
    start()
    init_args()
    set_input_output()
    get_para()
    integer_correction()
    move_metafiles()

    finish()


if __name__ == "__main__":
    main()
