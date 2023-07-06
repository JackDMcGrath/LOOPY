#!/usr/bin/env python3
"""
========
Overview
========
This script sets up an iterative loop to perform time series inversion and automatic unwrapping correction until time series reaches good quality

===============
Input & output files
===============
Inputs in GEOCml*/ :
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw
   - yyyymmdd_yyyymmdd.cc
   - yyyymmdd_yyyymmdd.conncomp
 - EQA.dem_par
 - slc.mli.par
 - baselines (may be dummy)
[- [ENU].geo]

Inputs in TS_GEOCml*/ :
 - info/
   - 11bad_ifg.txt
   - 12bad_ifg.txt
   - 120ref.txt
[-results/]
[  - coh_avg]
[  - hgt]
[  - n_loop_err]
[  - n_unw]
[  - slc.mli]

Outputs in TS_GEOCml*/ :
 - cum*.h5             : Cumulative displacement (time-seires) in mm
 - results*/
   - vel[.png]        : Velocity in mm/yr (positive means LOS decrease; uplift)
   - vintercept[.png] : Constant part of linear velocity (c for vt+c) in mm
   - resid_rms[.png]  : RMS of residual in mm
   - n_gap[.png]      : Number of gaps in SB network
   - n_ifg_noloop[.png] :  Number of ifgs with no loop
   - maxTlen[.png]    : Max length of continous SB network in year
 - info/
   - 13parameters.txt : List of used parameters
   - 13used_image.txt : List of used images
   - 13resid.txt      : List of RMS of residual for each ifg
   - 13ref.txt[kml]   : Auto-determined stable ref point
   - 13rms_cum_wrt_med[.png] : RMS of cum wrt median used for ref selection
 - 13increment*/yyyymmdd_yyyymmdd.increment.png
     : Comparison between unw and inverted incremental displacement
 - 13resid*/yyyymmdd_yyyymmdd.res.png : Residual for each ifg
 - network/network13*.png : Figures of the network

=====
Usage
=====
"""
#%% Import
import os
import time
import shutil
import numpy as np
import glob
import h5py as h5
from pathlib import Path
import argparse
import sys
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib


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
    parser.add_argument('-f', dest="frame_dir", default="./", help="directory of LiCSBAS output of a particular frame")
    parser.add_argument('-g', dest='unw_dir', default="GEOCml10GACOS", help="folder containing unw input")
    parser.add_argument('-t', dest='ts_dir', default="TS_GEOCml10GACOS", help="folder containing time series")
    parser.add_argument('-p', dest='percentile', type=float, help="percentile RMS for thresholding")
    args = parser.parse_args()


def start():
    global start_time
    # intialise and print info on screen
    start_time = time.time()
    ver="1.0"; date=20221020; author="Qi Ou"
    print("\n{} ver{} {} {}".format(os.path.basename(sys.argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)


def finish():
    #%% Finish
    elapsed_time = time.time() - start_time
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))
    print("\n{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)
    print('Output directory: {}\n'.format(os.path.relpath(tsadir)))


def set_input_output():
    global ifgdir, tsadir, infodir, ccdir, netdir
    # define input directories
    ccdir = args.unw_dir
    ifgdir = os.path.abspath(os.path.join(args.frame_dir, ccdir))  # to read .unw
    tsadir = os.path.abspath(os.path.join(args.frame_dir, args.ts_dir))   # to read 120.ref, to write cum.h5
    infodir = os.path.join(tsadir, 'info')  # to read 11bad_ifg.txt, 12bad_ifg.txt
    netdir = os.path.join(tsadir, 'network')


def get_ifgdates():
    global ifgdates
    # bad_ifg11file = os.path.join(infodir, '11bad_ifg.txt')
    # bad_ifg12file = os.path.join(infodir, '120bad_ifg.txt')
    #
    # #%% Read date and network information
    # ### Get all ifgdates in ifgdir
    # ifgdates_all = tools_lib.get_ifgdates(ifgdir)
    #
    # ### Remove bad_ifg11 and 120bad_ifg
    # bad_ifg11 = io_lib.read_ifg_list(bad_ifg11file)
    # bad_ifg12 = io_lib.read_ifg_list(bad_ifg12file)
    # bad_ifg_all = list(set(bad_ifg11+bad_ifg12))
    # bad_ifg_all.sort()
    #
    # ifgdates = list(set(ifgdates_all)-set(bad_ifg_all))
    # ifgdates.sort()

    strong_ifgfile = os.path.join(infodir, '120strong_connected_links.txt')
    ifgdates = io_lib.read_ifg_list(strong_ifgfile)
    ifgdates.sort()


def first_iteration(iter_unw_path):
    ''' Only link "good" ifgs defined by ifgdates to the folder'''
    # remove existing GEOCml10GACOS1 directory
    if os.path.exists(iter_unw_path): shutil.rmtree(iter_unw_path)
    Path(iter_unw_path).mkdir(parents=True, exist_ok=True)

    # Link unw
    for pair in ifgdates:
        pair_dir = os.path.join(iter_unw_path, pair)
        Path(pair_dir).mkdir(parents=True, exist_ok=True)
        unwfile = os.path.join(ifgdir, pair, pair + '.unw')
        linkfile = os.path.join(pair_dir, pair + '.unw')
        os.link(unwfile, linkfile)


def one_correction():
    # define first iteration output dir
    current_iter = 1
    current_iter_unwdir = ccdir+"{}".format(int(current_iter))
    current_iter_unw_abspath = os.path.abspath(os.path.join(args.frame_dir, current_iter_unwdir))  # to read .unw
    next_iter = current_iter + 1
    next_iter_unwdir = ccdir + "{}".format(int(next_iter))
    masking_iter = next_iter + 1
    masking_unwdir = ccdir + "{}".format(int(masking_iter))

    # set up directories for the first iteration, linking good unw to a separate folder:
    first_iteration(current_iter_unw_abspath)

    # run one ts inversion, followed by one correction and one more ts inversion
    run_130(current_iter_unwdir, current_iter)
    run_131(current_iter)

    ifg_to_correct, good_ifg = get_good_bad_list(current_iter)
    if len(ifg_to_correct) == 0:
        run_133(current_iter)
    else:
        run_132(current_iter_unwdir, next_iter_unwdir, current_iter)
        # run_130(next_iter_unwdir, next_iter)
        # run_131(next_iter)
        # run_133(next_iter)

        # run_131(next_iter)
        # run_masking(next_iter_unwdir, masking_unwdir, next_iter)
        #
        # run_130(masking_unwdir, masking_iter)
        # run_131(masking_iter)
        #
        # # compile all results into cum.h5
        # run_133(masking_iter)


def get_good_bad_list(current_iter):
    stats_file = os.path.join(infodir, '131resid_2pi{}.txt'.format(current_iter))
    correction_thresh = float(io_lib.get_param_par(stats_file, 'RMS_thresh'))
    ifg_to_correct = []
    good_ifg = []
    with open(stats_file) as f:
        line = f.readline()
        if line.startswith("2"):
            ifg, rms = line.split()
            if float(rms) > correction_thresh:
                ifg_to_correct.append(ifg)
            else:
                good_ifg.append(ifg)
    return ifg_to_correct, good_ifg


def run_130(unw_dir, current_iter):
    os.system('LiCSBAS130_sb_inv.py -c {} -d {} -t {} --suffix {} --inv_alg WLS --keep_incfile --nopngs'.format(
        ccdir, unw_dir, args.ts_dir, int(current_iter)))


def run_131(current_iter):
    if args.percentile:
        os.system('LiCSBAS131_residual_threshold.py -g {} -t {} -p {} --suffix {} '.format(
            ccdir, args.ts_dir, args.percentile, int(current_iter)))
    else:
        os.system('LiCSBAS131_residual_threshold.py -g {} -t {} --suffix {} '.format(ccdir, args.ts_dir, int(current_iter)))


def run_132(before_dir, after_dir, current_iter):
    os.system('LiCSBAS132_3D_correction.py -c {} -d {} -o {} -t {} --suffix {} -s 0.2'.format(
        ccdir, before_dir, after_dir, args.ts_dir, int(current_iter)))


def run_133(current_iter):
    os.system('LiCSBAS133_write_h5.py -c {} -t {} --suffix {} '.format(
        ccdir, args.ts_dir, int(current_iter)))


def main():
    start()
    init_args()
    set_input_output()
    get_ifgdates()
    one_correction()
    finish()


if __name__ == "__main__":
    main()
