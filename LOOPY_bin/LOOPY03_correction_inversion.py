#!/usr/bin/env python3
"""
v1.5.2 20210311 Yu Morishita, GSI

This script inverts the SB network of unw to obtain the loop closure correction
for each interferogram, then applies the correction, using the L1 Regularisation
(Yunjun et al., 2019).
Parallised on a pixel-by-pixel basis, rather than patches, due to size of G-matrix
that would be required by L1reg

===============
Input & output files
===============
Inputs in GEOCml*/:
 -yyyymmdd_yyyymmdd/
   -yyyymmdd_yyyymmdd.unw
 -EQA.dem_par
 -slc.mli.par

Inputs in TS_GEOCml*/ :
 -info/
   -11bad_ifg.txt
   -ref.txt (if non-supplied, it searches for 13ref.txt then 12ref.txt)
[-results/]
[  -coh_avg]

=====
Usage
=====
LOOPY03_correction_inversion.py -d ifgdir [-t tsadir] [--mem_size float] [--gamma float] [--n_para int] [--n_unw_r_thre float]

 -d  Path to the GEOCml* dir containing stack of unw data
 -t  Path to the output TS_GEOCml* dir.
 --mem_size   Max memory size for each patch in MB. (Default: 8000)
 --gamma      Gamma value for L1 Regulariastion (Default: 0.001)
 --n_para     Number of parallel processing (Default:  # of usable CPU)
 --n_unw_r_thre
     Threshold of n_unw (number of used unwrap data)
     (Note this value is ratio to the number of images; i.e., 1.5*n_im)
     Larger number (e.g. 2.5) makes processing faster but result sparser.
     (Default: 1 and 0.5 for C- and L-band, respectively)

"""
# %% Change log
'''
v1.0.0 20230321 Jack McGrath, Uni of Leeds
 - Initial implementation based of LiCSBAS13_invert_small_baselines.py
'''

# %% Import
import getopt
import os
import sys
import re
import time
import psutil
import numpy as np
import multiprocessing as multi
import SCM
import LiCSBAS_io_lib as io_lib
import LiCSBAS_inv_lib as inv_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_loop_lib as loop_lib
import LiCSBAS_plot_lib as plot_lib
import LOOPY_L1reg_lib as l1_lib
from cvxopt import matrix


class Usage(Exception):
    """Usage context manager"""

    def __init__(self, msg):
        self.msg = msg


# %% Main
def main(argv=None):

    # %% Check argv
    if argv is None:
        argv = sys.argv

    start = time.time()
    ver = "1.5.2"
    date = 20210311
    author = "Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)

    # For parallel processing
    global n_para_gap, G, Aloop, unwpatch, imdates, incdir, ifgdir, length, width,\
        coef_r2m, ifgdates, ref_unw, cycle, keep_incfile, resdir, restxtfile, \
        cmap_vel, cmap_wrap, wavelength, refx1, refx2, refy1, refy2

    # %% Set default
    ifgdir = []
    tsadir = []
    v = -1

    try:
        n_para = len(os.sched_getaffinity(0))
    except:
        n_para = multi.cpu_count()

    os.environ["OMP_NUM_THREADS"] = "1"
    # Because np.linalg.lstsq use full CPU but not much faster than 1CPU.
    # Instead parallelize by multiprocessing

    gamma = 0.001
    n_unw_r_thre = []

    cmap_vel = SCM.roma.reversed()
    cmap_noise = 'viridis'
    cmap_noise_r = 'viridis_r'
    cmap_wrap = SCM.romaO

    # %% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hd:t:v:",
                                       ["help", "gamma=",
                                        "n_unw_r_thre=", "n_para="])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-d':
                ifgdir = a
            elif o == '-t':
                tsadir = a
            elif o == '-v':
                v = float(a)
            elif o == '--gamma':
                gamma = float(a)
            elif o == '--n_unw_r_thre':
                n_unw_r_thre = float(a)
            elif o == '--n_para':
                n_para = int(a)

        if not ifgdir:
            raise Usage('No data directory given, -d is not optional!')
        elif not os.path.isdir(ifgdir):
            raise Usage('No {} dir exists!'.format(ifgdir))
        elif not os.path.exists(os.path.join(ifgdir, 'slc.mli.par')):
            raise Usage('No slc.mli.par file exists in {}!'.format(ifgdir))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  " + str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2

    if sys.platform == "linux" or sys.platform == "linux2":
        q = multi.get_context('fork')
    elif sys.platform == "win32":
        q = multi.get_context('spawn')
        n_para = 1
        print('WINDOWS CANNOT USE GLOBAL VARIABLES IN MULTIPROCEESSING. SETTING N_PARA to 1')

    # %% Directory settings
    ifgdir = os.path.abspath(ifgdir)

    if not tsadir:
        tsadir = os.path.join(os.path.dirname(ifgdir), 'TS_' + os.path.basename(ifgdir))

    if not os.path.isdir(tsadir):
        print('\nNo {} exists!'.format(tsadir), file=sys.stderr)
        return 1

    tsadir = os.path.abspath(tsadir)
    infodir = os.path.join(tsadir, 'info')

    bad_ifg11file = os.path.join(infodir, '11bad_ifg.txt')
    reffile = os.path.join(infodir, 'ref.txt')
    if not os.path.exists(reffile):
        reffile = os.path.join(infodir, '13ref.txt')
    if not os.path.exists(reffile):
        reffile = os.path.join(infodir, '12ref.txt')

    # %% Check files
    try:
        if not os.path.exists(bad_ifg11file):
            raise Usage('No 11bad_ifg.txt file exists in {}!'.format(infodir))
        if not os.path.exists(reffile):
            raise Usage('No ref.txt file exists in {}!'.format(infodir))
        else:
            print('Ref file: {}'.format(reffile))
    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  " + str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2

    # %% Set preliminaly reference
    with open(reffile, "r") as f:
        refarea = f.read().split()[0]  # str, x1/x2/y1/y2
    refx1, refx2, refy1, refy2 = [int(s) for s in re.split('[:/]', refarea)]

    # %% Read data information
    # Get size
    mlipar = os.path.join(ifgdir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))
    speed_of_light = 299792458  # m/s
    radar_frequency = float(io_lib.get_param_par(mlipar, 'radar_frequency'))  # Hz
    wavelength = speed_of_light / radar_frequency  # meter
    coef_r2m = -wavelength / 4 / np.pi * 1000  # rad -> mm, positive is -LOS

    # Set n_unw_r_thre and cycle depending on L- or C-band
    if wavelength > 0.2:  # L-band
        if not n_unw_r_thre:
            n_unw_r_thre = 0.5
        cycle = 1.5  # 2pi/cycle for comparison png
    elif wavelength <= 0.2:  # C-band
        if not n_unw_r_thre:
            n_unw_r_thre = 1.0
        cycle = 3  # 3*2pi/cycle for comparison png

    # %% Read date and network information
    # Get all ifgdates in ifgdir
    ifgdates_all = tools_lib.get_ifgdates(ifgdir)
    imdates_all = tools_lib.ifgdates2imdates(ifgdates_all)
    n_im_all = len(imdates_all)
    n_ifg_all = len(ifgdates_all)

    # Read bad_ifg11
    bad_ifg11 = io_lib.read_ifg_list(bad_ifg11file)
    bad_ifg_all = list(set(bad_ifg11))
    bad_ifg_all.sort()

    # Remove bad ifgs and images from list
    ifgdates = list(set(ifgdates_all) - set(bad_ifg_all))
    ifgdates.sort()

    imdates = tools_lib.ifgdates2imdates(ifgdates)

    n_ifg = len(ifgdates)
    n_ifg_bad = len(set(bad_ifg11))
    n_im = len(imdates)
    n_unw_thre = int(n_unw_r_thre * n_im)

    # Make 13used_image.txt
    imfile = os.path.join(infodir, '13used_image.txt')
    with open(imfile, 'w') as f:
        for i in imdates:
            print('{}'.format(i), file=f)

    # Construct G and Aloop matrix for increment and n_gap
    G = inv_lib.make_sb_matrix(ifgdates)
    Aloop = loop_lib.make_loop_matrix(ifgdates)
    n_loop = Aloop.shape[0]

    # Extract no loop ifgs
    ns_loop4ifg = np.abs(Aloop).sum(axis=0)
    ixs_ifg_no_loop = np.where(ns_loop4ifg == 0)[0]
    no_loop_ifg = [ifgdates[ix] for ix in ixs_ifg_no_loop]

    # %% Display and output settings & parameters
    print('')
    print('Size of image (w,l)    : {}, {}'.format(width, length))
    print('# of all images        : {}'.format(n_im_all))
    print('# of images to be used : {}'.format(n_im))
    print('# of all ifgs          : {}'.format(n_ifg_all))
    print('# of ifgs to be used   : {}'.format(n_ifg))
    print('# of removed ifgs      : {}'.format(n_ifg_bad))
    print('Threshold of used unw  : {}'.format(n_unw_thre))
    print('')
    print('Reference area (X/Y)   : {}:{}/{}:{}'.format(refx1, refx2, refy1, refy2))
    print('Gamma value            : {}'.format(gamma), flush=True)

    # %% Read data in parallel
    _n_para = n_para if n_para < n_ifg else n_ifg
    print("  Reading {0} ifg's unw data...".format(n_ifg), flush=True)

    # Allocate memory
    unw = np.zeros((n_ifg, length, width), dtype=np.float32)

    if n_para == 1:
        print('with no parallel processing...', flush=True)

        for ii in range(n_ifg):
            unw[ii, :, :] = read_unw(ii)

    else:
        print('with {} parallel processing...'.format(_n_para), flush=True)
        # Parallel processing
        p = q.Pool(_n_para)
        unw = np.array(p.map(read_unw, range(n_ifg)))
        p.close()


    # # %% For each patch
    # for i_patch, rows in enumerate(patchrow):
    #     print('\nProcess {0}/{1}th line ({2}/{3}th patch)...'.format(rows[1], patchrow[-1][-1], i_patch + 1, n_patch), flush=True)
    #     start2 = time.time()

    #     # %% Read data
    #     # Allocate memory
    #     lengththis = rows[1] - rows[0]
    #     n_pt_all = lengththis * width
    #     unwpatch = np.zeros((n_ifg, lengththis, width), dtype=np.float32)

    #     # For each ifg
    #     print("  Reading {0} ifg's unw data...".format(n_ifg), flush=True)
    #     countf = width * rows[0]
    #     countl = width * lengththis
    #     for i, ifgd in enumerate(ifgdates):
    #         unwfile = os.path.join(ifgdir, ifgd, ifgd + '.unw')
    #         f = open(unwfile, 'rb')
    #         f.seek(countf * 4, os.SEEK_SET)  # Seek for >=2nd patch, 4 means byte

    #         # Read unw data (mm) at patch area
    #         unw = np.fromfile(f, dtype=np.float32, count=countl).reshape((lengththis, width)) * coef_r2m
    #         unw[unw == 0] = np.nan  # Fill 0 with nan
    #         unw = unw - ref_unw[i]
    #         unwpatch[i] = unw
    #         f.close()

    #     unwpatch = unwpatch.reshape((n_ifg, n_pt_all)).transpose()  # (n_pt_all, n_ifg)

    #     # %% Remove points with less valid data than n_unw_thre
    #     ix_unnan_pt = np.where(np.sum(~np.isnan(unwpatch), axis=1) > n_unw_thre)[0]
    #     n_pt_unnan = len(ix_unnan_pt)
    #     corrFull = np.zeros(unwpatch.shape) * np.nan
    #     unwpatch = unwpatch[ix_unnan_pt, :]  # keep only unnan data
    #     corrpatch = np.zeros(unwpatch.shape) * np.nan

    #     print('  {} / {} points removed due to not enough ifg data...'.format(n_pt_all - n_pt_unnan, n_pt_all), flush=True)
    #     # breakpoint()
    #     wrap = 2 * np.pi

    #     # %% Compute number of gaps, ifg_noloop, maxTlen point-by-point
    #     if n_pt_unnan != 0:
    #         # %% Unwrapping corrections in a pixel by pixel basis (to be parallelised)
    #         print('\n Unwrapping Correction inversion for {:.0f} pixels...\n'.format(n_pt_unnan), flush=True)
    #         start2 = time.time()
    #         # for ii in range(n_pt_unnan):
    #         for ix, ii in enumerate(np.random.permutation(n_pt_unnan)):
    #             if (ix + 1) % 1000 == 0:
    #                 print('\t\t{:.0f}/{:.0f} in {:.2f} secs (ETC: {:.0f} secs)\n'.format(ix + 1, n_pt_unnan, time.time() - start2, (time.time() - start2) / (ix + 1) * n_pt_unnan))
    #             if (ix + 1) % 10000 == 0:
    #                 corrFull[ix_unnan_pt] = corrpatch
    #                 correction = corrFull.transpose().reshape(n_ifg, length, width)
    #                 loop_lib.plotmask(correction[96, :, :], centerz=True, title='Correction {} {:.0f}%'.format(ifgdates[96], 100 * ix / n_pt_unnan), interp='Nearest')

    #             disp = unwpatch[ii, :]
    #             # Remove nan-Ifg pixels from the inversion (drop from disp and the corresponding loops)
    #             nonNan = np.where(~np.isnan(disp))[0]
    #             nanDat = np.where(np.isnan(disp))[0]
    #             nonNanLoop = np.where((Aloop[:, nanDat] == 0).all(axis=1))[0]
    #             G = Aloop[nonNanLoop, :][:, nonNan]
    #             closure = (np.dot(G, disp[nonNan]) / wrap).round()
    #             G = matrix(G)
    #             d = matrix(closure)
    #             corrpatch[ii, nonNan] = np.array(l1_lib.l1regls(G, d, alpha=0.01, show_progress=0)).round()[:, 0]

    #         corrFull[ix_unnan_pt] = corrpatch
    #         correction = corrFull.transpose().reshape(n_ifg, length, width)
    #     # %% Finish patch
    #     elapsed_time2 = int(time.time() - start2)
    #     hour2 = int(elapsed_time2 / 3600)
    #     minite2 = int(np.mod((elapsed_time2 / 60), 60))
    #     sec2 = int(np.mod(elapsed_time2, 60))
    #     print("  Elapsed time for {0}th patch: {1:02}h {2:02}m {3:02}s".format(i_patch + 1, hour2, minite2, sec2), flush=True)

    # %% Finish
    elapsed_time = time.time() - start
    hour = int(elapsed_time / 3600)
    minute = int(np.mod((elapsed_time / 60), 60))
    sec = int(np.mod(elapsed_time, 60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour, minute, sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    # print('Output directory: {}\n'.format(os.path.relpath(tsadir)))


# %% Function to read IFGs to array
def read_unw(i):
    print(ifgdates[i])
    unwfile = os.path.join(ifgdir, ifgdates[i], ifgdates[i] + '.unw')
    # Read unw data (radians) at patch area
    unw1 = np.fromfile(unwfile, dtype=np.float32).reshape((length, width))
    unw1[unw1 == 0] = np.nan  # Fill 0 with nan
    ref_unw = []
    buff = 0  # Buffer to increase reference area until a value is found
    while not ref_unw:
        try:
            ref_unw = np.nanmean(unw1[refy1 - buff:refy2 + buff, refx1 - buff:refx2 + buff])
        except RuntimeWarning:
            buff += 1
    unw1 = unw1 - ref_unw

    return unw1


# %% main
if __name__ == "__main__":
    sys.exit(main())
