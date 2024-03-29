#!/usr/bin/env python3
"""
v1.5.2 20210311 Yu Morishita, GSI

This script inverts the SB network of unw to obtain the loop closure correction
for each interferogram, then applies the correction, using the L1 Regularisation
(Yunjun et al., 2019).
Parallised on a pixel-by-pixel basis, rather than patches, due to size of G-matrix
that would be required by L1reg

Limitations:
    No garauntee that the correction is correct, especially in networks with
    low redundancy
    No effect on no-loop IFGs
    Each pixel is considered independently. No spatial correlation of the errors


Additional modules to LiCSBAS needed:
- cvxopt

===============
Input & output files
===============
Inputs in GEOCml*/:
 -yyyymmdd_yyyymmdd/
   -yyyymmdd_yyyymmdd.unw
 -EQA.dem_par
 -slc.mli.par

Outputs in GEOCml*LoopMask/:
- yyyymmdd_yyyymmdd/
  - yyyymmdd_yyyymmdd.unw[.png] : Corrected unw
  - yyyymmdd_yyyymmdd.cc : Coherence file
  - yyyymmdd_yyyymmdd.L1compare.png : png of original + corrected unw, original npi map, and correction
- other metafiles produced by LiCSBAS02_ml_prep.py

=====
Usage
=====
LOOPY03_iteration_inversion.py -d ifgdir [-t tsadir] [-c corrdir] [--gamma float] [--n_para int] [--n_unw_r_thre float]

-d             Path to the GEOCml* dir containing stack of unw data
-t             Path to the output TS_GEOCml* dir
-c             Path to the correction dierectory (Default: GEOCml*L1)
--gamma        Gamma value for L1 Regulariastion (Default: 0.001)
--n_para       Number of parallel processing (Default:  # of usable CPU)
--n_unw_r_thre Threshold of n_unw (number of used unwrap data) (Note this value
               is ratio to the number of images; i.e., 1.5*n_im) Larger number
               (e.g. 2.5) makes processing faster but result sparser.
               (Default: 1 and 0.5 for C- and L-band, respectively)
"""
# %% Change log
'''
v1.0.0 20230321 Jack McGrath, Uni of Leeds
 - Initial implementation based of LiCSBAS13_invert_small_baselines.py
'''

# %% Import
import os
os.environ["OMP_NUM_THREADS"] = "1"
    # Because np.linalg.lstsq use full CPU but not much faster than 1CPU.
    # Instead parallelize by multiprocessing
import re
import SCM
import sys
import time
import getopt
import shutil
import numpy as np
import multiprocessing as multi
import LOOPY_lib as loopy_lib
import LiCSBAS_io_lib as io_lib
import LiCSBAS_inv_lib as inv_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_loop_lib as loop_lib
import LiCSBAS_plot_lib as plot_lib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from cvxopt import matrix
try:
    from tqdm.contrib.concurrent import process_map
    progress_bar = True
except ModuleNotFoundError:
    progress_bar = False


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
    global G, Aloop, imdates, ifgdir, length, width, ifgdates, cycle, \
        cmap_vel, cmap_wrap, wavelength, refx1, refx2, refy1, refy2, n_pt_unnan, Aloop, wrap, unw, \
        n_ifg, corrFull, corrdir, unw_all, unw_agg, unw_con, begin, n_para, plotdir, pix_plot, \
        pix_output, progress_bar

    # %% Set default
    ifgdir = []
    corrdir = []
    tsadir = []
    reset = True
    pix_plot = False
    pix_output = 1000

    try:
        n_para = len(os.sched_getaffinity(0))
    except:
        n_para = multi.cpu_count()

    gamma = 0.001
    n_unw_r_thre = []

    cmap_vel = SCM.roma.reversed()
    cmap_wrap = SCM.romaO

    # %% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hd:t:c:",
                                       ["help", "noreset", "nanUncorr", "gamma=", "pix_pngs",
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
            elif o == '-c':
                corrdir = a
            elif o == '--noreset':
                reset = False
            elif o == '--gamma':
                gamma = float(a)
            elif o == '--n_unw_r_thre':
                n_unw_r_thre = float(a)
            elif o == '--n_para':
                n_para = int(a)
            elif o == '--pix_pngs':
                pix_plot = False

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
        windows_multi = False
    elif sys.platform == "win32":
        q = multi.get_context('spawn')
        windows_multi = True
        import functools

    # %% Directory settings
    ifgdir = os.path.abspath(ifgdir)

    if not tsadir:
        tsadir = os.path.join(os.path.dirname(ifgdir), 'TS_' + os.path.basename(ifgdir))

    if not corrdir:
        corrdir = os.path.join(os.path.dirname(ifgdir), os.path.basename(ifgdir) + 'L1')
    
    plotdir = os.path.join(corrdir, 'plots')

    if not os.path.isdir(tsadir):
        print('\nNo {} exists!'.format(tsadir), file=sys.stderr)
        return 1

    if not os.path.exists(corrdir):
        os.mkdir(corrdir)

    if reset:
        print('Removing Previous Masks')
        if os.path.exists(corrdir):
            shutil.rmtree(corrdir)
    else:
        print('Preserving Premade Masks')

    if not os.path.exists(corrdir):
        loopy_lib.prepOutdir(corrdir, ifgdir)

    if not os.path.exists(plotdir):
            os.mkdir(plotdir)

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
    imfile = os.path.join(infodir, 'L03used_image.txt')
    with open(imfile, 'w') as f:
        for i in imdates:
            print('{}'.format(i), file=f)

    # Construct G and Aloop matrix for increment and n_gap
    G = inv_lib.make_sb_matrix(ifgdates)
    Aloop = loop_lib.make_loop_matrix(ifgdates)
    n_loop = Aloop.shape[0]

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
    unw_all = np.zeros((n_ifg, length, width), dtype=np.float32)
    unw_agg = np.zeros((n_ifg, length, width), dtype=bool)
    unw_con = np.zeros((n_ifg, length, width), dtype=bool)

    if n_para == 1:
        print('with no parallel processing...', flush=True)

        for ii in range(n_ifg):
            unw_all[ii, :, :] = read_unw(ii)
            unw_agg[ii, :, :] = read_agg(ii)
            unw_con[ii, :, :] = read_con(ii)

    else:
        print('with {} parallel processing...'.format(_n_para), flush=True)
        # Parallel processing
        p = q.Pool(_n_para)
        unw_all = np.array(p.map(read_unw, range(n_ifg)))
        unw_agg = np.array(p.map(read_agg, range(n_ifg)))
        unw_con = np.array(p.map(read_con, range(n_ifg)))
        p.close()

    elapsed_time = time.time() - start
    hour = int(elapsed_time / 3600)
    minute = int(np.mod((elapsed_time / 60), 60))
    sec = int(np.mod(elapsed_time, 60))
    print("\nUNW Loaded: {0:02}h {1:02}m {2:02}s".format(hour, minute, sec))

    n_pt_all = length * width

    unw_all = unw_all.reshape((n_ifg, n_pt_all)).transpose()  # (n_pt_all, n_ifg)
    unw_agg = unw_agg.reshape((n_ifg, n_pt_all)).transpose()  # (n_pt_all, n_ifg)
    unw_con = unw_con.reshape((n_ifg, n_pt_all)).transpose()  # (n_pt_all, n_ifg)

    # %% For each pixel
    # %% Remove points with less valid data than n_unw_thre
    ix_unnan_pt = np.where(np.sum(~np.isnan(unw_all), axis=1) > n_unw_thre)[0]
    n_pt_unnan = len(ix_unnan_pt)
    corrFull = np.zeros(unw_all.shape)
    unw_all = unw_all[ix_unnan_pt, :]  # keep only data for pixels where n_unw > n_unw_thre
    unw_agg = unw_agg[ix_unnan_pt, :]  # keep only data for pixels where n_unw > n_unw_thre
    unw_con = unw_con[ix_unnan_pt, :]  # keep only data for pixels where n_unw > n_unw_thre
    correction = np.zeros(unw_all.shape)

    print('  {} / {} points removed due to not enough ifg data...'.format(n_pt_all - n_pt_unnan, n_pt_all), flush=True)
    # breakpoint()
    wrap = 2 * np.pi

    # %% Unwrapping corrections in a pixel by pixel basis (to be parallelised)
    print('\n Unwrapping Correction inversion for {0:.0f} pixels in {1} loops...\n'.format(n_pt_unnan, n_loop), flush=True)

    begin = time.time()
    if _n_para == 1:
        print('with no parallel processing...', flush=True)
        for ii in range(n_pt_unnan):
            correction[ii, :] = unw_loop_corr(ii)
            if np.mod(ii,10) == 0:
                elapse = time.time() - begin
                print('{0}/{1} pixels in {2:.2f} secs (ETC: {3:.0f} secs)'.format(ii + 1, n_pt_unnan, elapse, (elapse / (ii + 1)) * n_pt_unnan))
    else:
        print('with {} parallel processing...'.format(_n_para), flush=True)
        if progress_bar:
            correction = np.array(process_map(unw_loop_corr, range(n_pt_unnan), max_workers=_n_para, chunksize=1))
        else:
            p = q.Pool(_n_para)
            correction = np.array(p.map(unw_loop_corr, range(n_pt_unnan)))
            p.close()

    elapsed_time = time.time() - start
    hour = int(elapsed_time / 3600)
    minute = int(np.mod((elapsed_time / 60), 60))
    sec = int(np.mod(elapsed_time, 60))
    print("\nCorrections Calculated: {0:02}h {1:02}m {2:02}s".format(hour, minute, sec))

    try:
        corrFull[ix_unnan_pt] = correction
    except ValueError:
        print('No Pixels have been corrected......')

    corrFull = corrFull.transpose().reshape(n_ifg, length, width).astype('float32')

    # %% Apply Correction to all IFGS
    # Reload unw for application of correction

    print("  Reloading {0} ifg's unw data...".format(n_ifg), flush=True)

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
        if windows_multi:
            unw = np.array(p.map(functools.partial(read_unw_win, ifgdates, length, width, refx1, refx2, refy1, refy2, ifgdir), range(n_ifg)))
        else:
            unw = np.array(p.map(read_unw, range(n_ifg)))
        p.close()

    elapsed_time = time.time() - start
    hour = int(elapsed_time / 3600)
    minute = int(np.mod((elapsed_time / 60), 60))
    sec = int(np.mod(elapsed_time, 60))
    print("\nUNW re-loaded: {0:02}h {1:02}m {2:02}s".format(hour, minute, sec))

    print("  Applying Corrections and making pngs...", flush=True)
    if not windows_multi and n_para != 1:
        print('with {} parallel processing...'.format(_n_para), flush=True)
        # Parallel processing
        p = q.Pool(_n_para)
        p.map(apply_correction, range(n_ifg))
        p.close()
    else:
        for ii in range(n_ifg):
            apply_correction(ii)

    print("\nCorrection Applied")
    # %% Finish
    elapsed_time = time.time() - start
    hour = int(elapsed_time / 3600)
    minute = int(np.mod((elapsed_time / 60), 60))
    sec = int(np.mod(elapsed_time, 60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour, minute, sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output directory: {}\n'.format(corrdir))
    print('Check Corrected images in {}'.format(os.path.basename(corrdir)))
    print('If you want to add some of these IFGs to be discared from time series, easiest way is to run LiCSBAS11_check_unw.py, and append IFG list to 11bad_ifg.txt')


# %% Function to read IFGs to array
def read_unw(i):
    if ((i + 1) / 10) == 0:
        print('{:.0f}/{:.0f}'.format(i + 1, len(ifgdates)))
    unwfile = os.path.join(ifgdir, ifgdates[i], ifgdates[i] + '.unw')
    # Read unw data (radians) at patch area
    try:
        unw1 = np.fromfile(unwfile, dtype=np.float32).reshape((length, width))
    except:
        unwfile = os.path.join(ifgdir, ifgdates[i], ifgdates[i] + '_orig.unw')
        unw1 = np.fromfile(unwfile, dtype=np.float32).reshape((length, width))
    unw1[unw1 == 0] = np.nan  # Fill 0 with nan


    return unw1

def read_agg(i):
    if ((i + 1) / 10) == 0:
        print('{:.0f}/{:.0f}'.format(i + 1, len(ifgdates)))
    unwfile = os.path.join(ifgdir, ifgdates[i], ifgdates[i] + '_agg.unw')
    try:
        # Read unw data (radians) at patch area
        unw1 = np.fromfile(unwfile, dtype=np.float32).reshape((length, width))
        unw1[unw1 == 0] = np.nan  # Fill 0 with nan
    except:
        unw1 = np.zeros((length, width)) * np.nan

    unw1 = ~np.isnan(unw1)
    return unw1

def read_con(i):
    if ((i + 1) / 10) == 0:
        print('{:.0f}/{:.0f}'.format(i + 1, len(ifgdates)))
    unwfile = os.path.join(ifgdir, ifgdates[i], ifgdates[i] + '_con.unw')
    try:
        # Read unw data (radians) at patch area
        unw1 = np.fromfile(unwfile, dtype=np.float32).reshape((length, width))
        unw1[unw1 == 0] = np.nan  # Fill 0 with nan
    except:
        unw1 = np.zeros((length, width)) * np.nan

    unw1 = ~np.isnan(unw1)
    return unw1


def read_unw_win(ifgdates, length, width, refx1, refx2, refy1, refy2, ifgdir, i):
    print('{:.0f}/{:.0f} {}'.format(i, len(ifgdates), ifgdates[i]))
    unwfile = os.path.join(ifgdir, ifgdates[i], ifgdates[i] + '.unw')
    # Read unw data (radians) at patch area
    unw1 = np.fromfile(unwfile, dtype=np.float32).reshape((length, width))
    unw1[unw1 == 0] = np.nan  # Fill 0 with nan
    buff = 0  # Buffer to increase reference area until a value is found
    while np.all(np.isnan(unw1[refy1 - buff:refy2 + buff, refx1 - buff:refx2 + buff])):
        buff += 1
    ref_unw = np.nanmean(unw1[refy1 - buff:refy2 + buff, refx1 - buff:refx2 + buff])
    unw1 = unw1 - ref_unw

    return unw1


def unw_loop_corr(ii):

    commence = time.time()
    disp_all = unw_all[ii, :]
    corr = np.zeros(disp_all.shape)
    
    ifg_good = np.array(ifgdates)[np.where(unw_agg[ii,:])[0]]
    ifg_cand = np.array(ifgdates)[np.where(unw_con[ii,:])[0]]
    ifg_bad = np.array(ifgdates)[~np.isnan(unw_all[ii,:])]
    ifg_good = list(set(ifg_good))
    ifg_cand = list(set(ifg_cand) - set(ifg_good))
    ifg_bad = list(set(ifg_bad) - set(ifg_cand) - set(ifg_good))

    good_ix = np.array([ix for ix, date in enumerate(ifgdates) if date in ifg_good])
    cand_ix = np.array([ix for ix, date in enumerate(ifgdates) if date in ifg_cand])
    bad_ix = np.array([ix for ix, date in enumerate(ifgdates) if date in ifg_bad])

    n_good = len(ifg_good)
    n_cand = len(ifg_cand)
    n_bad = len(ifg_bad)

    solve_order = np.concatenate((good_ix, cand_ix[np.random.permutation(n_cand)], bad_ix[np.random.permutation(n_bad)])).astype('int')

    # Change loop matrix to reflect solve order
    solveLoop = Aloop[:, solve_order]
    complete_loops = np.where(np.sum((solveLoop != 0), axis=1) == 3)[0]
    solveLoop = solveLoop[complete_loops, :]
    disp_all = disp_all[solve_order]
    nLoops, ifg_tot = solveLoop.shape

    if n_good < (ifg_tot / 4):
        if (n_good + n_cand) < (ifg_tot / 3):
            # If theres not enough that survived any nulling, don't try inverting
            # (Increased threshold to reflect potentially lower quality data)
            return corr
        else:
            n_good = int(ifg_tot / 5)

    n_it = 0

    while n_good < ifg_tot:
        n_it += 1
        n_invert = int(n_good * 1.25) if int(n_good * 1.25) < ifg_tot else ifg_tot

        if np.mod(ii, pix_output) == 0 and n_it == 1 and pix_plot:
            closure_orig = (np.dot(solveLoop, disp_all) / wrap).round() # Closure in integer 2pi

        disp = disp_all[:n_invert]

        G_all = solveLoop[:, :n_invert] # Select only the IFGs to be inverted

        # Now remove any incomplete loops from the matrix
        complete_loops = np.where(np.sum((G_all != 0), axis=1) == 3)[0]
        G = G_all[complete_loops, :]

        # Some interferograms will now have no loops. Remove these
        loopIfg = np.where((G != 0).any(axis=0))[0]
        G = G[:, loopIfg]
        if G.shape[0] > 10:
            invIfg_ix = np.arange(n_invert)
            invIfg_ix = invIfg_ix[loopIfg]
            disp = disp[loopIfg]
            closure = (np.dot(G, disp) / wrap).round() # Closure in integer 2pi
            if (closure != 0).any():
                G = matrix(G)
                d = matrix(closure)
                correction = np.array(loopy_lib.l1regls(G, d, alpha=0.01, show_progress=0)).round()[:, 0]
                disp_all[invIfg_ix] -= correction * wrap
                corr[solve_order[invIfg_ix]] += correction
        
        n_good = n_invert

    if pix_plot and np.mod(ii, pix_output) == 0:
        try:
            closure_final = (np.dot(solveLoop, disp_all) / wrap).round() # Closure in integer 2pi
            # grdx = int(max(closure_orig) - min(closure_orig)) * 1 
            # grdy = int(max(closure_final) - min(closure_final)) * 1
            # grdx = grdx if grdx != 0 else 1
            # grdy = grdy if grdy != 0 else 1
            # plt.hexbin(closure_orig, closure_final, gridsize=(grdx, grdy), mincnt=1, cmap='inferno', norm=colors.LogNorm(vmin=1))
            # plt.colorbar()
            # plt.xlabel('Input')
            # plt.ylabel('Corrected')
            # plt.savefig(os.path.join(plotdir, '{}_all.png'.format(ii)))
            # plt.close()
            # print('Plotted {}'.format(os.path.join(plotdir, '{}_all.png'.format(ii))))
            improve = 100 * sum(abs(closure_final) < abs(closure_orig)) / nLoops
            unchange = 100 * sum(closure_final == closure_orig) / nLoops
            worsen = 100 * sum(abs(closure_final) > abs(closure_orig)) / nLoops
            grdx = int(max(abs(closure_orig)) - min(abs(closure_orig))) * 1 
            grdy = int(max(abs(closure_final)) - min(abs(closure_final))) * 1
            grdx = grdx if grdx != 0 else 1
            grdy = grdy if grdy != 0 else 1
            plt.hexbin(abs(closure_orig), abs(closure_final), gridsize=(grdx, grdy), mincnt=1, cmap='inferno', norm=colors.LogNorm(vmin=1))
            plt.plot([0,max([max(abs(closure_orig)), max(abs(closure_final))])],[0,max([max(abs(closure_orig)), max(abs(closure_final))])]) 
            plt.colorbar()
            plt.title('{} ifgs in {} loops\nImproved: {:.0f}% Same: {:.0f}% Worse: {:.0f}%'.format(ifg_tot, nLoops, improve, unchange, worsen))
            plt.xlabel('Input')
            plt.ylabel('Corrected')
            plt.savefig(os.path.join(plotdir, '{}_all2.png'.format(ii)))
            plt.close()
            if not progress_bar:
                print('Plotted {}'.format(os.path.join(plotdir, '{}_all2.png'.format(ii))))      
        except:
            print('Error in plotting {}_all'.format(ii))

    if not progress_bar:
      if np.mod(ii, pix_output) == 0 or n_para == 1:
        print('{}/{} {} iterations for {} ifgs in {} loops in {:.2f} seconds (Total Time: {:.2f} seconds)'.format(ii, n_pt_unnan,n_it, ifg_tot, nLoops, time.time() - commence, time.time() - begin))

    return corr


def unw_loop_corr_win(n_pt_unnan, Aloop, wrap, unw, i):
    if (i + 1) % np.floor(n_pt_unnan / 100) == 0:
        print('{:.0f}\t/ {:.0f}'.format(i + 1, n_pt_unnan))

    disp = unw[i, :]
    corr = np.zeros(disp.shape)
    # Remove nan-Ifg pixels from the inversion (drop from disp and the corresponding loops)
    nonNan = np.where(~np.isnan(disp))[0]
    nanDat = np.where(np.isnan(disp))[0]
    nonNanLoop = np.where((Aloop[:, nanDat] == 0).all(axis=1))[0]
    G = Aloop[nonNanLoop, :][:, nonNan]
    closure = (np.dot(G, disp[nonNan]) / wrap).round()
    G = matrix(G)
    d = matrix(closure)
    corr[nonNan] = np.array(loopy_lib.l1regls(G, d, alpha=0.01, show_progress=0)).round()[:, 0]

    return corr


def apply_correction(i):
    """
    Apply Correction to UNW ifg, and save outputs, preserving the original data
    """
    if ((i + 1) / 10) == 0:
        print('{0}/{1}'.format(i + 1, n_ifg))

    if not os.path.exists(os.path.join(corrdir, ifgdates[i])):
        os.mkdir((os.path.join(corrdir, ifgdates[i])))

    if not os.path.exists(os.path.join(corrdir, ifgdates[i], ifgdates[i] + '.cc')):
        shutil.copy(os.path.join(ifgdir, ifgdates[i], ifgdates[i] + '.cc'), os.path.join(corrdir, ifgdates[i], ifgdates[i] + '.cc'))

    unwfile = os.path.join(corrdir, ifgdates[i], ifgdates[i] + '.unw')
    unwpngfile = os.path.join(corrdir, ifgdates[i], ifgdates[i] + '.unw.png')
    corrcomppng = os.path.join(corrdir, ifgdates[i], ifgdates[i] + '.L1_compare.png')

    # Convert correction to radians
    unw1 = unw[i, :, :]
    npi = (unw1 / np.pi).round()

    correction = corrFull[i, :, :] * wrap

    corr_unw = unw[i, :, :] - correction
    corr_unw.tofile(unwfile)
    # Create correction png image (UnCorr_unw, npi, correction, Corr_unw)
    titles4 = ['Uncorrected (RMS: {:.2f})'.format(np.sqrt(np.nanmean(unw1.flatten() ** 2))),
               'Corrected (RMS: {:.2f})'.format(np.sqrt(np.nanmean(corr_unw.flatten() ** 2))),
               'Modulo nPi',
               'L1 Correction (nPi)']

    loopy_lib.make_compare_png(unw1, corr_unw, npi, corrFull[i, :, :], corrcomppng, titles4, 3)

    plot_lib.make_im_png(np.angle(np.exp(1j * unw1 / 3) * 3), unwpngfile, cmap_wrap, ifgdates[i] + '.unw', vmin=-np.pi, vmax=np.pi, cbar=False)

# %% main
if __name__ == "__main__":
    sys.exit(main())
