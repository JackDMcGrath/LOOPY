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
LOOPY03_correction_inversion.py -d ifgdir [-t tsadir] [-c corrdir] [--gamma float] [--n_para int] [--n_unw_r_thre float]

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
import getopt
import os
import sys
import re
import time
import shutil
import numpy as np
import multiprocessing as multi
import SCM
import LOOPY_lib as loopy_lib
import LiCSBAS_io_lib as io_lib
import LiCSBAS_inv_lib as inv_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_loop_lib as loop_lib
import LiCSBAS_plot_lib as plot_lib
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
    global n_para_gap, G, Aloop, imdates, incdir, ifgdir, length, width,\
        coef_r2m, ifgdates, ref_unw, cycle, keep_incfile, resdir, restxtfile, \
        cmap_vel, cmap_wrap, wavelength, refx1, refx2, refy1, refy2, n_pt_unnan, Aloop, wrap, unw, \
        n_ifg, corrFull, corrdir, nullmask, nullify

    # %% Set default
    ifgdir = []
    corrdir = []
    tsadir = []
    reset = True
    nullify = []

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
    cmap_wrap = SCM.romaO

    # %% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hd:t:c:n:",
                                       ["help", "reset", "gamma=",
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
            elif o == '-n':
                nullify = float(a)
            elif o == '--reset':
                reset = True
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
        windows_multi = False
    elif sys.platform == "win32":
        if nullify:
            print('ERROR: Nullify wont work with windows (not written the parallelising finction yet)')
            print('Exiting')
            return
        q = multi.get_context('spawn')
        windows_multi = True
        import functools

    # %% Directory settings
    ifgdir = os.path.abspath(ifgdir)

    if not tsadir:
        tsadir = os.path.join(os.path.dirname(ifgdir), 'TS_' + os.path.basename(ifgdir))

    if not corrdir:
        corrdir = os.path.join(os.path.dirname(ifgdir), os.path.basename(ifgdir) + 'L1')

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
    if nullify:
        nullmask = np.zeros((n_ifg, length, width), dtype=np.float32)

    if n_para == 1:
        print('with no parallel processing...', flush=True)

        for ii in range(n_ifg):
            unw[ii, :, :] = read_unw(ii)
            if nullify:
                nullmask[ii, :, :] = read_mask(ii)

    else:
        print('with {} parallel processing...'.format(_n_para), flush=True)
        # Parallel processing
        p = q.Pool(_n_para)
        if windows_multi:
            unw = np.array(p.map(functools.partial(read_unw_win, ifgdates, length, width, refx1, refx2, refy1, refy2, ifgdir), range(n_ifg)))
        else:
            unw = np.array(p.map(read_unw, range(n_ifg)))
            if nullify:
                nullmask = np.array(p.map(read_mask, range(n_ifg)))
        p.close()

    elapsed_time = time.time() - start
    hour = int(elapsed_time / 3600)
    minute = int(np.mod((elapsed_time / 60), 60))
    sec = int(np.mod(elapsed_time, 60))
    print("\nUNW Loaded: {0:02}h {1:02}m {2:02}s".format(hour, minute, sec))

    n_pt_all = length * width
    unw = unw.reshape((n_ifg, n_pt_all)).transpose()  # (n_pt_all, n_ifg)

    # %% For each pixel
    # %% Remove points with less valid data than n_unw_thre
    ix_unnan_pt = np.where(np.sum(~np.isnan(unw), axis=1) > n_unw_thre)[0]
    n_pt_unnan = len(ix_unnan_pt)
    corrFull = np.zeros(unw.shape) * np.nan
    unw = unw[ix_unnan_pt, :]  # keep only data for pixels where n_unw > n_unw_thre
    nullmask = nullmask[ix_unnan_pt, :]  # keep only data for pixels where n_unw > n_unw_thre
    correction = np.zeros(unw.shape) * np.nan

    print('  {} / {} points removed due to not enough ifg data...'.format(n_pt_all - n_pt_unnan, n_pt_all), flush=True)
    # breakpoint()
    wrap = 2 * np.pi

    # %% Unwrapping corrections in a pixel by pixel basis (to be parallelised)
    print('\n Unwrapping Correction inversion for {0:.0f} pixels in {1} loops...\n'.format(n_pt_unnan, n_loop), flush=True)

    if n_para == 1:
        print('with no parallel processing...', flush=True)
        begin = time.time()
        for ii in range(n_pt_unnan):
            if (ii + 1) % 1000 == 0:
                elapse = time.time() - begin
                print('{0}/{1} pixels in {2:.2f} secs (ETC: {3:.0f} secs)'.format(ii + 1, n_pt_unnan, elapse, (elapse / ii) * n_pt_unnan))
            correction[ii, :] = unw_loop_corr(ii)
    else:
        print('with {} parallel processing...'.format(_n_para), flush=True)
        # Parallel processing
        begin = time.time()
        p = q.Pool(_n_para)
        if windows_multi:
            correction = np.array(p.map(functools.partial(unw_loop_corr_win, n_pt_unnan, Aloop, wrap, unw), range(n_pt_unnan)))
        else:
            correction = np.array(p.map(unw_loop_corr, range(n_pt_unnan)))
        p.close()

    elapsed_time = time.time() - start
    hour = int(elapsed_time / 3600)
    minute = int(np.mod((elapsed_time / 60), 60))
    sec = int(np.mod(elapsed_time, 60))
    print("\nCorrections Calculated: {0:02}h {1:02}m {2:02}s".format(hour, minute, sec))

    corrFull[ix_unnan_pt] = correction
    corrFull = corrFull.transpose().reshape(n_ifg, length, width).astype('float32')

    # %% Apply Correction to all IFGS
    # Reload unw for application of correction

    print("  Reloading {0} ifg's unw and nullmask data...".format(n_ifg), flush=True)

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
    print('{:.0f}/{:.0f} {}'.format(i + 1, len(ifgdates), ifgdates[i]))
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


def read_mask(i):
    maskfile = os.path.join(ifgdir, ifgdates[i], ifgdates[i] + '.nullify.mask')
    # Read ifg nullify mask
    if os.path.exists(maskfile):
        print('{:.0f}/{:.0f} {}'.format(i + 1, len(ifgdates), ifgdates[i]))
        null = np.fromfile(maskfile, dtype=np.int16).reshape((length, width))
        # Reset so masked pixels == 1
        null = (null == 0).astype(np.float32)
    else:
        print('{:.0f}/{:.0f} {} Does not exist. Creating 0 mask'.format(i + 1, len(ifgdates), ifgdates[i]))
        null = np.zeros((length, width), dtype=np.float32)

    return null


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


def unw_loop_corr(i):
    if (i + 1) % np.floor(n_pt_unnan / 100) == 0:
        print('{:.0f} / {:.0f}'.format(i + 1, n_pt_unnan))

    disp = unw[i, :]
    corr = np.zeros(disp.shape)
    # Remove nan-Ifg pixels from the inversion (drop from disp and the corresponding loops)
    nonNan = np.where(~np.isnan(disp))[0]
    nanDat = np.where(np.isnan(disp))[0]
    nonNanLoop = np.where((Aloop[:, nanDat] == 0).all(axis=1))[0]
    G = Aloop[nonNanLoop, :][:, nonNan]
    closure = (np.dot(G, disp[nonNan]) / wrap).round()
    if nullify:
        null = nullmask[i, nonNan]
        G[:, np.where(null)] = G[:, np.where(null)] * nullify
    G = matrix(G)
    d = matrix(closure)
    corr[nonNan] = np.array(loopy_lib.l1regls(G, d, alpha=0.01, show_progress=0)).round()[:, 0]

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
    print('{0}/{1} {2}'.format(i + 1, n_ifg, ifgdates[i]))

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
    titles4 = ['{} Uncorrected'.format(ifgdates[i]),
               '{} Corrected'.format(ifgdates[i]),
               'Modulo nPi',
               'L1 Correction (nPi)']
    loopy_lib.make_compare_png(unw1, corr_unw, npi, corrFull[i, :, :], corrcomppng, titles4, 3)
    plot_lib.make_im_png(np.angle(np.exp(1j * unw1 / 3) * 3), unwpngfile, cmap_wrap, ifgdates[i] + '.unw', vmin=-np.pi, vmax=np.pi, cbar=False)


# %% main
if __name__ == "__main__":
    sys.exit(main())
