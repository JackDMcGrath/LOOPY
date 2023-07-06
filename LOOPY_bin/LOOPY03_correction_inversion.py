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
LOOPY03_correction_inversion.py -d ifgdir [-t tsadir] [-c corrdir] [--gamma float] [--n_para int] [--n_unw_r_thre float] [--nanUncorr] [--coast] [--dilation] [--randpix] [--mask]

-d             Path to the GEOCml* dir containing stack of unw data
-t             Path to the output TS_GEOCml* dir
-c             Path to the correction dierectory (Default: GEOCml*L1)
--gamma        Gamma value for L1 Regulariastion (Default: 0.001)
--n_para       Number of parallel processing (Default:  # of usable CPU)
--n_unw_r_thre Threshold of n_unw (number of used unwrap data) (Note this value
               is ratio to the number of images; i.e., 1.5*n_im) Larger number
               (e.g. 2.5) makes processing faster but result sparser.
               (Default: 1 and 0.5 for C- and L-band, respectively)
--coast        Only run correction on the coastlines and don't round inversions - I.E. FORCING ALL LOOPS TO BE CLOSED
--dilation     Number of dilations to be carried out when searching for the coast (Default: 1)
--randpix      Number of pixels to randomly select for inversion
--merge        Invert unmasked pixels

--nanUncorr    Nan anything that can't be inverted due to no loops
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
from skimage import filters
from skimage.morphology import disk
from scipy.ndimage import binary_opening, binary_closing, binary_dilation
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import ConvexHull
from skimage import feature


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
        n_ifg, corrFull, corrdir, nanUncorr, coast, land, nrandpix, n_pix_inv

    # %% Set default
    ifgdir = []
    corrdir = []
    tsadir = []
    reset = True
    nanUncorr = False
    coast = False
    dilation_its = 1
    nrandpix = 0
    merge = False

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
            opts, args = getopt.getopt(argv[1:], "hd:t:c:",
                                       ["help", "noreset", "nanUncorr", "gamma=", "coast",
                                        "dilation=", "randpix=", "n_unw_r_thre=", "n_para=", "merge="])
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
            elif o == '--nanUncorr':
                nanUncorr = True
            elif o == '--coast':
                coast = True;
            elif o == '--dilation':
                dilation_its = int(a)
            elif o == '--randpix':
                nrandpix = int(a)
            elif o == '--merge':
                nrandpix = int(a)

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

    if nanUncorr:
        print('**********\nCAUTION: ANY PIXEL NOT AVAILIABLE TO BE INVERTED WILL BE NANNED\nYOU BETTE BE CONFIDENT COS IM NOT SAVING AN UNNANNED VERSION\n*********')

    tsadir = os.path.abspath(tsadir)
    infodir = os.path.join(tsadir, 'info')
    resultsdir = os.path.join(tsadir, 'results')

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
    if coast:
        n_unw_thre = 1
    else:
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

    if coast:
        print('Dilation Iterations    : {}'.format(dilation_its), flush=True)

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
        if windows_multi:
            unw = np.array(p.map(functools.partial(read_unw_win, ifgdates, length, width, refx1, refx2, refy1, refy2, ifgdir), range(n_ifg)))
        else:
            unw = np.array(p.map(read_unw, range(n_ifg)))
        p.close()

    elapsed_time = time.time() - start
    hour = int(elapsed_time / 3600)
    minute = int(np.mod((elapsed_time / 60), 60))
    sec = int(np.mod(elapsed_time, 60))
    print("\nUNW Loaded: {0:02}h {1:02}m {2:02}s".format(hour, minute, sec))

    n_pt_all = length * width

    unw = unw.reshape((n_ifg, n_pt_all)).transpose()  # (n_pt_all, n_ifg)

    subset = np.arange(n_pt_all)

    if coast:
        dem = io_lib.read_img(os.path.join(ifgdir, 'hgt'), length, width)
        land = np.logical_and(~np.isnan(dem), dem > 0.1).astype(np.float32) # Identify land                                             LAND == 1
        # plot_lib.make_im_png(land, os.path.join(corrdir, 'coastline1.png'), 'viridis', 'Land', cbar=True)
        sigma = 1
        coastline = feature.canny(land, sigma=sigma).astype(np.float32) # Identify coast (or data boundary....) #                       COASTLINE == 1
        # plot_lib.make_im_png(coastline, os.path.join(corrdir, 'coastline2.png'), 'viridis', 'Canny Coast: Sigma {}'.format(sigma), cbar=True)
        its = dilation_its
        if dilation_its > 0:
            coastline = binary_dilation(coastline, structure=np.ones((3,3)), iterations=dilation_its).astype(np.float32) #                           COASTLINE == 1
        # plot_lib.make_im_png(coastline, os.path.join(corrdir, 'coastline3.png'), 'viridis', 'Canny Coast Dilated x {}'.format(dilation_its), cbar=True)
        land[np.where(coastline == 1)] = 2 #                                                                                            LAND == 1, COAST == 2
        # plot_lib.make_im_png(land, os.path.join(corrdir, 'coastline4.png'), 'viridis', 'Land and Coast', cbar=True)
        coastline = np.logical_and(land > 0, coastline == 1).astype(np.float32) # Find the intersection of the land and coast           Coast + Land == 1
        # plot_lib.make_im_png(coastline, os.path.join(corrdir, 'coastline5.png'), 'viridis', 'Land - Coast Intersect', cbar=True)
        coast_pix = np.where(coastline == 1) # All Pixels that are onland (according to the data) and coast (according to canny)
        land[coast_pix] = 3 #                                                                                                           LAND == 1, COASTLINE == 2, BOTH == 3
        print('# Coast Pixels Identified: {}'.format(np.nansum(land.flatten() == 3)))
        plot_lib.make_im_png(land, os.path.join(corrdir, 'coastline.png'), 'viridis', 'Land', cbar=True)
        coast_pix = np.where(land.reshape((1, n_pt_all)) == 3)[1]
        # unw = unw[coast_pix, :] # Extract the coast pixels from the unw dataframe
        subset = coast_pix

    if merge:
        ngap = io_lib.read_img(os.path.join(resultsdir, 'n_gap'), length, width).flatten()
        mergepix = np.where(ngap <= merge)[0].T
        print('# Merge Pixels Identified: {}'.format(np.nansum(ngap.flatten() <= merge)))
        if coast:
            print('Joining Coastal and Merge Selections')
            coast_pix = np.concatenate([coast_pix, mergepix])
        else:
            coast_pix = mergepix

        subset = np.unique(coast_pix) # Prevent time wasting by running the same pixels
        coast = True


    if nrandpix > 0:
        mask = io_lib.read_img(os.path.join(resultsdir, 'n_gap'), length, width) # Read in ngaps as primary mask parameter
        nan_ix = np.isnan(mask) # Find nan pixels (ie sea)
        if merge:
            mask = (mask <= merge).astype(np.float32) # Already going to invert for ngap < merge, so don't filter based on this
        else:
            mask = (mask == 0).astype(np.float32) # Create mask where good pixels are where ngap == 0
        mask[nan_ix] = np.nan # Re-add the sea
        mask_pix = np.where(mask != 0) # Identify masked out pixels we are to sample from
        coh = io_lib.read_img(os.path.join(resultsdir, 'coh_avg'), length, width)
        coh[mask_pix] = np.nan
        plot_lib.make_im_png(coh, os.path.join(corrdir, 'cohmask0.png'), 'viridis', 'Coherence Mask', cbar=True)
        coh_thre = np.nanpercentile(coh, 75)
        coh_thre = np.nanmedian(coh)
        coh = coh >= coh_thre
        plot_lib.make_im_png(coh, os.path.join(corrdir, 'cohmask.png'), 'viridis', 'Coherence Threshold: >= {:.2f}'.format(coh_thre), cbar=True)
        ngap = io_lib.read_img(os.path.join(resultsdir, 'n_gap'), length, width)
        ngap[mask_pix] = np.nan
        plot_lib.make_im_png(ngap, os.path.join(corrdir, 'ngapmask0.png'), 'viridis', 'ngap mask', cbar=True)
        ngap_thre = np.nanpercentile(ngap.flatten(), 25)
        ngap_thre = np.nanmedian(ngap)
        ngap = ngap <= ngap_thre
        plot_lib.make_im_png(ngap, os.path.join(corrdir, 'ngapmask.png'), 'viridis', 'ngap Threshold: <= {}'.format(ngap_thre), cbar=True)
        maxTlen = io_lib.read_img(os.path.join(resultsdir, 'maxTlen'), length, width)
        maxTlen[mask_pix] = np.nan
        plot_lib.make_im_png(maxTlen, os.path.join(corrdir, 'maxTlenmask0.png'), 'viridis', 'MaxTLen mask', cbar=True)
        maxTlen_thre = np.nanpercentile(maxTlen, 50)
        maxTlen_thre = np.nanmedian(maxTlen)
        maxTlen = maxTlen >= maxTlen_thre
        plot_lib.make_im_png(maxTlen, os.path.join(corrdir, 'maxTlenmask.png'), 'viridis', 'MaxTLen Threshold: >= {:.2f}'.format(maxTlen_thre), cbar=True)
        n_unw = io_lib.read_img(os.path.join(resultsdir, 'n_unw'), length, width)
        n_unw[mask_pix] = np.nan
        n_unw = np.nanpercentile(n_unw, 75)
        n_unw = n_unw > n_unw_thre

        print('Search Thresholds')
        print('Parameter\tThreshold')
        print('Coherence:\t{:.3f}'.format(coh_thre))
        print('n_gap:\t\t{}'.format(ngap_thre))
        print('maxTlen:\t{:.1f}'.format(maxTlen_thre))

        cands = np.logical_and(np.logical_and(coh, ngap), maxTlen)

        cands_ix = np.where(cands.flatten())[0]
        ncands = len(cands_ix)

        if ncands == 0:
            print('NO PIXELS AVAILIABLE - GOING TO HAVE TO CHANGE THE SEARCH PARAMETERS')
        else:
            if nrandpix > ncands:
                nnoisepix = nrandpix - ncands
                nrandpix = ncands
                print('Only {} pixels availiable to sample'.format(nrandpix))
                # print('Selecting {} more pixels at random to increase area covered'.format(nnoisepix))

            else:
                print('Randomly selecting {} from {} pixels'.format(nrandpix, ncands))
                rand_ix = np.random.permutation(ncands)[:nrandpix]
                cands_ix = cands_ix[rand_ix]

            if coast:
                print('Joining Coastal and RandPix Selections')
                subset = np.concatenate([coast_pix, cands_ix.T])
                subset = np.unique(subset) # Prevent time wasting by running the same pixels
                land = land.flatten()
                land[cands_ix] = 3
                land = land.reshape((length,width))
            else:
                print('Only using RandPix selection')
                subset = cands_ix

        grid = mask
        grid[~np.isnan(mask)] = 0 # Land Pixels
        grid[np.where(mask == 0)] = 1 # Masked Pixels (Based off ngap)
        grid[cands] = 2 # Candidate Pixels
        grid = grid.flatten()
        grid[cands_ix] = 3  #Selected Pixels
        print(os.path.join(corrdir, 'randPix.png'))
        plot_lib.make_im_png(grid.reshape((length, width)), os.path.join(corrdir, 'randPix.png'), 'tab20c', 'RandPix Selection', cbar=True, vmin=0, vmax=3)

    unw = unw[subset, :] # Extract the selected pixels from the unw dataframe

    # %% For each pixel
    # %% Remove points with less valid data than n_unw_thre
    ix_unnan_pt = np.where(np.sum(~np.isnan(unw), axis=1) > n_unw_thre)[0]
    n_pt_unnan = len(ix_unnan_pt)
    corrFull = np.zeros(unw.shape) * np.nan
    unw = unw[ix_unnan_pt, :]  # keep only data for pixels where n_unw > n_unw_thre
    correction = np.zeros(unw.shape) * np.nan

    print('  {} / {} points removed due to not enough ifg data...'.format(n_pt_all - n_pt_unnan, n_pt_all), flush=True)
    # breakpoint()
    wrap = 2 * np.pi

    # %% Unwrapping corrections in a pixel by pixel basis (to be parallelised)
    print('\n Unwrapping Correction inversion for {0:.0f} pixels in {1} loops...\n'.format(n_pt_unnan, n_loop), flush=True)

    n_para_tmp = n_para
    n_para = 1 # Trust me, I've done the tests. 1 is faster
    n_pix_inv = 0
    if n_para == 1:
        print('with no parallel processing...', flush=True)
        begin = time.time()
        for ii in range(n_pt_unnan):
            if (ii + 1) % 1000 == 0:
                elapse = time.time() - begin
                print('{0}/{1} pixels in {2:.2f} secs (ETC: {3:.0f} secs)'.format(ii + 1, n_pt_unnan, elapse, (elapse / ii) * n_pt_unnan))
            correction[ii, :] = unw_loop_corr(ii)
    else:
        adapt_core = False
        if not adapt_core:
            print('with {} parallel processing...'.format(_n_para), flush=True)
            # Parallel processing
            begin = time.time()
            p = q.Pool(_n_para)
            if windows_multi:
                correction = np.array(p.map(functools.partial(unw_loop_corr_win, n_pt_unnan, Aloop, wrap, unw), range(n_pt_unnan)))
            else:
                correction = np.array(p.map(unw_loop_corr, range(n_pt_unnan)))
            p.close()
        else:
            print('With Adaptive parallel Processing...', flush=True)
            begin = time.time()
            NLoop_pix = np.zeros(unw.shape[0])
            print('Calculating n_loops....')
            for i in range(unw.shape[0]):
                disp = unw[i, :]
                # Remove nan-Ifg pixels from the inversion (drop from disp and the corresponding loops)
                nonNan = np.where(~np.isnan(disp))[0]
                nanDat = np.where(np.isnan(disp))[0]
                nonNanLoop = np.where((Aloop[:, nanDat] == 0).all(axis=1))[0]
                NLoop_pix[i]=Aloop[nonNanLoop, :][:, nonNan].shape[0]

            loop_lims = [0, 10, 250, 500, 1000, 1500]
            para_use = [0, n_para, 40, 20, 10, 5]
            para_use = [0, 40, 40, 40, 40, 40]
            unw_all = unw.copy()
            for i in np.arange(1, len(loop_lims) - 1):
                use_ix = np.where(np.logical_and(NLoop_pix >= loop_lims[i], NLoop_pix < loop_lims[i + 1]))[0]
                if len(use_ix) > 0:
                    _para_adapt = para_use[i] if para_use[i] < len(use_ix) else len(use_ix)
                    _para_adapt = _para_adapt if _para_adapt <= n_para else n_para
                    print('Processing with {} cores for {} pixels with between {} and {} loops'.format(_para_adapt, len(use_ix), loop_lims[i], loop_lims[i + 1]))
                    corr_tmp = np.zeros((len(use_ix), n_ifg)) * np.nan
                    unw = unw_all[use_ix, :]
                    p = q.Pool(_para_adapt)
                    corr_tmp = np.array(p.map(unw_loop_corr, range(len(use_ix))))
                    p.close()
                    correction[use_ix,:] = corr_tmp

                    n_pix_inv += len(use_ix)

            if len(np.where(NLoops_pix <= loop_lims[-1])):
                use_ix = np.where(NLoop_pix >= loop_lims[-1])[0]
                _para_adapt = para_use[i] if para_use[i] < len(use_ix) else len(use_ix)
                _para_adapt = _para_adapt if _para_adapt <= n_para else n_para
                print('Processing with {} cores for {} pixels with between {} and {} loops'.format(_para_adapt, len(use_ix), loop_lims[i], loop_lims[i + 1]))
                corr_tmp = np.zeros((len(use_i), n_ifg)) * np.nan
                unw = unw_all[use_ix, :]
                p = q.Pool(_para_adapt)
                corr_tmp = np.array(p.map(unw_loop_corr, range(len(use_ix))))
                p.close()
                correction[use_ix,:] = corr_tmp
    n_para = n_para_tmp

    elapsed_time = time.time() - start
    hour = int(elapsed_time / 3600)
    minute = int(np.mod((elapsed_time / 60), 60))
    sec = int(np.mod(elapsed_time, 60))
    print("\nCorrections Calculated: {0:02}h {1:02}m {2:02}s".format(hour, minute, sec))

    try:
        corrFull[ix_unnan_pt] = correction
    except ValueError:
        print('No Pixels have been corrected......')

    if coast or nrandpix > 0:
        corrFull_coast = corrFull.copy() # Rename the corrFull variable
        corrFull = np.zeros((n_ifg, n_pt_all)).transpose() # Variable the size of all the data, not just coast
        corrFull[subset, :] = corrFull_coast # Add the corrections for the coast to the correct location
        corrFull[np.where(np.isnan(corrFull))] = 0 # Lets not nan anything
        del corrFull_coast

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
    unw1 = np.fromfile(unwfile, dtype=np.float32).reshape((length, width))
    unw1[unw1 == 0] = np.nan  # Fill 0 with nan
    buff = 0  # Buffer to increase reference area until a value is found
    while np.all(np.isnan(unw1[refy1 - buff:refy2 + buff, refx1 - buff:refx2 + buff])):
        buff += 1
    ref_unw = np.nanmean(unw1[refy1 - buff:refy2 + buff, refx1 - buff:refx2 + buff])
    unw1 = unw1 - ref_unw

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


def unw_loop_corr(i):
    #if (i + 1) % np.floor(n_pt_unnan / 100) == 0:
    #    print('{:.0f} / {:.0f}'.format(i + 1, n_pt_unnan))
    commence=time.time()
    disp = unw[i, :]
    corr = np.zeros(disp.shape)
    if nanUncorr:
        corr = corr * np.nan
    # Remove nan-Ifg pixels from the inversion (drop from disp and the corresponding loops)
    nonNan = np.where(~np.isnan(disp))[0]
    nanDat = np.where(np.isnan(disp))[0]
    nonNanLoop = np.where((Aloop[:, nanDat] == 0).all(axis=1))[0]
    G = Aloop[nonNanLoop, :][:, nonNan]
    NLoop=G.shape[0]
    if NLoop > 10:
        if coast or nrandpix > 0: #Try Always rounding the closure phase, but lets not round the corrections
            closure = (np.dot(G, disp[nonNan]) / wrap).round()
        else:
            closure = (np.dot(G, disp[nonNan]) / wrap).round() # Closure in interger 2pi
        G = matrix(G)
        d = matrix(closure)
        # commence=time.time()
        if coast:
            corr[nonNan] = np.array(loopy_lib.l1regls(G, d, alpha=0.01, show_progress=0))[:, 0]
        else:
            corr[nonNan] = np.array(loopy_lib.l1regls(G, d, alpha=0.01, show_progress=0)).round()[:, 0]
    else:
        commence = time.time()

    cease=time.time()
    if n_pt_unnan >= 100:
        if (i + 1 + n_pix_inv) % np.floor(n_pt_unnan / 20) == 0:
            # print('{:.0f} {:.3f}'.format(NLoop, cease-commence), flush=True)
            print('{:.0f} / {:.0f} ({:.0f}%) NLoops: {:.0f} Inv Time: {:.3f}'.format(i + 1, n_pt_unnan, ((i + 1) / n_pt_unnan) * 100, NLoop, cease-commence), flush=True)
    else:
        print('{:.0f}/{:.0f}({:.0f}%)NLoops: {:.0f} Inv Time: {:.3f}'.format(i + 1, n_pt_unnan, ((i + 1) / n_pt_unnan) * 100, NLoop, cease-commence), flush=True)
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
    corrfiltpng = os.path.join(corrdir, ifgdates[i], ifgdates[i] + '.L1_compareFilt.png')
    corrfile = os.path.join(corrdir, ifgdates[i], ifgdates[i] + '.corr')

    # Convert correction to radians
    unw1 = unw[i, :, :]
    npi = (unw1 / np.pi).round()

    # Options for filtering the corrections
    filtType = 'noFilt'  # noFilt, median, binary
    filtWidth = 5 # Width of kernels
    if filtType == 'noFilt':
        correction = corrFull[i, :, :] * wrap

    elif filtType == 'median':
        kernel = disk(radius=np.ceil(filtWidth / 2))  # Circular kernel, convert filtWidth to radius
        correction = corrFull[i, :, :] + 1 - np.nanmin(corrFull[i, :, :]) # Add offset to ensure all values are > 0 (converting to uint8 makes -1 = 255, np.nan=0)
        correction = filters.rank.median(correction.astype('uint8'), kernel).astype(np.float32) # Run median filter, return to float32
        correction[np.where(correction == 0)] = np.nan # Return 0's to nan
        correction = correction - 1 + np.nanmin(corrFull[i, :, :]) # Remove offset
        correction[np.where(np.isnan(corrFull[i, :, :]))] = np.nan # Make all original NaN's nans
        correction =  correction * wrap # Convert from npi to rads

    elif filtType == 'binary': # Method using binary opening and binary
        grid = np.zeros((length,width))
        corrorig = corrFull[i, :, :] # Original Correction
        corrorig.tofile(corrfile)
        grid[np.where(abs(corrFull[i, :, :]) > 0)] = 1 # Find all areas that have a correction
#        grid = binary_opening(grid, structure=disk(radius=np.ceil(filtWidth / 2))).astype('int') # Remove wild spikes
        grid = binary_closing(grid, structure=disk(radius=np.ceil(filtWidth / 2))).astype('int') # Fill in any holes
        grid = binary_opening(grid, structure=disk(radius=np.ceil(filtWidth / 2))).astype('int') # Remove wild spikes
        correction = corrFull[i, :, :].copy() # Inverted Correction
        correction[np.where(grid == 0)] = 0 # Remove wild spikes from inverted Correction
        grid = grid + (np.abs(correction) > 0).astype('int')  # Make grid where 0  = no correction, 1 = need interpolating, 2 = Inverted Correction
        mask = np.where(grid == 2)
        interp = NearestNDInterpolator(np.transpose(mask), correction[mask]) # Create interpolator
        interp_to = np.where(grid == 1) # Find where to interpolate to
        nearest_data = interp(*interp_to)  # Interpolate
        correction[interp_to] = nearest_data  # Apply corrected data
        correction = correction * wrap # convert from npi to rads
        correction[np.where(np.isnan(corrFull[i, :, :]))] = np.nan

    else:
        raise Usage('Not defined the filter type! (Currently a hard code)')


    corr_unw = unw[i, :, :] - correction
    # print('UNW1 data type: {}'.format(unw1.dtype))
    # print('UNW1 length: {0}, Width: {1}'.format(unw1.shape[0], unw1.shape[1]))
    # print('correction data type: {}'.format(correction.dtype))
    # print('correction length: {0}, Width: {1}'.format(correction.shape[0], correction.shape[1]))
    # print('corr_unw data type: {}'.format(corr_unw.dtype))
    # print('corr_unw length: {0}, Width: {1}'.format(corr_unw.shape[0], corr_unw.shape[1]))
    corr_unw.tofile(unwfile)
    # Create correction png image (UnCorr_unw, npi, correction, Corr_unw)
    titles4 = ['Uncorrected (RMS: {:.2f})'.format(np.sqrt(np.nanmean(unw1.flatten() ** 2))),
               'Corrected (RMS: {:.2f})'.format(np.sqrt(np.nanmean(corr_unw.flatten() ** 2))),
               'Modulo nPi',
               'L1 Correction (nPi)']
    if not coast:
        loopy_lib.make_compare_png(unw1, corr_unw, npi, corrFull[i, :, :], corrcomppng, titles4, 3)
    else:
        corr =  corrFull[i, :, :]
        corr[np.where(land != 3)] = np.nan
        loopy_lib.make_compare_png(unw1, corr_unw, npi, corr, corrcomppng, titles4, 3)

    if filtType != 'noFilt':
        if filtType == 'median':
            titles4 = ['{} Uncorrected'.format(ifgdates[i]), '{} Corrected'.format(ifgdates[i]), 'UnFiltered', 'Med Filtered']
        else:
            titles4 = ['{} Uncorrected'.format(ifgdates[i]),
                      '{} Corrected'.format(ifgdates[i]),
                      'UnFiltered',
                      'Binary Closed']

        loopy_lib.make_filt_png(unw1, corr_unw, corrFull[i, :, :], correction / wrap, corrfiltpng, titles4, 3)

    plot_lib.make_im_png(np.angle(np.exp(1j * unw1 / 3) * 3), unwpngfile, cmap_wrap, ifgdates[i] + '.unw', vmin=-np.pi, vmax=np.pi, cbar=False)

# %% Function to carry out nearest neighbour interpolation
def NN_interp(data):
    mask = np.where(~np.isnan(data))
    interp = NearestNDInterpolator(np.transpose(mask), data[mask])
    interped_data = data.copy()
    interp_to = np.where(((~np.isnan(coh)).astype('int') + np.isnan(data).astype('int')) == 2)
    nearest_data = interp(*interp_to)
    interped_data[interp_to] = nearest_data
    return interped_data

# %% main
if __name__ == "__main__":
    sys.exit(main())
