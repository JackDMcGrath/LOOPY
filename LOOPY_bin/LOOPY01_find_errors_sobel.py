#!/usr/bin/env python3
"""
========
Overview
========
This script identifies errors in unwrapped interferograms, and creates a mask
that can be applied to these interferograms before they are then used to
correct other IFGs. It is recommended that you run this script on an ml1
dataset, and multilook the resulting masks.

1) Read in IFG, interpolate to full area, and find modulo 2pi values
2) Carry out modal filtering to reduce noise
3) Convert to greyscale, and use sobel filter to find boundaries
4) Compare boundaries for original data, +1 pi and -1pi. Classify intersection
   of all 3 as unwrapping error
5) Add unwrapping errors back into original IFG, and re-interpolate
6) Label regions split up by error lines. Classify any pixel not in the same
   region as the reference pixel as an unwrapping error

Limitations:
    Can't identify regions that are correctly unwrapped, but isolated from main
    pixel, either by being an island, or because it's cut-off by another
    unwrapping error

New modules needed:
- scipy
- skimage
- PIL

===============
Input & output files
===============
Inputs in GEOCml*/:
- yyyymmdd_yyyymmdd/
 - yyyymmdd_yyyymmdd.unw[.png]
 - yyyymmdd_yyyymmdd.cc
- slc.mli.par

Inputs in TS_GEOCml */:
- results/coh_avg  : Average coherence

Outputs in GEOCml */:
- yyyymmdd_yyyymmdd/
  - yyyymmdd_yyyymmdd.mask : Boolean Mask
  - yyyymmdd_yyyymmdd.unw_mask : Masked unwrapped IFG
  - yyyymmdd_yyyymmdd.unw_mask.png : png comparison of unw, npi and mask

Outputs in TS_GEOCml*/:
- info/
 - mask_info.txt : Basic stats of mask coverage

=====
Usage
=====
LOOPY01_find_errors.py -d ifgdir [-t tsadir] [-m int] [--reset] [--n_para]

-d       Path to the GEOCml* dir containing stack of unw data.
-t       Path to the output TS_GEOCml* dir. (Default: TS_GEOCml*)
-m       Output multilooking factor (Default: No multilooking of mask)
-v       IFG to give verbose timings for (Development option, Default: -1 (not verbose))
--reset  Remove previous corrections
--n_para Number of parallel processing (Default: # of usable CPU)

=========
Changelog
=========
v1.1 20220615 Jack McGrath, Uni of Leeds
- Edit to run from command line
v1.0 20220608 Jack McGrath, Uni of Leeds
- Original implementation
"""

import os
import re
import sys
import time
import getopt
import numpy as np
import multiprocessing as multi
import LOOPY_mask_lib as mask_lib
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib
from PIL import Image, ImageFilter
from scipy.ndimage import label
from scipy.interpolate import NearestNDInterpolator
from skimage import filters


insar = tools_lib.get_cmap('SCM.romaO')

class Usage(Exception):
    """Usage context manager"""

    def __init__(self, msg):
        self.msg = msg


# %% Main
def main(argv = None):

    # %% Check argv
    if argv is None:
        argv = sys.argv

    start = time.time()
    ver = "1.1.0"; date = 20220615; author = "J. McGrath"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)

    global plot_figures, tol, ml_factor, refx1, refx2, refy1, refy2, n_ifg, \
        length, width, ifgdir, ifgdates, coh, i, v, begin

    # %% Set default
    ifgdir = []
    tsadir = []
    ml_factor = 1  # Amount to multilook the resulting masks
    reset = False
    plot_figures = False
    v = -1

    # Parallel Processing options
    try:
        n_para = len(os.sched_getaffinity(0))
    except:
        n_para = multi.cpu_count()

    if sys.platform == "linux" or sys.platform == "linux2":
        q = multi.get_context('fork')
    elif sys.platform == "win32":
        q = multi.get_context('spawn')

    # %% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hd:t:m:v:", ["help", "reset", "n_para= "])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == ' --help':
                print(__doc__)
                return 0
            elif o == '-d':
                ifgdir = a
            elif o == '-t':
                tsadir = a
            elif o == '-m':
                ml_factor = int(a)
            elif o == '-v':
                v = int(a) - 1
            elif o == '--reset':
                reset = True
            elif o == '--n_para':
                n_para = int(a)

        if not ifgdir:
            raise Usage('No data directory given, - d is not optional!')
        elif not os.path.isdir(ifgdir):
            raise Usage('No {} dir exists!'.format(ifgdir))
        elif not os.path.exists(os.path.join(ifgdir, 'slc.mli.par')):
            raise Usage('No slc.mli.par file exists in {}!'.format(ifgdir))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  " + str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or -- help.\n", file=sys.stderr)
        return 2

    # %% Directory setting
    ifgdir = os.path.abspath(ifgdir)

    if not tsadir:
        tsadir = os.path.join(os.path.dirname(ifgdir), 'TS_' + os.path.basename(ifgdir))

    if not os.path.exists(tsadir):
        os.mkdir(tsadir)

    netdir = os.path.join(tsadir, 'network')
    if not os.path.exists(netdir):
        os.mkdir(netdir)

    infodir = os.path.join(tsadir, 'info')
    if not os.path.exists(infodir):
        os.mkdir(infodir)

    resultsdir = os.path.join(tsadir, 'results')
    if not os.path.exists(resultsdir):
        os.mkdir(resultsdir)

    #  #TODO: Ensure that the reset flag works following altering of this script
    if reset:
        print('Removing Previous Masks')
        mask_lib.reset_masks(ifgdir)
    else:
        print('Preserving Premade Masks')

    # %% File Setting
    # #TODO: Make sure that the ml10 reffile is adjusted for ml1 data
    ref_file = os.path.join(infodir, '12ref.txt')
    mlipar = os.path.join(ifgdir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))

    # Use coherence to find out how far to interpolate ifg to
    cohfile = os.path.join(resultsdir, 'coh_avg')
    # If no coh file, use slc
    if not os.path.exists(cohfile):
        cohfile = os.path.join(ifgdir, 'slc.mli')
        print('No Coherence File - using SLC instead')

    coh = io_lib.read_img(cohfile, length=length, width=width)
    n_px = sum(sum(~np.isnan(coh[:])))

    # Open file to store mask info
    mask_info_file = os.path.join(infodir, 'mask_info.txt')
    f = open(mask_info_file, 'w')
    print('# Size: {0}({1}x{2}), n_valid: {3}'.format(width * length, width, length, n_px), file=f)
    print('# ifg dates         mask_cov', file=f)
    f.close()

    # %% Prepare variables
    # Get ifg dates
    ifgdates = tools_lib.get_ifgdates(ifgdir)
    n_ifg = len(ifgdates)
    mask_cov = []

    # Find reference pixel. If none provided, use highest coherence pixel
    if os.path.exists(ref_file):
        with open(ref_file, "r") as f:
            refarea = f.read().split()[0]  # str, x1/x2/y1/y2
        refx1, refx2, refy1, refy2 = [int(s) for s in re.split('[:/]', refarea)]

        if np.isnan(coh[refy1:refy2, refx1:refx2]):
            print('Ref point = [{}, {}] invalid. Using max coherent pixel'.format(refy1, refx1))
            refy1, refx1 = np.where(coh == np.nanmax(coh))
            refy1 = refy1[0]
            refy2 = refy1 + 1
            refx1 = refx1[0]
            refx2 = refx1 + 1

    else:
        print('No Reference Pixel provided - using max coherent pixel')

        refy1, refx1 = np.where(coh == np.nanmax(coh))
        refy1 = refy1[0]
        refy2 = refy1 + 1
        refx1 = refx1[0]
        refx2 = refx1 + 1

    print('Ref point = [{}, {}]'.format(refy1, refx1))
    print('Mask Multilooking Factor = {}'.format(ml_factor))

    # %% Run correction in parallel
    _n_para = n_para if n_para < n_ifg else n_ifg
    print('\nRunning error mapping for all {} ifgs, '.format(n_ifg), flush=True)

    if n_para == 1:
        print('with no parallel processing...', flush=True)
        if v >= 0:
            print('In an overly verbose way for IFG {}'.format(v + 1))

        for i in range(n_ifg):
            mask_cov_tmp = mask_unw_errors(i)
            mask_cov.append(mask_cov_tmp)

    else:
        print('with {} parallel processing...'.format(_n_para), flush=True)
        if v >= 0:
            print('In an overly verbose way for IFG {}'.format(v + 1))

        # Parallel processing
        p = q.Pool(_n_para)
        mask_cov = np.array(p.map(mask_unw_errors, range(n_ifg)))
        p.close()

    f = open(mask_info_file, 'a')
    for i in range(n_ifg):
        print('{0}  {1:6.2f}'.format(ifgdates[i], mask_cov[i] / n_px), file=f)
    f.close()

    # %% Finish
    print('\nCheck network/*, 11bad_ifg_ras/* and 11ifg_ras/* in TS dir.')
    print('If you want to change the bad ifgs to be discarded, re-run with different thresholds or make a ifg list and indicate it by --rm_ifg_list option in the next step.')

    elapsed_time = time.time() - start
    hour = int(elapsed_time / 3600)
    minute = int(np.mod((elapsed_time / 60), 60))
    sec = int(np.mod(elapsed_time, 60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour, minute, sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output directory: {}\n'.format(os.path.relpath(ifgdir)))


# %% Function to mask unwrapping errors
def mask_unw_errors(i):
    global begin
    begin = time.time()
    date = ifgdates[i]
    if i == v:
        print('        Starting')
    if os.path.exists(os.path.join(ifgdir, date, date + '.unw_mask')):
        print('    ({}/{}): {}  Mask Exists. Skipping'.format(i + 1, n_ifg, date))
        # #TODO: Rather than set to zero find old mask coverage script and get details from there
        mask_coverage = 0
        return mask_coverage
    else:
        print('    ({}/{}): {}'.format(i + 1, n_ifg, date))

    if i == v:
        print('        Loading')

    # Read in IFG
    unw = io_lib.read_img(os.path.join(ifgdir, date, date + '.unw'), length=length, width=width)
    if i == v:
        print('        UNW Loaded {:.2f}'.format(time.time() - begin))

    # Find Reference Value, and reference all IFGs to same value
    ref = np.nanmean(unw[refy1:refy2, refx1:refx2])
    if np.isnan(ref):
        print('Invalid Ref Value found. Setting to 0')
        ref = 0

    ifg = unw.copy()
    ifg = ifg - ref  # Maybe no need to use a reference - would be better to subtract 0.5 pi or something, incase IFG is already referenced
    if i == v:
        print('        Reffed {:.2f}'.format(time.time() - begin))

    # Interpolate IFG to entire frame
    filled_ifg = NN_interp(ifg)

    if i == v:
        print('        Interpolated {:.2f}'.format(time.time() - begin))

    # Find modulo 2pi values for original IFG, and after adding and subtracting 1pi
    # Use round rather than // to account for slight noise
    npi_og = (filled_ifg / (2 * np.pi)).round()
    npi_p1 = ((filled_ifg + np.pi) / (2 * np.pi)).round()
    npi_m1 = ((filled_ifg - np.pi) / (2 * np.pi)).round()

    if i == v:
        print('        npi_calculated {:.2f}'.format(time.time() - begin))

    # Modal filtering of npi images
    npi_og = mode_filter(npi_og)
    if i == v:
        print('            npi 0 filter  {:.2f}'.format(time.time() - begin))
    npi_p1 = mode_filter(npi_p1)
    if i == v:
        print('            npi p1 filter  {:.2f}'.format(time.time() - begin))
    npi_m1 = mode_filter(npi_m1)
    if i == v:
        print('            npi m1 filter  {:.2f}'.format(time.time() - begin))

    # Create greyscale images for filtering
    graynpi0 = (npi_og - np.nanmin(npi_og) + 1) / (np.nanmax(npi_og) - np.nanmin(npi_og) + 1)
    graynpip1 = (npi_p1 - np.nanmin(npi_p1) + 1) / (np.nanmax(npi_p1) - np.nanmin(npi_p1) + 1)
    graynpim1 = (npi_m1 - np.nanmin(npi_m1) + 1) / (np.nanmax(npi_m1) - np.nanmin(npi_m1) + 1)

    if i == v:
        print('        Greyscale {:.2f}'.format(time.time() - begin))

    # Run Sobel filter for edge detection. Set any edges to 0
    sobeltime = time.time()
    sobel0 = filters.sobel(graynpi0)
    if i == v:
        print('        Sobel0 {:.2f}'.format(time.time() - sobeltime))
        sobeltime = time.time()
    sobel0[sobel0 > 0] = 1
    if i == v:
        print('        Sobel0 binarized {:.2f}'.format(time.time() - sobeltime))
        sobeltime = time.time()
    sobelp1 = filters.sobel(graynpip1)
    if i == v:
        print('        Sobelp1 {:.2f}'.format(time.time() - sobeltime))
        sobeltime = time.time()
    sobelp1[sobelp1 > 0] = 1
    if i == v:
        print('        Sobelp1 binarized {:.2f}'.format(time.time() - sobeltime))
        sobeltime = time.time()
    sobelm1 = filters.sobel(graynpim1)
    if i == v:
        print('        Sobelm1 {:.2f}'.format(time.time() - sobeltime))
        sobeltime = time.time()
    sobelm1[sobelm1 > 0] = 1
    if i == v:
        print('        Sobelm1 binarized {:.2f}'.format(time.time() - sobeltime))
        print('        Sobelled {:.2f}'.format(time.time() - begin))

    # Add up all filters. Class anywhere boundary in all three as an error
    boundary_tot = sobel0 + sobelp1 + sobelm1
    boundary_err = (boundary_tot == 3).astype('int')

    if i == v:
        print('        Boundaries Classified {:.2f}'.format(time.time() - begin))

    # Add error lines to the original IFG, and interpolate with these values to
    # create IFG split up by unwrapping error boundaries
    ifg2 = unw.copy()
    if i == v:
        print('        Copied unw {:.2f}'.format(time.time() - begin))
    err_val = 10 * np.nanmax(ifg2)
    if i == v:
        print('        err_val set {:.2f}'.format(time.time() - begin))
    ifg2[np.where(boundary_err == 1)] = err_val
    if i == v:
        print('        Boundaries added {:.2f}'.format(time.time() - begin))
    filled_ifg2 = NN_interp(ifg2)

    if i == v:
        print('        IFG2 interpolated {:.2f}'.format(time.time() - begin))

    # Binarise the IFG
    filled_ifg2[np.where(filled_ifg2 == err_val)] = np.nan
    filled_ifg2[~np.isnan(filled_ifg2)] = 1
    filled_ifg2[np.isnan(filled_ifg2)] = 0

    if i == v:
        print('        Binarised {:.2f}'.format(time.time() - begin))

    # Label the binary IFG into connected regions
    regions, count = label(filled_ifg2)

    if i == v:
        print('        Added to IFG {:.2f}'.format(time.time() - begin))

    # Find region number of reference pixel. All pixels in this region to be
    # considered unw error free. Mask where 1 == good pixel, 0 == bad
    ref_region = regions[refy1:refy2, refx1:refx2]
    mask = regions == ref_region

    if i == v:
        print('        Mask made {:.2f}'.format(time.time() - begin))

    title3 = ['Original unw', 'Interpolated unw / 2pi', 'Unwrapping Error Mask']

    mask_lib.make_unw_npi_mask_png([unw, npi_og, mask], os.path.join(ifgdir, date, date + '.mask.png'), [insar, 'tab20c', 'viridis'], title3)

    # %% Save Masked UNW to save time in corrections
    mask_coverage = sum(sum(mask == 1)) / sum(sum(~np.isnan(unw)))
    if i == v:
        print('        {}/{} pixels unmasked ({})'.format(sum(sum(mask == 1)), sum(sum(~np.isnan(unw))), mask_coverage))
    masked_ifg = unw.copy().astype('float32')
    masked_ifg[mask == 0] = np.nan

    if i == v:
        print('        IFG masked {:.2f}'.format(time.time() - begin))

    # %% Multilook mask if required
    if ml_factor > 1:
        mask = tools_lib.multilook(mask, ml_factor, ml_factor, 0.1).astype('bool')
        mask = (mask > 0.5)
        masked_ifg = tools_lib.multilook(masked_ifg, ml_factor, ml_factor, 0.1).astype('bool')
        titles = ['UNW', 'ML{} Mask'.format(ml_factor)]
        mask_lib.make_npi_mask_png([unw, mask], os.path.join(ifgdir, date, date + '.ml_mask.png'), [insar, 'viridis'], titles)

    mask.astype('bool').tofile(os.path.join(ifgdir, date, date + '.mask'))
    masked_ifg.tofile(os.path.join(ifgdir, date, date + '.unw_mask'))

    if i == v:
        print('        Saved {:.2f}'.format(time.time() - begin))

    return mask_coverage


# %% Function to carry out nearest neighbour interpolation
def NN_interp(data):
    mask = np.where(~np.isnan(data))
    interp = NearestNDInterpolator(np.transpose(mask), data[mask])
    interped_data = data.copy()
    interp_to = np.where(((~np.isnan(coh)).astype('int') + np.isnan(data).astype('int')) == 2)
    nearest_data = interp(*interp_to)
    interped_data[interp_to] = nearest_data
    return interped_data


# %% Function to modally filter arrays using PIL
def mode_filter(data, filtSize=11):
    npi_min = np.nanmin(data) - 1
    npi_range = np.nanmax(data) - npi_min

    # Convert array into 0-255 range
    greyscale = ((data - npi_min) / npi_range) * 255
    im = Image.fromarray(greyscale.astype('uint8'))

    # Filter image, convert back to np.array, and repopulate with nans
    im_mode = im.filter(ImageFilter.ModeFilter(size=filtSize))
    dataMode = (np.array(im_mode, dtype='float32') / 255) * npi_range + npi_min
    dataMode[np.where(np.isnan(data))] = np.nan

    return dataMode
# %% main
if __name__ == "__main__":
    sys.exit(main())
