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
3) Find difference between adjacent pixels
4) Classify errors as any pixel that borders a pixel with > 1pi difference
5) Add unwrapping errors back into original IFG, and re-interpolate
6) Label regions split up by error lines. Classify any pixel not in the same
   region as the reference pixel as an unwrapping error
7) Look to correct any error region bordering the good region

Limitations:
    Can't identify regions that are correctly unwrapped, but isolated from main
    pixel, either by being an island, or because it's cut-off by another
    unwrapping error (Latter option would require iterations)
    Needs an unwrapping error to be complete enclosed in order to be masked
    ('Twist' errors can't be IDd')

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
LOOPY01_find_errors.py -d ifgdir [-t tsadir] [-m int] [--full_res] [--reset] [--n_para]

-d        Path to the GEOCml* dir containing stack of unw data
-t        Path to the output TS_GEOCml* dir. (Default: TS_GEOCml*)
-m        Output multilooking factor (Default: No multilooking of mask)
-v        IFG to give verbose timings for (Development option, Default: -1 (not verbose))
--fullres Create masks from full res data, and multilook to -m (ie. orginal geotiffs) (Assume in folder called GEOC)
--reset   Remove previous corrections
--n_para  Number of parallel processing (Default: # of usable CPU)

=========
Changelog
=========
v1.3 20230201 Jack McGrath, Uni of Leeds
- Allow option to mask geotiffs directly
v1.2 20230131 Jack McGrath, Uni of Leeds
- Change masking method to edge detection
v1.1 20220615 Jack McGrath, Uni of Leeds
- Edit to run from command line
v1.0 20220608 Jack McGrath, Uni of Leeds
- Original implementation
"""

import os
import re
import sys
import SCM
import time
import glob
import getopt
import shutil
import numpy as np
import multiprocessing as multi
import LOOPY_mask_lib as mask_lib
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib
from osgeo import gdal
from PIL import Image, ImageFilter
from skimage.filters.rank import modal
from scipy.ndimage import label
from scipy.ndimage import binary_dilation
from scipy.interpolate import NearestNDInterpolator
from scipy.stats import mode
from skimage import filters


insar = tools_lib.get_cmap('SCM.romaO')


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
    ver = "1.3.0"; date = 20230201; author = "J. McGrath"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)

    global plot_figures, tol, ml_factor, refx1, refx2, refy1, refy2, n_ifg, \
        length, width, ifgdir, ifgdates, coh, i, v, begin, fullres, geocdir

    # %% Set default
    ifgdir = []
    tsadir = []
    ml_factor = []  # Amount to multilook the resulting masks
    fullres = False
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
            opts, args = getopt.getopt(argv[1:], "hd:t:m:v:", ["help", "reset", "n_para=", "fullres"])
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
            elif o == '--fullres':
                fullres = True

        if not ifgdir:
            raise Usage('No data directory given, -d is not optional!')
        elif not os.path.isdir(ifgdir):
            raise Usage('No {} dir exists!'.format(ifgdir))
        elif not os.path.exists(os.path.join(ifgdir, 'slc.mli.par')):
            raise Usage('No slc.mli.par file exists in {}!'.format(ifgdir))

        if fullres:
            if not ml_factor:
                raise Usage('No multilooking factor given, -m is not optional when using --fullres!')
        else:
            ml_factor = 1

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  " + str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
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

    # Find how far to interpolate IFG to
    if fullres:
        geocdir = os.path.abspath(os.path.join(ifgdir, '..', 'GEOC'))
        print('Processing full resolution masks direct from tifs in {}'.format(geocdir))

        # Create full res mli
        print('\nCreate slc.mli', flush=True)
        mlitif = glob.glob(os.path.join(geocdir, '*.geo.mli.tif'))
        if len(mlitif) > 0:
            mlitif = mlitif[0]  # First one
            coh = np.float32(gdal.Open(mlitif).ReadAsArray())  # Coh due to previous use of coherence to find IFG limits
            coh[coh == 0] = np.nan

            mlifile = os.path.join(geocdir, 'slc.mli')
            coh.tofile(mlifile)
            mlipngfile = mlifile + '.png'
            mli = np.log10(coh)
            vmin = np.nanpercentile(mli, 5)
            vmax = np.nanpercentile(mli, 95)
            plot_lib.make_im_png(mli, mlipngfile, 'gray', 'MLI (log10)', vmin, vmax, cbar=True)
            print('  slc.mli[.png] created', flush=True)
            ref_type = 'MLI'
        else:
            print('  No *.geo.mli.tif found in {}'.format(os.path.basename(geocdir)), flush=True)

    else:
        cohfile = os.path.join(resultsdir, 'coh_avg')
        ref_type = 'coherence'
        # If no coh file, use slc
        if not os.path.exists(cohfile):
            cohfile = os.path.join(ifgdir, 'slc.mli')
            print('No Coherence File - using MLI instead')
            ref_type = 'MLI'

        coh = io_lib.read_img(cohfile, length=length, width=width)

    # Find reference pixel. If none provided, use highest coherence pixel
    if os.path.exists(ref_file):
        with open(ref_file, "r") as f:
            refarea = f.read().split()[0]  # str, x1/x2/y1/y2
        refx1, refx2, refy1, refy2 = [int(s) for s in re.split('[:/]', refarea)]

        # Change reference pixel in case working with fullres data
        if fullres:
            refx1 = refx1 * ml_factor
            refx2 = refx2 * ml_factor
            refy1 = refy1 * ml_factor
            refy2 = refy2 * ml_factor

        if np.isnan(np.nanmean(coh[refy1:refy2, refx1:refx2])):
            print('Ref point = [{}, {}] invalid. Using max {} pixel'.format(refy1, refx1, ref_type))
            refy1, refx1 = np.where(coh == np.nanmax(coh))
            refy1 = refy1[0]
            refy2 = refy1 + 1
            refx1 = refx1[0]
            refx2 = refx1 + 1

    else:
        print('No Reference Pixel provided - using max {} pixel'.format(ref_type))

        refy1, refx1 = np.where(coh == np.nanmax(coh))
        refy1 = refy1[0]
        refy2 = refy1 + 1
        refx1 = refx1[0]
        refx2 = refx1 + 1

        # Change reference pixel in case working with fullres data
        if fullres:
            refx1 = refx1 * ml_factor
            refx2 = refx2 * ml_factor
            refy1 = refy1 * ml_factor
            refy2 = refy2 * ml_factor

    print('Ref point = [{}, {}]'.format(refy1, refx1))
    print('Mask Multilooking Factor = {}'.format(ml_factor))

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
    if fullres:
        unw = gdal.Open(os.path.join(geocdir, date, date + '.geo.unw.tif')).ReadAsArray()
        unw[unw == 0] = np.nan
    else:
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
    boundary_err = binary_dilation(boundary_err, iterations=5)

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
    regions = regions.astype('float32')
    regionId, regionSize = np.unique(regions, return_counts=True)

    if i == v:
        print('        Added to IFG {:.2f}'.format(time.time() - begin))

    # Set a minimum size of region to be corrected
    min_corr_size = ml_factor * ml_factor * 10  # 10 pixels at final ml size

    # Region IDs to small to corrected
    drop_regions = regionId[np.where(regionSize < min_corr_size)]
    regions[np.where(np.isin(regions, np.append(drop_regions, 0)))] = np.nan

    # Reinterpolate without tiny regions
    regions = NN_interp(regions)

    # Find region number of reference pixel. All pixels in this region to be
    # considered unw error free. Mask where 1 == good pixel, 0 == bad
    # Use mode incase ref area is > 1 pixel (eg if working at full res)
    ref_region = mode(regions[refy1:refy2, refx1:refx2].flatten(), keepdims=True)[0][0]
    mask = regions == ref_region

    if i == v:
        print('        Mask made {:.2f}'.format(time.time() - begin))

    # Make an array exclusively holding the good values
    good_vals = np.zeros(mask.shape) * np.nan
    good_vals[mask] = npi_og[mask]

    # Make an array to hold the correction
    correction = np.zeros(mask.shape)

    # Boolean array of the outside boundary of the good mask
    good_border = filters.sobel(mask).astype('bool')
    corr_regions = np.unique(regions[good_border])
    corr_regions = np.delete(corr_regions, np.array([np.where(corr_regions == ref_region)[0][0], np.where(np.isnan(corr_regions))[0][0]])).astype('int')

    for corrIx in corr_regions:
        # Make map only of the border regions
        border_regions = np.zeros(mask.shape)
        border_regions[good_border] = regions[good_border]

        # Plot boundary in isolation
        border = np.zeros(mask.shape).astype('int')
        border[np.where(border_regions == corrIx)] = 1
        # Dilate boundary so it crosses into both regions
        border_dil = binary_dilation(border).astype('int')
        av_err = mode(npi_og[np.where(border == 1)], nan_policy='omit', keepdims=False)[0]
        av_good = mode(good_vals[np.where(border_dil == 1)], nan_policy='omit', keepdims=False)[0]

        correction[np.where(regions == corrIx)] = (av_good - av_err) * 2 * np.pi

    # Apply correction to original version of IFG
    corr_unw = unw.copy()
    corr_unw[np.where(~np.isnan(corr_unw))] = corr_unw[np.where(~np.isnan(corr_unw))] + correction[np.where(~np.isnan(corr_unw))]

    # %% Make PNGs
    title3 = ['Original unw', 'Interpolated unw / pi', 'Unwrapping Error Mask']
    mask_lib.make_unw_npi_mask_png([unw, (filled_ifg / (np.pi)).round(), mask], os.path.join(ifgdir, date, date + '.mask.png'), [insar, 'tab20c', 'viridis'], title3)

    title3 = ['Original unw', 'Correction', 'Corrected IFG']
    mask_lib.make_unw_mask_corr_png([unw, correction, corr_unw], os.path.join(ifgdir, date, date + '.corr.png'), [insar, 'tab20c', insar], title3)

    # %% Save Masked UNW to save time in corrections
    # If working with full res data, load in the ml IFG to be masked
    if fullres:
        masked_ifg = io_lib.read_img(os.path.join(ifgdir, date, date + '.unw'), length, width)
        n_px = sum(sum(~np.isnan(masked_ifg)))
        if i == v:
            print('        Loaded ML{} IFG {:.2f}'.format(ml_factor, time.time() - begin))

        mask = tools_lib.multilook(mask, ml_factor, ml_factor, 0.5)
        corr_unw = tools_lib.multilook(corr_unw, ml_factor, ml_factor, 0.1)
        if i == v:
            print('        Mask multilooked {:.2f}'.format(time.time() - begin))
        mask = (mask > 0.5)
        if i == v:
            print('        Mask re-binarised {:.2f}'.format(time.time() - begin))

        masked_ifg[np.where(mask == 0)] = np.nan
        if i == v:
            print('        IFG masked {:.2f}'.format(time.time() - begin))

        titles = ['UNW', 'ML{} Mask'.format(ml_factor)]
        mask_lib.make_npi_mask_png([unw, mask], os.path.join(ifgdir, date, date + '.ml_mask.png'), [insar, 'viridis'], titles)
        if i == v:
            print('        Multilooked png made {:.2f}'.format(time.time() - begin))

    else:
        masked_ifg = unw.copy().astype('float32')
        n_px = sum(sum(~np.isnan(masked_ifg)))
        masked_ifg[mask == 0] = np.nan
        if i == v:
            print('        IFG masked {:.2f}'.format(time.time() - begin))

    unmasked_percent = sum(sum(~np.isnan(masked_ifg))) / n_px
    mask_coverage = sum(sum(mask == 1))  # Number of pixels that are unmasked
    if i == v:
        print('        {}/{} pixels unmasked ({}) {:.2f}'.format(sum(sum(~np.isnan(masked_ifg))), n_px, unmasked_percent, time.time() - begin))

    # Backup original unw file and loop png
    shutil.move(os.path.join(ifgdir, date, date + '.unw'), os.path.join(ifgdir, date, date + '_uncorr.unw'))
    shutil.move(os.path.join(ifgdir, date, date + '.unw.png'), os.path.join(ifgdir, date, date + '_uncorr.unw.png'))
    title = '{} ({}pi/cycle)'.format(date, 3 * 2)
    plot_lib.make_im_png(np.angle(np.exp(1j * corr_unw / 3) * 3), os.path.join(ifgdir, date, date + '.unw.png'), SCM.romaO, title, -np.pi, np.pi, cbar=False)

    # Make new unw file from corrected data and new loop png
    corr_unw.tofile(os.path.join(ifgdir, date, date + '.unw'))

    mask.tofile(os.path.join(ifgdir, date, date + '.mask'))
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

# %% Function to modally filter arrays using Scikit
def mode_filter(data, filtSize=21):
    npi_min = np.nanmin(data) - 1
    npi_range = np.nanmax(data) - npi_min

    # Convert array into 0-255 range
    greyscale = ((data - npi_min) / npi_range) * 255

    # Filter image, convert back to np.array, and repopulate with nans
    im_mode = modal(greyscale.astype('uint8'), np.ones([filtSize, filtSize]))
    dataMode = ((np.array(im_mode, dtype='float32') / 255) * npi_range + npi_min).round()
    dataMode[np.where(np.isnan(data))] = np.nan
    dataMode[np.where(dataMode == npi_min)] = np.nan

    return dataMode

# %% Function to modally filter arrays using PIL (40% slower than scikit)
def mode_filterPIL(data, filtSize=11):
    npi_min = np.nanmin(data) - 1
    npi_range = np.nanmax(data) - npi_min

    # Convert array into 0-255 range
    greyscale = ((data - npi_min) / npi_range) * 255
    im = Image.fromarray(greyscale.astype('uint8'))

    # Filter image, convert back to np.array, and repopulate with nans
    im_mode = im.filter(ImageFilter.ModeFilter(size=filtSize))
    dataMode = ((np.array(im_mode, dtype='float32') / 255) * npi_range + npi_min).round()
    dataMode[np.where(np.isnan(data))] = np.nan
    dataMode[np.where(dataMode == npi_min)] = np.nan

    return dataMode


# %% main
if __name__ == "__main__":
    sys.exit(main())
