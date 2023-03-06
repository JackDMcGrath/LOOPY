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
    Needs an unwrapping error to be complete enclosed in order to be masked

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
LOOPY01_find_errors.py -d ifgdir [-t tsadir] [-m int] [-e int] [-v int] [--full_res] [--reset] [--n_para]

-d        Path to the GEOCml* dir containing stack of unw data
-t        Path to the output TS_GEOCml* dir. (Default: TS_GEOCml*)
-m        Output multilooking factor (Default: No multilooking of mask)
-e        Number of iterations of binary dilation of the error boundaries. Set to zero for no dilation (Default: 0.5 * ml_factor)
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
import time
import glob
import getopt
import numpy as np
import multiprocessing as multi
import LOOPY_mask_lib as mask_lib
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib
from osgeo import gdal
from PIL import Image, ImageFilter
from scipy.ndimage import label
from scipy.interpolate import NearestNDInterpolator
from skimage import filters
from skimage.segmentation import flood, flood_fill
from scipy.ndimage import binary_dilation

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
    ver = "1.3.0"
    date = 20230201
    author = "J. McGrath"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)

    global plot_figures, tol, ml_factor, refx1, refx2, refy1, refy2, n_ifg, \
        length, width, ifgdir, ifgdates, coh, coast, i, v, begin, fullres, geocdir, err_dil

    # %% Set default
    ifgdir = []
    tsadir = []
    ml_factor = []  # Amount to multilook the resulting masks
    err_dil = []  # Amount to binary dilate the error mask
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
            opts, args = getopt.getopt(argv[1:], "hd:t:m:v:e", ["help", "reset", "n_para=", "fullres"])
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
            elif o == '-m':
                ml_factor = int(a)
            elif o == '-v':
                v = int(a) - 1
            elif o == '-e':
                err_dil = int(a)
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

        if not err_dil:
            err_dil = np.ceil(ml_factor / 2)
            print('No iteration number set for error boundary dilation. Setting to {:.0f} (0.5 * ml_factor)'.format(err_dil))

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

    # Find coastlines (or border of data region)
    coast = filters.sobel(~np.isnan(coh).astype('int'))

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

    # %% Remove isolated regions (ie islands) so that if they are outliers we don't waste time on them
    # Label the binary IFG into connected regions
    regions = np.zeros(filled_ifg.shape)
    regions[~np.isnan(filled_ifg)] = 1
    regions, count = label(regions)

    # Get starting pixel
    floodx = np.floor((refx1 + refx2) / 2).astype('int')
    floody = np.floor((refy1 + refy2) / 2).astype('int')

    refregion = regions[floody, floodx]

    filled = filled_ifg.copy()
    filled[np.where(regions != refregion)] = np.nan
    ref = np.nanmean(filled[floody, floodx])

    maxval = np.nanmax(filled)
    minval = np.nanmin(filled)

    tol = 1

    nIter = np.ceil(max([abs(maxval - ref), abs(ref - minval)]) / (tol * np.pi)).astype('int')

    if i == v:
        print('        DeIslanded {:.2f}'.format(time.time() - begin))

    errors = np.zeros(filled.shape)
    flood_next = np.zeros(filled.shape)
    sobel_next = np.zeros(filled.shape)

    for ii in range(0, nIter):
        sobel_prev = sobel_next.copy()
        flood_next = flood(filled, (floody, floodx), tolerance=(ii + 1) * tol * np.pi)

        sobel_next = filters.sobel(flood_next)
        sobel_next = (sobel_next > 0).astype('int')

        new_errors = ((sobel_prev + sobel_next) == 2).astype('int')
        errors = errors + new_errors

    # Add coastline incase gap in the errors means that sea can be flooded
    errors = (errors + coast).astype('bool').astype('int')

    # Add errors to IFG, reinterpolate, and see what the mask does
    max_unw = np.nanmax(ifg)
    mask_ifg = ifg.copy()
    mask_ifg[np.where(errors == 1)] = max_unw * 5

    # Reinterpolate ifg with errors
    mask_ifg = NN_interp(mask_ifg)

    # Identify errors based on the max_unw threshold
    errors = (mask_ifg == (max_unw * 5)).astype('int')

    # Dilate errors to fill holes
    if err_dil > 0:
        errors = binary_dilation(errors, iterations=err_dil).astype('int')

    # Reflood from reference pixel to find the error regions, where 1 is good data and 0 is bad data
    mask = (flood_fill(errors, (floody, floodx), 2) == 2).astype('int')

    if i == v:
        print('        Mask made {:.2f}'.format(time.time() - begin))

    title3 = ['Original unw', 'Interpolated unw / pi', 'Unwrapping Error Mask']

    mask_lib.make_unw_npi_mask_png([unw, (filled_ifg / (np.pi)).round(), mask], os.path.join(ifgdir, date, date + '.mask.png'), [insar, 'tab20c', 'viridis'], title3)

    # %% Save Masked UNW to save time in corrections
    # If working with full res data, load in the ml IFG to be masked
    if fullres:
        masked_ifg = io_lib.read_img(os.path.join(ifgdir, date, date + '.unw'), length, width)
        n_px = sum(sum(~np.isnan(masked_ifg)))
        if i == v:
            print('        Loaded ML{} IFG {:.2f}'.format(ml_factor, time.time() - begin))

        mask = tools_lib.multilook(mask, ml_factor, ml_factor, 0.5)
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
