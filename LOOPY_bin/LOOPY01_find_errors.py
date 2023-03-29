#!/usr/bin/env python3
"""
========
Overview
========
This script identifies errors in unwrapped interferograms, and aims to correct
them. It must be run before the application of GACOS and LOOPY03 if being used
on the ful resolution data (as recommended)

1) Read in IFG, interpolate to full area, and find modulo 2pi values
2) Carry out modal filtering to reduce noise
3) Find difference between adjacent pixels
4) Classify an error boundary as anywhere between 2 pixels with a > modulo pi
    difference
5) Add unwrapping errors back into original IFG, and re-interpolate to create
    an IFG seperated into distinct regions by unwrapping errors
6) All pixels in the same region as the reference are to be considered good.
    Correct all regions bordering this with a static correction to reduce the
    difference to < 2pi

Limitations:
    Can't identify regions that are correctly unwrapped, but isolated from main
    pixel, either by being an island, or because it's cut-off by another
    unwrapping error (Latter option would require iterations)
    Needs an unwrapping error to be complete enclosed in order to be masked
    ('Twist' errors can't be identified')

Additional modules to LiCSBAS needed:
- scipy
- skimage

===============
Input & output files
===============
If not working at full res:
Inputs in GEOCml*/:
- yyyymmdd_yyyymmdd/
 - yyyymmdd_yyyymmdd.unw
 - yyyymmdd_yyyymmdd.cc
- slc.mli.par

Inputs in TS_GEOCml */:
- results/coh_avg  : Average coherence

If working at full res:
Inputs in GEOCml*/:
- yyyymmdd_yyyymmdd/

Inputs in GEOC/:
- yyyymmdd_yyyymmdd/
 - yyyymmdd_yyyymmdd.geo.unw.tif
 - yyyymmdd_yyyymmdd.geo.cc.tif
- frame.geo.[E, N, U, hgt, mli]
- slc.mli

Outputs in GEOCml*LoopMask/:
- yyyymmdd_yyyymmdd/
  - yyyymmdd_yyyymmdd.unw[.png] : Corrected unw
  - yyyymmdd_yyyymmdd.cc : Coherence file
  - yyyymmdd_yyyymmdd.errormap.png : png of identified error boundaries
  - yyyymmdd_yyyymmdd.npicorr.png : png of original + corrected unw, with npi maps
  - yyyymmdd_yyyymmdd.maskcorr.png : png of original + corrected unw, original npi map, and correction
- known_errors.png : png of the input know error mask
- other metafiles produced by LiCSBAS02_ml_prep.py

=====
Usage
=====
LOOPY01_find_errors.py -d ifgdir [-t tsadir] [-c corrdir] [-m int] [-e errorfile] [-v int] [--full_res] [--reset] [--n_para]

-d        Path to the GEOCml* dir containing stack of unw data
-t        Path to the output TS_GEOCml* dir. (Default: TS_GEOCml*)
-c        Path to the correction dierectory (Default: GEOCml*LoopMask)
-m        Output multilooking factor (Default: No multilooking of mask, REQUIRED FOR FULL RES)
-e        Text file, where each row is a known error location, in form lon1,lat1,....,lonn,latn
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
import LiCSBAS_io_lib as io_lib
import LiCSBAS_plot_lib as plot_lib
import LiCSBAS_tools_lib as tools_lib
import LOOPY_lib as loopy_lib
from osgeo import gdal
from scipy.stats import mode
from scipy.ndimage import label
from scipy.ndimage import binary_dilation
from scipy.interpolate import NearestNDInterpolator
from skimage import filters
from skimage.filters.rank import modal

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
        length, width, ifgdir, ifgdates, coh, i, v, begin, fullres, geocdir, \
        corrdir, bool_mask, cycle, n_valid_thre

    # %% Set default
    ifgdir = []  # GEOCml* dir
    tsadir = []  # TS_GEOCml* dir
    corrdir = []  # Directory to hold the corrections
    ml_factor = []  # Amount to multilook the resulting masks
    errorfile = []  # File to hold lines containing known errors
    fullres = False
    reset = False
    plot_figures = False
    v = -1
    cycle = 3
    n_valid_thre = 0.5

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
            opts, args = getopt.getopt(argv[1:], "hd:t:c:m:e:v:", ["help", "reset", "n_para=", "fullres"])
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
            elif o == '-c':
                corrdir = a
            elif o == '-m':
                ml_factor = int(a)
            elif o == '-e':
                errorfile = a
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

    if not corrdir:
        corrdir = os.path.join(os.path.dirname(ifgdir), os.path.basename(ifgdir) + 'LoopMask')

    if not os.path.exists(tsadir):
        os.mkdir(tsadir)

    if not os.path.exists(corrdir):
        os.mkdir(corrdir)

    netdir = os.path.join(tsadir, 'network')
    if not os.path.exists(netdir):
        os.mkdir(netdir)

    infodir = os.path.join(tsadir, 'info')
    if not os.path.exists(infodir):
        os.mkdir(infodir)

    resultsdir = os.path.join(tsadir, 'results')
    if not os.path.exists(resultsdir):
        os.mkdir(resultsdir)

    if reset:
        print('Removing Previous Masks')
        if os.path.exists(corrdir):
            shutil.rmtree(corrdir)
    else:
        print('Preserving Premade Masks')

    if not os.path.exists(corrdir):
        loopy_lib.prepOutdir(corrdir, ifgdir)

    # %% File Setting
    ref_file = os.path.join(infodir, '12ref.txt')
    mlipar = os.path.join(ifgdir, 'slc.mli.par')

    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))

    # %% Prepare variables
    # Get ifg dates
    ifgdates = tools_lib.get_ifgdates(ifgdir)
    n_ifg = len(ifgdates)

    # Find how far to interpolate IFG to
    if fullres:
        geocdir = os.path.abspath(os.path.join(ifgdir, '..', 'GEOC'))
        print('Processing full resolution masks direct from tifs in {}'.format(geocdir))

        # Create full res mli
        print('\nCreate slc.mli', flush=True)
        cohfile = os.path.join(ifgdir, ifgdates[0], ifgdates[0])
        mlitif = glob.glob(os.path.join(geocdir, '*.geo.mli.tif'))
        if len(mlitif) > 0:
            mlitif = mlitif[0]  # First one
            coh = gdal.Open(mlitif).ReadAsArray()  # Coh due to previous use of coherence to find IFG limits
            if isinstance(coh, type(None)):
                print('Full Res Coherence == NoneType. Using hgt')
                coh = gdal.Open(glob.glob(os.path.join(geocdir, '*.geo.hgt.tif'))[0]).ReadAsArray()
            coh[coh == 0] = np.nan
            mlifile = os.path.join(geocdir, 'slc.mli')
            coh.tofile(mlifile)
            mlipngfile = mlifile + '.png'
            mli = np.log10(coh)
            vmin = np.nanpercentile(mli, 5)
            vmax = np.nanpercentile(mli, 95)
            plot_lib.make_im_png(mli, mlipngfile, 'gray', 'MLI (log10)', vmin=vmin, vmax=vmax, cbar=True)
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

    if fullres:
        geotiff = gdal.Open(mlitif)
        widthtiff = geotiff.RasterXSize
        lengthtiff = geotiff.RasterYSize
        bool_mask = np.zeros((lengthtiff, widthtiff))
    else:
        bool_mask = np.zeros((length, width))

    if errorfile:
        print('Reading known errors')
        with open(errorfile) as f:
            poly_strings_all = f.readlines()

        if fullres:
            lon_w_p, postlon, _, lat_n_p, _, postlat = geotiff.GetGeoTransform()
            # lat lon are in pixel registration. dlat is negative
            lon1 = lon_w_p + postlon / 2
            lat1 = lat_n_p + postlat / 2
            lat2 = lat1 + postlat * (lengthtiff - 1)  # south
            lon2 = lon1 + postlon * (widthtiff - 1)  # east
            lon, lat = np.linspace(lon1, lon2, widthtiff), np.linspace(lat1, lat2, lengthtiff)
        else:
            dempar = os.path.join(ifgdir, 'EQA.dem_par')
            lat1 = float(io_lib.get_param_par(dempar, 'corner_lat'))  # north
            lon1 = float(io_lib.get_param_par(dempar, 'corner_lon'))  # west
            postlat = float(io_lib.get_param_par(dempar, 'post_lat'))  # negative
            postlon = float(io_lib.get_param_par(dempar, 'post_lon'))  # positive
            lat2 = lat1 + postlat * (length - 1)  # south
            lon2 = lon1 + postlon * (width - 1)  # east
            lon, lat = np.linspace(lon1, lon2, width), np.linspace(lat1, lat2, length)

        for poly_str in poly_strings_all:
            bool_mask = bool_mask + tools_lib.poly_mask(poly_str, lon, lat, radius=2)

        bool_mask[np.where(bool_mask != 0)] = 1
        bool_plot = bool_mask.copy()
        bool_plot[np.where(np.isnan(coh))] = np.nan

        if fullres:
            bool_plot = tools_lib.multilook(bool_plot, ml_factor, ml_factor, n_valid_thre=0.1)
        title = 'Known UNW error Locations)'
        plot_lib.make_im_png(bool_plot, os.path.join(corrdir, 'known_errors.png'), 'viridis', title, vmin=0, vmax=1, cbar=False)
        print('Map of known error locations made')

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

    # %% Run correction in parallel
    _n_para = n_para if n_para < n_ifg else n_ifg
    print('\nRunning error mapping for all {} ifgs, '.format(n_ifg), flush=True)

    if n_para == 1:
        print('with no parallel processing...', flush=True)
        if v >= 0:
            print('In an overly verbose way for IFG {}'.format(v + 1))

        for i in range(n_ifg):
            mask_unw_errors(i)

    else:
        print('with {} parallel processing...'.format(_n_para), flush=True)
        if v >= 0:
            print('In an overly verbose way for IFG {}'.format(v + 1))

        # Parallel processing
        p = q.Pool(_n_para)
        p.map(mask_unw_errors, range(n_ifg))
        p.close()

    # %% Finish
    print('\nCheck network/*, 11bad_ifg_ras/* and 11ifg_ras/* in TS dir.')
    print('If you want to change the bad ifgs to be discarded, re-run with different thresholds or make a ifg list and indicate it by --rm_ifg_list option in the next step.')

    elapsed_time = time.time() - start
    hour = int(elapsed_time / 3600)
    minute = int(np.mod((elapsed_time / 60), 60))
    sec = int(np.mod(elapsed_time, 60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour, minute, sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output directory: {}\n'.format(os.path.relpath(corrdir)))


# %% Function to mask unwrapping errors
def mask_unw_errors(i):
    global begin
    begin = time.time()
    date = ifgdates[i]
    if i == v:
        print('        Starting')
    if not os.path.exists(os.path.join(corrdir, date)):
        os.mkdir(os.path.join(corrdir, date))
    if os.path.exists(os.path.join(corrdir, date, date + '.unw')):
        print('    ({}/{}): {}  Mask Exists. Skipping'.format(i + 1, n_ifg, date))
        return
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
    if np.all(np.isnan(unw[refy1:refy2, refx1:refx2])):
        print('Invalid Ref Value found for IFG {}. Setting to 0'.format(date))
        ref = 0
    else:
        ref = np.nanmean(unw[refy1:refy2, refx1:refx2])

    ifg = unw.copy()
    ifg = ifg - ref  # Maybe no need to use a reference - would be better to subtract 0.5 pi or something, incase IFG is already referenced
    if i == v:
        print('        Reffed {:.2f}'.format(time.time() - begin))
    # %%
    # Interpolate IFG to entire frame
    filled_ifg = NN_interp(ifg)

    if i == v:
        print('        Interpolated {:.2f}'.format(time.time() - begin))

    # Find modulo 2pi values for original IFG, and after adding and subtracting 1pi
    # Use round rather than // to account for slight noise
    nPi = 1
    npi = (filled_ifg / (nPi * np.pi)).round()

    if i == v:
        print('        npi_calculated {:.2f}'.format(time.time() - begin))

    # Modal filtering of npi images
    start = time.time()
    npi = mode_filter(npi, filtSize=21)

    if i == v:
        print('        Scipy filtered {:.2f} ({:.2f} s)'.format(time.time() - begin, time.time() - start))

    # %%
    errors = np.zeros(npi.shape) * np.nan
    errors[np.where(~np.isnan(npi))] = 0

    # Compare with 1 row below
    error_rows, error_cols = np.where((np.abs(npi[:-1, :] - npi[1:, :]) > 1))
    errors[error_rows, error_cols] = 1

    # Compare with 1 row above
    error_rows, error_cols = np.where((np.abs(npi[1:, :] - npi[:-1, :]) > 1))
    errors[error_rows + 1, error_cols] = 1

    # Compare to column to the left
    error_rows, error_cols = np.where((np.abs(npi[:, 1:] - npi[:, :-1]) > 1))
    errors[error_rows, error_cols] = 1

    # Compare to column to the right
    error_rows, error_cols = np.where((np.abs(npi[:, :-1] - npi[:, 1:]) > 1))
    errors[error_rows, error_cols + 1] = 1

    # Add know error locations
    errors[np.where(bool_mask == 1)] = 1

    if i == v:
        print('        Boundaries Classified {:.2f}'.format(time.time() - begin))

    # %%
    # Add error lines to the original IFG, and interpolate with these values to
    # create IFG split up by unwrapping error boundaries
    ifg2 = unw.copy()
    if i == v:
        print('        Copied unw {:.2f}'.format(time.time() - begin))
    err_val = 10 * np.nanmax(ifg2)
    if i == v:
        print('        err_val set {:.2f}'.format(time.time() - begin))

    ifg2[np.where(errors == 1)] = err_val
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
    timer=time.time()
    too_small = np.where(np.isin(regions, drop_regions))
    error_interp = np.where((regions == 0) == ~np.isnan(coh))
    interp_to = (np.concatenate((too_small[0], error_interp[0])), np.concatenate((too_small[1], error_interp[1])))
    borders = np.zeros(regions.shape)
    borders[interp_to] = 1
    border_dil = binary_dilation(borders, structure=np.ones((3, 3))).astype('int')
    borders = np.where(borders != border_dil)

    regions[np.where(np.isin(regions, np.append(drop_regions, 0)))] = np.nan

    # Cease if there are no regions left to be checked
    if drop_regions.shape[0] == regionId.shape[0]:
        correction = np.zeros(unw.shape)
    else:
        # Remove error areas and tiny regions from NPI so they match
        npi_corr = npi.copy()
        npi_interp = np.ones(npi.shape) * np.nan
        npi_interp[borders] = npi[borders]
        # Reinterpolate without tiny regions

        regions_interp = np.ones(npi.shape) * np.nan
        regions_interp[borders] = regions[borders]
        if i == v:
            print('interp prep in {:.2f} secs'.format(time.time() -  timer))
        timer = time.time()
        regions_interp = NN_interp(regions_interp, interp_to=interp_to)
        regions[interp_to] = regions_interp[interp_to]
        npi_interp = NN_interp(npi_interp, interp_to=interp_to)
        npi_corr[interp_to] = npi_interp[interp_to]
        if i == v:
            print('interp in {:.2f} secs'.format(time.time() -  timer))

        # Find region number of reference pixel. All pixels in this region to be
        # considered unw error free. Mask where 1 == good pixel, 0 == bad
        # Use mode incase ref area is > 1 pixel (eg if working at full res)
        ref_region = mode(regions[refy1:refy2, refx1:refx2].flatten(), keepdims=True)[0][0]
        mask = regions == ref_region

        if i == v:
            print('        Mask made {:.2f}'.format(time.time() - begin))
        # breakpoint()
        # Make an array exclusively holding the good values
        good_vals = np.zeros(mask.shape) * np.nan
        good_vals[mask] = npi_corr[mask]

        # Make an array to hold the correction
        correction = np.zeros(mask.shape)

        # Boolean array of the outside boundary of the good mask
        good_border = filters.sobel(mask).astype('bool')
        corr_regions = np.unique(regions[good_border])
        corr_regions = np.delete(corr_regions, np.array([np.where(corr_regions == ref_region)[0][0], np.where(np.isnan(corr_regions))[0][0]])).astype('int')

        if i == v:
            print('        Preparing Corrections {:.2f}'.format(time.time() - begin))
    # %%
        for ii, corrIx in enumerate(corr_regions):
            # Make map only of the border regions
            start = time.time()
            border_regions = np.zeros(mask.shape)
            border_regions[good_border] = regions[good_border]
            # Plot boundary in isolation
            border = np.zeros(mask.shape).astype('int')
            border[np.where(border_regions == corrIx)] = 1
            # Dilate boundary so it crosses into both regions
            border_dil = binary_dilation(border).astype('int')

            av_err = mode(npi_corr[np.where(border == 1)], nan_policy='omit', keepdims=False)[0]
            av_good = mode(good_vals[np.where(border_dil == 1)], nan_policy='omit', keepdims=False)[0]

            corr_val = ((av_good - av_err) * (nPi / 2)).round() * 2 * np.pi
            correction[np.where(regions == corrIx)] = corr_val
            if i == v:
                print('AV ERR')
                print(np.unique(npi_corr[np.where(border == 1)], return_counts=True))
                print('AV GOOD')
                print(np.unique(good_vals[np.where(border_dil == 1)], return_counts=True))
            if i == v:
                print('            Done {:.0f}/{:.0f}: {:.2f} rads ({:.1f} - {:.1f}) {:.2f} secs'.format(ii + 1, len(corr_regions), corr_val, av_good, av_err, time.time() - start))
        if i == v:
            print('        Correction Calculated {:.2f}'.format(time.time() - begin))

    # Apply correction to original version of IFG
    corr_unw = unw.copy()
    if i == v:
        print('        UNW copied {:.2f}'.format(time.time() - begin))
    corr_unw[np.where(~np.isnan(corr_unw))] = corr_unw[np.where(~np.isnan(corr_unw))] + correction[np.where(~np.isnan(corr_unw))]
    if i == v:
        print('        Correction Applied {:.2f}'.format(time.time() - begin))

    # %% Multilook mask if required
    if fullres:
        unw = tools_lib.multilook(unw, ml_factor, ml_factor, n_valid_thre=n_valid_thre)
        if i == v:
            print('        Original IFG multilooked {:.2f}'.format(time.time() - begin))
        mask = tools_lib.multilook(mask, ml_factor, ml_factor, 0.1).astype('bool').astype('int')
        if i == v:
            print('        Mask multilooked {:.2f}'.format(time.time() - begin))
        npi = tools_lib.multilook((filled_ifg / (np.pi)).round(), ml_factor, ml_factor, n_valid_thre=0.1)
        if i == v:
            print('        Modulo NPI multilooked {:.2f}'.format(time.time() - begin))
        correction = tools_lib.multilook(correction, ml_factor, ml_factor, n_valid_thre=0.1)
        if i == v:
            print('        Correction multilooked {:.2f}'.format(time.time() - begin))
        corr_unw = tools_lib.multilook(corr_unw, ml_factor, ml_factor, n_valid_thre=n_valid_thre)
        if i == v:
            print('        Corrected IFG multilooked {:.2f}'.format(time.time() - begin))
        errors = tools_lib.multilook(errors, ml_factor, ml_factor, n_valid_thre=0.1)
        if i == v:
            print('        Error map multilooked {:.2f}'.format(time.time() - begin))

    # %% Make PNGs

    # Flip round now, so 1 = bad pixel, 0 = good pixel
    mask = (mask == 0).astype('int')
    mask[np.where(np.isnan(unw))] = 0
    title = '{} ({}pi/cycle)'.format(date, cycle * 2)
    plot_lib.make_im_png(np.angle(np.exp(1j * corr_unw / cycle) * cycle), os.path.join(corrdir, date, date + '.unw.png'), SCM.romaO, title, vmin=-np.pi, vmax=np.pi, cbar=False)
    # Make new unw file from corrected data and new loop png
    corr_unw.tofile(os.path.join(corrdir, date, date + '.unw'))
    mask.astype('bool').tofile(os.path.join(corrdir, date, date + '.mask'))
    # Create correction png image (UnCorr_unw, npi, correction, Corr_unw)
    corrcomppng = os.path.join(corrdir, date, date + '.maskcorr.png')
    titles4 = ['{} Uncorrected'.format(ifgdates[i]),
               '{} Corrected'.format(ifgdates[i]),
               'Modulo nPi',
               'Mask Correction (n * 2Pi)']
    loopy_lib.make_compare_png(unw, corr_unw, npi, correction / (2 * np.pi), corrcomppng, titles4, 3)

    title = 'Error Map'
    plot_lib.make_im_png(errors, os.path.join(corrdir, date, date + '.errormap.png'), 'viridis', title, vmin=0, vmax=1, cbar=False)

    if i == v:
        print('        pngs made {:.2f}'.format(time.time() - begin))

    # Link to the cc file
    shutil.copy(os.path.join(ifgdir, date, date + '.cc'), os.path.join(corrdir, date, date + '.cc'))

    if i == v:
        print('        Saved {:.2f}'.format(time.time() - begin))

    return


# %% Function to carry out nearest neighbour interpolation
def NN_interp(data, interp_to=[]):
    mask = np.where(~np.isnan(data))
    interp = NearestNDInterpolator(np.transpose(mask), data[mask])
    interped_data = data.copy()
    if not interp_to:
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

# %% main
if __name__ == "__main__":
    sys.exit(main())
