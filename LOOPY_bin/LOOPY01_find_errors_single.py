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
LOOPY01_find_errors.py -d ifgdir [-t tsadir] [-m int] [--fullres] [--reset] [--n_para]

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
import LOOPY_loop_lib as loop_lib
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib
from osgeo import gdal
from PIL import Image, ImageFilter
from scipy.ndimage import label
from scipy.ndimage import binary_closing, binary_opening, binary_dilation, binary_erosion
from scipy.interpolate import NearestNDInterpolator
from scipy.stats import mode
from skimage import filters
from numba import jit
from skimage.filters.rank import modal

insar = tools_lib.get_cmap('SCM.romaO')

global plot_figures, tol, ml_factor, refx1, refx2, refy1, refy2, n_ifg, \
    length, width, ifgdir, ifgdates, coh, i, v, begin, fullres, geocdir

# %% Set default
frame = '073D_13256_001823'  # '073D_13256_001823' '023A_13470_171714'
ifgdir = os.path.join('D:\\', 'LiCSBAS_singleFrames', frame, 'GEOCml10')
tsadir = []
ml_factor = 10  # Amount to multilook the resulting masks
fullres = True
reset = True
plot_figures = True
v = 0

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


# %% Function to carry out nearest neighbour interpolation
def NN_interp(data):
    mask = np.where(~np.isnan(data))
    interp = NearestNDInterpolator(np.transpose(mask), data[mask])
    interped_data = data.copy()
    interp_to = np.where(((~np.isnan(coh)).astype('int') + np.isnan(data).astype('int')) == 2)
    nearest_data = interp(*interp_to)
    interped_data[interp_to] = nearest_data
    return interped_data


# %% Function to modally filter arrays using skilearn
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
    # return data:
    return dataMode


# %% Function to mask unwrapping errors
i = 0
begin = time.time()
date = '20161116_20161122'  # 073D
# date = '20141115_20150303'  # 073D
date = '20160226_20161116'  # 073D
# date = '20150327_20151216'  # 073D
# date = '20161128_20161228'  # 073D
# date = '20170118_20181227'  # 023A
# date = '20160118_20171021'
# date = '20150920_20151014'
# date = '20170118_20181016'  # 023A

focus = False
if not fullres:
    focus = False
if focus:
    x1 = 1000
    x2 = 2500
    y1 = 3500
    y2 = 6000
    x1 = 500
    x2 = 1750
    y1 = 6450
    y2 = 7250
else:
    x1 = y1 = 1
    x2 = width
    y2 = length

if i == v:
    print('        Starting')
    print('    ({}/{}): {}'.format(i + 1, n_ifg, date))

if i == v:
    print('        Loading')

# Read in IFG
if fullres:
    unw = gdal.Open(os.path.join(geocdir, date, date + '.geo.unw.tif')).ReadAsArray()
    unw[unw == 0] = np.nan
    # loop_lib.plotmask(unw, centerz=False, title='Original IFG')

    # cc_tiffile = os.path.join(geocdir, date, date + '.geo.cc.tif')
    # cc = gdal.Open(cc_tiffile).ReadAsArray()
    # if cc.dtype == 'uint8':
    #     cc = cc / 255

    # cc_thresh = 0.002
    # unw[cc < cc_thresh] = np.nan
    # loop_lib.plotmask(unw, centerz=False, title='CC-masked IFG {:.2f}'.format(cc_thresh))

else:
    unw = io_lib.read_img(os.path.join(ifgdir, date, date + '.unw'), length=length, width=width)

if i == v:
    print('        UNW Loaded {:.2f}'.format(time.time() - begin))

# Find Reference Value, and reference all IFGs to same value
try:
    ref = np.nanmean(unw[refy1:refy2, refx1:refx2])
except 'RunTimeWarning':
    print('Invalid Ref Value found for IFG {}. Setting to 0'.format(date))
    ref = 0

ifg = unw.copy()
ifg = ifg - ref  # Maybe no need to use a reference - would be better to subtract 0.5 pi or something, incase IFG is already referenced
if i == v:
    print('        Reffed {:.2f}'.format(time.time() - begin))
# %%
# Interpolate IFG to entire frame
filled_ifg = NN_interp(ifg)
# breakpoint()
if plot_figures:
    loop_lib.plotmask(filled_ifg, centerz=False, title='Filled IFG')
    if focus:
        loop_lib.plotmask(filled_ifg[y1:y2, x1:x2], centerz=False, title='Filled IFG')
        loop_lib.plotmask(ifg[y1:y2, x1:x2], centerz=False, title='Orignal IFG')

if i == v:
    print('        Interpolated {:.2f}'.format(time.time() - begin))

# Find modulo 2pi values for original IFG, and after adding and subtracting 1pi
# Use round rather than // to account for slight noise
nPi = 1
npi_og = (filled_ifg / (nPi * np.pi)).round()

if plot_figures:
    maxpi = np.nanmax(npi_og)
    minpi = np.nanmin(npi_og)
    loop_lib.plotmask(npi_og, centerz=False, title='N {:.0f}PI'.format(nPi), cmap='tab20c')  # , vmin=-4.75, vmax=4.75)

if i == v:
    print('        npi_calculated {:.2f}'.format(time.time() - begin))

# npi_numba = numba_filter(np.ones((5, 5)), filtSize=3)
# npi_numba = numba_filter(npi_og, filtSize=21)
if i == v:
    print('        numba filtered {:.2f}'.format(time.time() - begin))
# Modal filtering of npi images
start = time.time()
npi_og = mode_filter(npi_og, filtSize=21)

if i == v:
    print('        Scipy filtered {:.2f} ({:.2f} s)'.format(time.time() - begin, time.time() - start))

if plot_figures:
    loop_lib.plotmask(npi_og, centerz=False, title='NPI filter', cmap='tab20c', vmin=minpi, vmax=maxpi)
    if focus:
        loop_lib.plotmask(npi_og[y1:y2, x1:x2], centerz=False, title='NPI filter', cmap='tab20c')  # , vmin=-4.75, vmax=4.75)
if i == v:
    print('            npi 0 filter  {:.2f}'.format(time.time() - begin))
# %%
errors = np.zeros(npi_og.shape) * np.nan
errors[np.where(~np.isnan(npi_og))] = 0

# Compare with 1 row below
error_rows, error_cols = np.where((np.abs(npi_og[:-1, :] - npi_og[1:, :]) > 1))
errors[error_rows, error_cols] = 1

# Compare with 1 row above
error_rows, error_cols = np.where((np.abs(npi_og[1:, :] - npi_og[:-1, :]) > 1))
errors[error_rows + 1, error_cols] = 1

# Compare to column to the left
error_rows, error_cols = np.where((np.abs(npi_og[:, 1:] - npi_og[:, :-1]) > 1))
errors[error_rows, error_cols] = 1

# Compare to column to the right
error_rows, error_cols = np.where((np.abs(npi_og[:, :-1] - npi_og[:, 1:]) > 1))
errors[error_rows, error_cols + 1] = 1
loop_lib.plotmask(errors, centerz=False, title='Error Boundaries')
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

if plot_figures:
    loop_lib.plotmask(filled_ifg2, centerz=False, title='Filled IFG2')

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

if plot_figures:
    loop_lib.plotmask(regions, centerz=False, title='Regions')

if i == v:
    print('        Added to IFG {:.2f}'.format(time.time() - begin))

# Set a minimum size of region to be corrected
min_corr_size = ml_factor * ml_factor * 10  # 10 pixels at final ml size

# Region IDs to small to corrected
drop_regions = regionId[np.where(regionSize < min_corr_size)]
regions[np.where(np.isin(regions, np.append(drop_regions, 0)))] = np.nan

# Cease if there are no regions left to be checked
if drop_regions.shape[0] == regionId.shape[0]:
    correction = np.zeros(unw.shape)
else:
    # Reinterpolate without tiny regions
    regions = NN_interp(regions)

    # Find region number of reference pixel. All pixels in this region to be
    # considered unw error free. Mask where 1 == good pixel, 0 == bad
    # Use mode incase ref area is > 1 pixel (eg if working at full res)
    ref_region = mode(regions[refy1:refy2, refx1:refx2].flatten(), keepdims=True)[0][0]
    mask = regions == ref_region

    if plot_figures:
        if focus:
            loop_lib.plotmask(mask[y1:y2, x1:x2], centerz=False, title='Mask')
        loop_lib.plotmask(mask, centerz=False, title='Mask')

    if i == v:
        print('        Mask made {:.2f}'.format(time.time() - begin))
    # breakpoint()
    # Make an array exclusively holding the good values
    good_vals = np.zeros(mask.shape) * np.nan
    good_vals[mask] = npi_og[mask]

    # Make an array to hold the correction
    correction = np.zeros(mask.shape)

    # Boolean array of the outside boundary of the good mask
    good_border = filters.sobel(mask).astype('bool')
    corr_regions = np.unique(regions[good_border])
    corr_regions = np.delete(corr_regions, np.array([np.where(corr_regions == ref_region)[0][0], np.where(np.isnan(corr_regions))[0][0]])).astype('int')
#%%
    for ii, corrIx in enumerate(corr_regions):
        # Make map only of the border regions
        start = time.time()
        border_regions = np.zeros(mask.shape)
        if ii % 50 == 0:
            print('1 {:.2f}'.format(time.time() - start))
        border_regions[good_border] = regions[good_border]
        if ii % 50 == 0:
            print('2 {:.2f}'.format(time.time() - start))
        # Plot boundary in isolation
        border = np.zeros(mask.shape).astype('int')
        if ii % 50 == 0:
            print('3 {:.2f}'.format(time.time() - start))
        border[np.where(border_regions == corrIx)] = 1
        if ii % 50 == 0:
            print('4 {:.2f}'.format(time.time() - start))
        # Dilate boundary so it crosses into both regions
        border_dil = binary_dilation(border).astype('int')
        if ii % 50 == 0:
            print('5 {:.2f}'.format(time.time() - start))
        av_err = mode(npi_og[np.where(border == 1)], nan_policy='omit', keepdims=False)[0]
        if ii % 50 == 0:
            print('6 {:.2f}'.format(time.time() - start))
        av_good = mode(good_vals[np.where(border_dil == 1)], nan_policy='omit', keepdims=False)[0]
        if ii % 50 == 0:
            print('7 {:.2f}'.format(time.time() - start))

        corr_val = ((av_good - av_err) * (nPi / 2)).round() * 2 * np.pi
        correction[np.where(regions == corrIx)] = corr_val
        print('Done {:.0f}/{:.0f}: {:.2f} rads ({:.1f} - {:.1f}) {:.2f} secs'.format(ii + 1, len(corr_regions), corr_val, av_good, av_err, time.time() - start))

# Apply correction to original version of IFG
loop_lib.plotmask(correction / (2 * np.pi), centerz=False, title='Correction', cmap='tab20c')
#%%
corr_unw = unw.copy()
corr_unw[np.where(~np.isnan(corr_unw))] = corr_unw[np.where(~np.isnan(corr_unw))] + correction[np.where(~np.isnan(corr_unw))]
breakpoint()
# %% Make PNGs
title3 = ['Original unw', 'Interpolated unw / pi', 'Unwrapping Error Mask']
mask_lib.make_unw_npi_mask_png([unw, (filled_ifg / (np.pi)).round(), mask], os.path.join(ifgdir, date, date + '.mask.png'), [insar, 'tab20c', 'viridis'], title3)

title3 = ['Original unw', 'Correction', 'Corrected IFG']
mask_lib.make_unw_mask_corr_png([unw, correction, corr_unw], os.path.join(ifgdir, date, date + '.corr.png'), [insar, 'tab20c', insar], title3)
breakpoint()
# %% Save Masked UNW to save time in corrections
masked_ifg = unw.copy().astype('float32')
masked_ifg[mask == 0] = np.nan
if i == v:
    print('        IFG masked {:.2f}'.format(time.time() - begin))

unmasked_percent = sum(sum(~np.isnan(masked_ifg))) / sum(sum(~np.isnan(unw)))
mask_coverage = sum(sum(mask == 1))  # Number of pixels that are unmasked
if i == v:
    print('        {}/{} pixels unmasked ({}) {:.2f}'.format(sum(sum(~np.isnan(masked_ifg))), sum(sum(~np.isnan(unw))), unmasked_percent, time.time() - begin))

# %% Multilook mask if required
if fullres:
    mask = tools_lib.multilook(mask, ml_factor, ml_factor, 0.1).astype('bool').astype('int')
    if i == v:
        print('        Mask multilooked {:.2f}'.format(time.time() - begin))
#    mask = (mask > 0.5)
#    if i == v:
#        print('        Mask re-binarised {:.2f}'.format(time.time() - begin))
    masked_ifg = tools_lib.multilook(masked_ifg, ml_factor, ml_factor, 0.1)
    corr_unw = tools_lib.multilook(corr_unw, ml_factor, ml_factor, 0.1)
    unw = tools_lib.multilook(unw, ml_factor, ml_factor, 0.1)
    if i == v:
        print('        Masked IFG multilooked {:.2f}'.format(time.time() - begin))
    titles = ['UNW', 'ML{} Mask'.format(ml_factor)]
    mask_lib.make_npi_mask_png([unw, mask], os.path.join(ifgdir, date, date + '.ml_mask.png'), [insar, 'viridis'], titles)
    if i == v:
        print('        Multilooked png made {:.2f}'.format(time.time() - begin))

# Flip round now, so 1 = bad pixel, 0 = good pixel
mask = (mask == 0).astype('int')
mask[np.where(np.isnan(unw))] = 0

# Backup original unw file and loop png
shutil.move(os.path.join(ifgdir, date, date + '.unw'), os.path.join(ifgdir, date, date + '_uncorr.unw'))
shutil.move(os.path.join(ifgdir, date, date + '.unw.png'), os.path.join(ifgdir, date, date + '_uncorr.unw.png'))
title = '{} ({}pi/cycle)'.format(date, 3 * 2)
plot_lib.make_im_png(np.angle(np.exp(1j * corr_unw / 3) * 3), os.path.join(ifgdir, date, date + '.unw.png'), SCM.romaO, title, -np.pi, np.pi, cbar=False)

# Make new unw file from corrected data and new loop png
corr_unw.tofile(os.path.join(ifgdir, date, date + '.unw'))

mask.astype('bool').tofile(os.path.join(ifgdir, date, date + '.mask'))
masked_ifg.tofile(os.path.join(ifgdir, date, date + '.unw_mask'))

if plot_figures:
    loop_lib.plotmask(mask, centerz=False, title='Inverted Mask')

if i == v:
    print('        Saved {:.2f}'.format(time.time() - begin))

    print('Finished {:.2f}'.format(time.time() - begin))
