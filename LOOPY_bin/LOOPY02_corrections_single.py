#!/usr/bin/env python3
"""
========
Overview
========
This script corrects errors in unwrapped interferograms unsing loop closures.
Errors that have been identified in unw IFGs are masked out before applying the
correction

===============
Input & output files
===============
Inputs in GEOCml*/ :
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw
   - yyyymmdd_yyyymmdd.unw_mask
 - slc.mli.par
 - baselines (may be dummy)

Inputs in TS_GEOCml*/ :
 - info/
   - 11bad_ifg.txt      : List of bad ifgs identified in step11
   - 12ref.txt          : Preliminary ref point for SB inversion (X/Y)
   - 12bad_ifg.txt      : List of bad ifgs from LiCSBAS12 (opt, for network)
   - 12bad_ifg_cand.txt : List of bad cand ifgs from LiCSBAS12 (opt, for network)

Outputs in GEOCml*/ :
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw[.png] : Corrected IFG
   - yyyymmdd_yyyymmdd_uncorr.unw[.png] : Original IFG
   - yyyymmdd_yyyymmdd_npi.unw.png : png image of applied correction

 - Corr_dir/
   - yyyymmdd_yyyymmdd_compare.unw.png : png image of correction comparison



Outputs in TS_GEOCml*/ :
 - 12loop/*_loop_png
   - yyyymmdd_yyyymmdd_yyyymmdd_V[0-3].png : png images of progressive corrections

  Not completed: Add in list of stats re-computed


=====
Usage
=====

LOOPY02_corrections.py -d ifgdir [-t tsadir] [--reset] [--n_para]

 -d       Path to the GEOCml* dir containing stack of unw data.
 -t       Path to the output TS_GEOCml* dir. (Default: TS_GEOCml*)
 --reset  Remove previous corrections
 --n_para Number of parallel processing (Default: # of usable CPU) [To be implemented]

=========
Changelog
=========
v1.1 20220615 Jack McGrath, Uni of Leeds
 - Edit to run from command line
v1.0 20220608 Jack McGrath, Uni of Leeds
 - Original implementation
"""

import os
import sys
import time
import getopt
import shutil
# import warnings # Just to shut up the cmap error
import numpy as np
import pandas as pd
# import multiprocessing as multi
# import scipy.stats as stats
import LOOPY_loop_lib as loop_lib
import LiCSBAS_io_lib as io_lib
# import LiCSBAS_inv_lib as inv_lib
import LiCSBAS_plot_lib as plot_lib
import LiCSBAS_tools_lib as tools_lib
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import label
from PIL import Image, ImageFilter

global plot_figures, tol, min_size, refx1, refx2, refy1, refy2, n_ifg, \
    length, width, ifgdir, ifgdates, coh

# %% Set default
ifgdir = os.path.join('D:\\', '073D_13256_001823', 'GEOCml10')
tsadir = []
reset = True
plot_figures = True
nolics12 = False
tol = 0.5  # N value to use for modulo pi division
min_size = 1  # Minimum size of labelled regions


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

    return (dataMode / np.pi).round() * np.pi


# %% Assign Parameters, Paths and Variables
# Restore to uncorrected state
start = time.time()

if not tsadir:
    tsadir = os.path.join(os.path.dirname(ifgdir), 'TS_' + os.path.basename(ifgdir))

# Define Frame, directories and files
loopdir = os.path.join(tsadir, '12loop')
infodir = os.path.join(tsadir, 'info')

# Check to see if LiCSBAS step 12 has aready been run
if not os.path.exists(loopdir):
    nolics12 = True
    print('No Loop Closure Check previously carried out \nLook out for issues from no ref_file')
else:
    bad_ifgfile = os.path.join(infodir, '12bad_ifg.txt')
    bad_ifg_cands_file = os.path.join(infodir, '12bad_ifg_cand.txt')
    ref_file = os.path.join(infodir, '12ref.txt')
    loop_infofile = os.path.join(loopdir, 'loop_info.txt')

mlipar = os.path.join(ifgdir, 'slc.mli.par')
mask_infofile = os.path.join(infodir, 'mask_info.txt')

if reset:
    print('Replacing files')
    loop_lib.reset_loops(ifgdir, loopdir, tsadir)
    if nolics12:
        os.makedirs(loopdir, exist_ok=True)
        os.makedirs(os.path.join(loopdir, 'loop_pngs'), exist_ok=True)
else:
    print('Keeping old files')

# Make folder to hold comparisons of corrected and uncorrected IFGS
if not os.path.exists(os.path.join(ifgdir, 'Corr_dir')):
    os.mkdir(os.path.join(ifgdir, 'Corr_dir'))

print('Loading Information')
# ## Read in information
# Bad ifgs from LiCSBAS11
Step11BadIfgFile = os.path.join(infodir, '11bad_ifg.txt')
Step11BadIfg = io_lib.read_ifg_list(Step11BadIfgFile)

# Read in Loop info (+thresh), bad ifgs and bad cands identified in LiCSBAS12
if nolics12:
    bad_ifg_list = []
    bad_ifg_cands_list = []
    ref_file = []
    thresh = 1.5
else:
    with open(loop_infofile, 'r') as loop_info:
        loop_info = loop_info.read().splitlines()
        thresh = float(loop_info[0].split()[2])
        loop_info = loop_info[4:]

    with open(bad_ifgfile, 'r') as bad_ifg_list:
        bad_ifg_list = bad_ifg_list.read().splitlines()

    with open(bad_ifg_cands_file, 'r') as bad_ifg_cands_list:
        bad_ifg_cands_list = bad_ifg_cands_list.read().splitlines()

with open(mask_infofile, 'r') as mask_info:
    mask_info = mask_info.read().splitlines()
    mask_info = mask_info[2:]

# Get ifg dates
ifgdates = tools_lib.get_ifgdates(ifgdir)

# Remove bad ifgs and images from list (as defined in step 11)
ifgdates = list(set(ifgdates) - set(Step11BadIfg))
ifgdates.sort()
imdates = tools_lib.ifgdates2imdates(ifgdates)
n_ifg = len(ifgdates)

bad_ifg_list = list(set(bad_ifg_list) - set(Step11BadIfg))
bad_ifg_list.sort()
bad_ifg_cands_list = list(set(bad_ifg_cands_list) - set(Step11BadIfg))
bad_ifg_cands_list.sort()

# Get size and baseline data
width = int(io_lib.get_param_par(mlipar, 'range_samples'))
length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))

bperp_file = os.path.join(ifgdir, 'baselines')
if os.path.exists(bperp_file):
    bperp = io_lib.read_bperp_file(bperp_file, imdates)
else:  # dummy
    bperp = np.random.random(len(imdates)).tolist()

bperp = [bperp[i] for i, im in enumerate(imdates) if im in '_'.join(ifgdates).split('_')]
pngfile = os.path.join(tsadir, 'network', 'network12cands_precorrection.png')
plot_lib.plot_cand_network(ifgdates, bperp, bad_ifg_list, pngfile, bad_ifg_cands_list)

# %% Create Loops and dictionaries
print('Creating Loops')
A3loop = loop_lib.make_loop_matrix(ifgdates)

if nolics12 is False:  # Skip check if there has been no check of LiCSBAS12
    for i in range(len(loop_info) - 1, -1, -1):
        loop = loop_info[i]
        for ifg in Step11BadIfg:
            if ifg.split('_')[0] in loop and ifg.split('_')[1] in loop:
                loop_info.remove(loop)
                break

    # Check that there same number of loops have been calculated as LiCSBAS12 did
    if np.shape(A3loop)[0] != np.shape(loop_info)[0]:
        print('ERROR: Different number of Loops in A3loop and loop_info.txt. Stopping')
        sys.exit()

# Create dictionaries of every loop an ifg is in (ifg_dict), and every ifg in a loop (loop_dict)
ifg_dict, loop_dict = loop_lib.create_loop_dict(A3loop, ifgdates)


# %% Create dataframe of IFG number, Date, # of loops, Unmasked pixels, correction status
ifg_df = pd.DataFrame([], columns=['ix', 'Date', 'Loops', 'MeanCov', 'Corrected'])

for count, ifg in enumerate(ifgdates):
    loops = ifg_dict[ifg]
    if len(loops) == 0:
        mean_cov = 1
    else:
        mask_ifgs = []
        for loop in loops:
            mask_ifgs = mask_ifgs + loop_dict[loop]
        mask_ifgs = list(set(mask_ifgs))
        mask_ifgs.remove(count)
        mean_cov = 0
        for i in mask_ifgs:
            mean_cov += float(mask_info[i].split()[1])
        mean_cov = mean_cov / len(mask_ifgs)
    ifg_df.loc[count] = [count, ifg, len(ifg_dict[ifg]), mean_cov, 0]
ifg_df = ifg_df.set_index(['ix'])

# Sort IFG fix order by number of loops, then mean mask coverage (least masked first)
ifg_df = ifg_df.sort_values(['Loops', 'MeanCov'], ascending=[False, False], ignore_index=False)

# %% Start to fix IFGs
master_name = '20150102_20150303'
master_name = '20160601_20160719'
for i, ix in enumerate(ifg_df.index.values):
    ifg_name = ifg_df.loc[ix, 'Date']

    if master_name:
        ifg_name = master_name
        ix = ifg_df[ifg_df['Date'] == master_name].index.tolist()[0]

    # Check if IFG is isolated
    if ifg_df.loc[ix, 'Loops'] == 0:
        print('    Correction ({}/{}):'.format(i + 1, n_ifg), ifg_name, 'NO LOOPS')
        continue
    else:
        print('    Correction ({}/{}):'.format(i + 1, n_ifg), ifg_name)

    # Check if IFG has already been corrected
    if os.path.exists(os.path.join(ifgdir, ifg_name, ifg_name + '_uncorr.unw')):
        continue

    # Find the loops associated with this IFG
    loops2fix = ifg_dict[ifg_name]

    # Variable to store the corrections
    corr_all = np.empty((length, width, len(loops2fix)))

    # For each loop, find the correction that would be required
    for count, loop in enumerate(loops2fix):
        if nolics12 is False:
            print('    ', count, ' ', loop_info[loop])
        ifg_position = loop_lib.calc_bad_ifg_position_single(ifg_name, A3loop[loop, :], ifgdates)
        corr_all[:, :, count] = loop_lib.get_corr(A3loop[loop, :], ifg_position, thresh, ifgdates, ifgdir, length, width, ref_file)
        loop_lib.plotmask(corr_all[:, :, count], centerz=False, title='Corr {}'.format(count), vmax=5 * np.pi, vmin=-5 * np.pi, cmap='tab20c')

    # Find modal correction for each IFG based off all of the loops
    # New method of finding mode, where in the event of 2 modal values, nearest neighbour interp is used
    # Find any pixel where there is a correction value
    py, px = np.where(np.any(~np.isnan(corr_all), axis=2))
    # Prepare array to hold correction
    corr = np.zeros((length, width))
    # For each pixel with a correction
    for ix, y in enumerate(py):
        x = px[ix]
        # Get all correction values for said pixel
        corr_slice = corr_all[y, x, :]
        # Find unique values of the correction, with the counts
        u, c = np.unique(corr_slice[~np.isnan(corr_slice)], return_counts=True)
        # If there is more than 1 potential correction value, set pixel to 1 in corr
        if np.shape(np.where(c == c.max()))[1] > 1:
            corr[y, x] = 1  # Using 1 is ok, as all corrections will = n * pi
        else:  # 1 Modal correction found. Use this
            corr[y, x] = u[np.where(c == c.max())[0][0]]

    # Find all pixels where we have a correction value
    mask = np.where(((~np.isnan(corr)).astype('int') + (corr != 1).astype('int')) == 2)
    # Interpolate SOMEHOW? -> Need to set this so only interpolate to the mask values
    # Make interpolation object,
    interp = NearestNDInterpolator(np.transpose(mask), corr[mask])
    # Interpolate to all places where there was a conflict
    corr_interp = interp(*np.where(corr == 1))
    # Fill in correction with interpolated value
    corr[np.where(corr == 1)] = corr_interp

    # HERE IS A GOOD POINT TO APPLY CORRECTION TO MASK VALUE
    # Set all non-correction values to 0
    corr = np.array(corr, dtype='float32')
    corr[corr == 0] = np.nan
    corr_save = corr.copy()
    ifg = io_lib.read_img(os.path.join(ifgdir, ifg_name, ifg_name + '.unw'), length, width)
    maskfile = os.path.join(ifgdir, ifg_name, ifg_name + '.mask')
    mask = io_lib.read_img(maskfile, length, width, dtype='bool').astype('float32')
    # mask[np.isnan(ifg)] = np.nan
    corr[mask == 0] = np.nan

    loop_lib.plotmask(ifg, centerz=False, title='Orignal IFG', vmax=10 * np.pi, vmin=-10 * np.pi, cmap='tab20c')
    loop_lib.plotmask(corr, centerz=False, title='Correction', vmax=5 * np.pi, vmin=-5 * np.pi, cmap='tab20c')
    loop_lib.plotmask(corr[650:750, 0:200], centerz=False, title='Correction', vmax=5 * np.pi, vmin=-5 * np.pi, cmap='tab20c')
    loop_lib.plotmask(mask, centerz=False, title='MASK1', vmax=1, vmin=0)

    interpolate_corr = True
    if interpolate_corr:
        # Try interpolating correction to error regions
        interp = NearestNDInterpolator(np.transpose(np.where(~np.isnan(corr))), corr[~np.isnan(corr)])

        # Label error regions
        labs, count = label(mask)
        # Find regions with a correction in it
        corr_regions = np.unique(labs[np.where(np.all(np.stack([mask, corr / corr], axis=2) == 1, axis=2))])
        # Interpolate the correction to every region, and add to corr
        corr_interp = interp(*np.where(np.isin(labs, corr_regions)))
        corr[np.where(np.isin(labs, corr_regions))] = corr_interp
        loop_lib.plotmask(corr, centerz=False, title='Interpolated Correction', vmax=5 * np.pi, vmin=-5 * np.pi, cmap='tab20c')
        loop_lib.plotmask(corr[650:750, 0:200], centerz=False, title='Interpolated Correction', vmax=5 * np.pi, vmin=-5 * np.pi, cmap='tab20c')
        corr_unfilt = corr[650:750, 0:200]
        corr = mode_filter(corr, filtSize=5)
        corr_filt = corr[650:750, 0:200]
        loop_lib.plotmask(corr[650:750, 0:200], centerz=False, title='Interpolated + filtered Correction', vmax=5 * np.pi, vmin=-5 * np.pi, cmap='tab20c')
        mask[~np.isnan(corr)] = 0
        # breakpoint()

    # loop_lib.correct_bad_ifg(bad_ifg_name, corr, ifgdir, length, width, A3loop[loops2fix, :], ifgdates, tsadir, ref_file)
    loop_lib.plotmask(corr[650:750, 0:200], centerz=False, title='Interpolated Correction2', vmax=5 * np.pi, vmin=-5 * np.pi, cmap='tab20c')
    breakpoint()
    loop_lib.correct_bad_ifg(ifg_name, corr, ifgdir, length, width, A3loop[loops2fix, :], ifgdates, loopdir, ref_file)
    loop_lib.plotmask(corr[650:750, 0:200], centerz=False, title='Interpolated Correction3', vmax=5 * np.pi, vmin=-5 * np.pi, cmap='tab20c')

    # Generate new mask for corrected IFG based on how much of it's error area has been corrected
#    nocorr = np.all(np.isnan(corr_all), axis=2).astype('float32') == 0  # Areas where everything is masked

    loop_lib.plotmask(mask, centerz=False, title='Old Mask', vmax=1, vmin=0)
    loop_lib.plotmask(mask[650:750, 0:200], centerz=False, title='Old Mask', vmax=1, vmin=0)
    loop_lib.plotmask(corr[650:750, 0:200], centerz=False, title='Interpolated Correction', vmax=5 * np.pi, vmin=-5 * np.pi, cmap='tab20c')

    # mask[~np.isnan(corr)] = 0
    loop_lib.plotmask(mask, centerz=False, title='New Mask', vmax=1, vmin=0)
    unw = io_lib.read_img(os.path.join(ifgdir, ifg_name, ifg_name + '.unw'), length, width)
    loop_lib.plotmask(unw, centerz=False, title='Corrected IFG', vmax=10 * np.pi, vmin=-10 * np.pi, cmap='tab20c')
    loop_lib.plotmask(ifg[650:750, 0:200], centerz=False, title='Original IFG', vmax=10 * np.pi, vmin=-10 * np.pi, cmap='tab20c')
    loop_lib.plotmask(corr[650:750, 0:200], centerz=False, title='Interpolated Correction', vmax=5 * np.pi, vmin=-5 * np.pi, cmap='tab20c')
    loop_lib.plotmask(unw[650:750, 0:200], centerz=False, title='Corrected IFG', vmax=10 * np.pi, vmin=-10 * np.pi, cmap='tab20c')
    unw[mask == 1] = np.nan
    loop_lib.plotmask(unw, centerz=False, title='Corrected IFG w/ newmask', vmax=10 * np.pi, vmin=-10 * np.pi, cmap='tab20c')
    breakpoint()
    # Remove old masked IFG
    # shutil.move(os.path.join(ifgdir, ifg_name, ifg_name + '.unw_mask'), os.path.join(ifgdir, ifg_name, ifg_name + '.unw_maskold'))
    # unw.tofile(os.path.join(ifgdir, ifg_name, ifg_name + '.unw_mask'))
    # if not os.path.exists(os.path.join(ifgdir, ifg_name, ifg_name + '.01_mask')):
    #     shutil.move(os.path.join(ifgdir, ifg_name, ifg_name + '.mask'), os.path.join(ifgdir, ifg_name, ifg_name + '.01_mask'))
    # mask.astype('bool').tofile(os.path.join(ifgdir, ifg_name, ifg_name + '.mask'))

    ifg_df.loc[ix, 'Corrected'] = 1
    breakpoint()

pngfile = os.path.join(tsadir, 'network', 'network12cands_postcorrection.png')
plot_lib.plot_cand_network(ifgdates, bperp, bad_ifg_list, pngfile, bad_ifg_cands_list)

print('\nCorrections complete. Re-run LiCSBAS12_loop_closure.py to find good network')

# %% Finish
elapsed_time = time.time() - start
hour = int(elapsed_time / 3600)
minute = int(np.mod((elapsed_time / 60), 60))
sec = int(np.mod(elapsed_time, 60))
print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour, minute, sec))

print('\nSuccessfully finished!!\n')
print('Output directory: {}\n'.format(tsadir))
