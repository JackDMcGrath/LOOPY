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


=========
Changelog
=========
v1.0 20220608 Jack McGrath, Uni of Leeds
 - Original implementation
"""

# import pdb
import os
import sys
import time
import shutil
import warnings # Just to shut up the cmap error
import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats as stats
import LiCSBAS_io_lib as io_lib
import LiCSBAS_inv_lib as inv_lib
import LiCSBAS_plot_lib as plot_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_loopy_lib14 as loop_lib

# class Usage(Exception):
#     """Usage context manager"""
#     def __init__(self, msg):
#         self.msg = msg


# def main(argv=None):

# global ifgdates, ifgdir, tsadir, length, width, refx1, refx2, refy1, refy2

warnings.filterwarnings('ignore')
#%% Assign Parameters, Paths and Variables
# Restore to uncorrected state
clear_directory = False
start = time.time()

# Define Frame, directories and files
frame = '028A_05817_131313'
# frame = '073D_13256_001823'
print(frame)
homedir = 'D:\\' + frame
ifgdir = 'GEOCml10GACOS'
tsadir = os.path.join(homedir, 'TS_'+ifgdir)
ifgdir = os.path.join(homedir, ifgdir)
loopdir = os.path.join(tsadir,'12loop')
infodir = os.path.join(tsadir,'info')
resultsdir = os.path.join(tsadir,'results')

loop_infofile = os.path.join(loopdir, 'loop_info.txt')
bad_ifgfile = os.path.join(infodir, '12bad_ifg.txt')
bad_ifg_cands_file = os.path.join(infodir, '12bad_ifg_cand.txt')
ref_file = os.path.join(infodir,'12ref.txt')
mlipar = os.path.join(ifgdir, 'slc.mli.par')
mask_infofile = os.path.join(infodir,'mask_info.txt')

if clear_directory:
    print('Replacing files')
    loop_lib.reset_loops(ifgdir, loopdir, tsadir)
else:
    print('Keeping old files')

# Make folder to hold comparisons of corrected and uncorrected IFGS
if not os.path.exists(os.path.join(ifgdir, 'Corr_dir')):
    os.mkdir(os.path.join(ifgdir, 'Corr_dir'))

print('Loading Information')
### Read in information
# Bad ifgs from LiCSBAS11
Step11BadIfgFile = os.path.join(infodir, '11bad_ifg.txt')
Step11BadIfg = io_lib.read_ifg_list(Step11BadIfgFile)

# Read in Loop info (+ thresh), bad ifgs and bad cands identified in LiCSBAS12
with open(loop_infofile,'r') as loop_info:
    loop_info = loop_info.read().splitlines()
    thresh = float(loop_info[0].split()[2])
    loop_info = loop_info[4:]

with open(bad_ifgfile,'r') as bad_ifg_list:
    bad_ifg_list = bad_ifg_list.read().splitlines()

with open(bad_ifg_cands_file,'r') as bad_ifg_cands_list:
    bad_ifg_cands_list = bad_ifg_cands_list.read().splitlines()

with open(mask_infofile,'r') as mask_info:
    mask_info = mask_info.read().splitlines()
    mask_info = mask_info[2:]
    
# Get ifg dates
ifgdates = tools_lib.get_ifgdates(ifgdir)


# Remove bad ifgs and images from list (as defined in step 11)
ifgdates = list(set(ifgdates)-set(Step11BadIfg))
ifgdates.sort()
imdates = tools_lib.ifgdates2imdates(ifgdates)
n_ifg = len(ifgdates)
n_im = len(imdates)

bad_ifg_list = list(set(bad_ifg_list)-set(Step11BadIfg))
bad_ifg_list.sort()
bad_ifg_cands_list = list(set(bad_ifg_cands_list)-set(Step11BadIfg))
bad_ifg_cands_list.sort()

# Get size and baseline data
width = int(io_lib.get_param_par(mlipar, 'range_samples'))
length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))

bperp_file = os.path.join(ifgdir, 'baselines')
if os.path.exists(bperp_file):
    bperp = io_lib.read_bperp_file(bperp_file, imdates)
else: #dummy
    bperp = np.random.random(len(imdates)).tolist()

bperp = [bperp[i] for i, im in enumerate(imdates) if im in '_'.join(ifgdates).split('_')]
pngfile = os.path.join(tsadir, 'network', 'network12cands_precorrection.png')
plot_lib.plot_cand_network(ifgdates, bperp, bad_ifg_list, pngfile, bad_ifg_cands_list)

# Assign bad and cand ifgs to variable
n_bad_ifg = len(bad_ifg_list)
n_bad_ifg_cands = len(bad_ifg_cands_list)

bad_ifg_ix = [x for x,ifg in enumerate(ifgdates) if ifg in bad_ifg_list]
bad_ifg_cands_ix = [x for x,ifg in enumerate(ifgdates) if ifg in bad_ifg_cands_list]

cohfile=os.path.join(resultsdir,'coh_avg')
coh=io_lib.read_img(cohfile,length=length, width=width)

del bad_ifgfile, bad_ifg_cands_file, mlipar, Step11BadIfgFile
del clear_directory, frame, homedir, loop_infofile, bperp_file

#%% Create Loops and dictionaries
print('Creating Loops')
A3loop = loop_lib.make_loop_matrix(ifgdates)

for i in range(len(loop_info)-1,-1,-1):
    loop = loop_info[i]
    for ifg in Step11BadIfg:
        if ifg.split('_')[0] in loop and ifg.split('_')[1] in loop:
            loop_info.remove(loop)
            break
        
# Check that there same number of loops have been calculated as LiCSBAS12 did
if np.shape(A3loop)[0] != np.shape(loop_info)[0]:
    print('ERROR: Different number of Loops in A3loop and loop_info.txt. Stopping')
    sys.exit()

ifg_dict, loop_dict = loop_lib.create_loop_dict(A3loop, ifgdates)

#%%
ifg_df = pd.DataFrame([], columns=['ix', 'Date','Loops', 'MeanCov', 'Corrected'])

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
        mean_cov = mean_cov/len(mask_ifgs)
    ifg_df.loc[count] = [count, ifg, len(ifg_dict[ifg]), mean_cov, 0]
ifg_df = ifg_df.set_index(['ix'])
ifg_df = ifg_df.sort_values(['Loops','MeanCov'], ascending=[False, True], ignore_index=False)


for i,ix in enumerate(ifg_df.index.values):
    ifg_name = ifg_df.loc[ix,'Date']
    
    if ifg_df.loc[ix,'Loops'] == 0:
        print('    Correction ({}/{}):'.format(i+1, n_ifg), ifg_name, 'NO LOOPS')
        continue
    else:
        print('    Correction ({}/{}):'.format(i+1, n_ifg), ifg_name)
    
    if os.path.exists(os.path.join(ifgdir,ifg_name,ifg_name+'_uncorr.unw')):
        continue
        
    loops2fix = ifg_dict[ifg_name]
    
    
    corr_all = np.empty((length,width,len(loops2fix)))

    for count, loop in enumerate(loops2fix):
        print('    ',loop_info[loop])
        ifg_position = loop_lib.calc_bad_ifg_position_single(ifg_name, A3loop[loop,:], ifgdates)
        corr_all[:,:,count] = loop_lib.get_corr(A3loop[loop,:], ifg_position, thresh, ifgdates, ifgdir, length, width, ref_file)
    
    corr, corrcount = stats.mode(corr_all, axis=2, nan_policy='omit')
    corr = np.array(corr[:,:,0])
    corr[corr==0] = np.nan
    corr = np.array(corr, dtype='float32')

    # loop_lib.correct_bad_ifg(bad_ifg_name,corr,ifgdir, length, width, A3loop[loops2fix,:], ifgdates, tsadir, ref_file)
    loop_lib.correct_bad_ifg(ifg_name,corr,ifgdir, length, width, A3loop[loops2fix,:], ifgdates, loopdir, ref_file)
    
    # Generate new mask for corrected IFG based on how much of it's error area has been corrected
    nocorr=np.all(np.isnan(corr_all),axis=2).astype('float32') # Areas where everything is masked
    mask = io_lib.read_img(os.path.join(ifgdir,ifg_name,ifg_name+'.mask'), length, width, dtype='bool').astype('float32')
    newmask=np.all(np.dstack([mask,nocorr])==1,axis=2)
    unw = io_lib.read_img(os.path.join(ifgdir,ifg_name,ifg_name+'.unw'), length, width)
    unw[newmask == 1] = np.nan
    # Remove old masked IFG
    shutil.move(os.path.join(ifgdir, ifg_name, ifg_name + '.unw_mask'), os.path.join(ifgdir, ifg_name, ifg_name + '.unw_maskold'))
    unw.tofile(os.path.join(ifgdir,ifg_name,ifg_name+'.unw_mask'))

    ifg_df.loc[ix, 'Corrected'] = 1




sys.exit()


           
pngfile = os.path.join(tsadir, 'network', 'network12cands_postcorrection.png')
plot_lib.plot_cand_network(ifgdates, bperp, bad_ifg_list, pngfile, bad_ifg_cands_list)

print('\nCorrections complete. Recomputing loops 3 and 4 of step 12 to find good network')
#%% 3rd loop closure check without bad ifgs wrt ref point
print('\n3rd loop closure check taking into account ref phase...', flush=True)
n_loop = np.shape(A3loop)[0]

bad_ifg_cand2 = []
good_ifg2 = []
loop_ph_rms_ifg2 = []

for i in range(n_loop):
    rms = loop_lib.loop_closure_3rd_wrapper(A3loop[ i ,:], ifgdates, ifgdir, length, width, ref_file)
    loop_ph_rms_ifg2.append(rms)

### List as good or bad candidate
    ### Find index of ifg
    ix_ifg12, ix_ifg23 = np.where(A3loop[i, :] == 1)[0]
    ix_ifg13 = np.where(A3loop[i, :] == -1)[0][0]
    ifgd12 = ifgdates[ix_ifg12]
    ifgd23 = ifgdates[ix_ifg23]
    ifgd13 = ifgdates[ix_ifg13]

    if np.isnan(loop_ph_rms_ifg2[i]): # Skipped
        loop_ph_rms_ifg2[i] = '--' ## Replace
    elif loop_ph_rms_ifg2[i] >= thresh: #Bad loop including bad ifg.
        bad_ifg_cand2.extend([ifgd12, ifgd23, ifgd13])
    else:
        good_ifg2.extend([ifgd12, ifgd23, ifgd13])

#%% Identify additional bad ifgs and output text
bad_ifg2 = loop_lib.identify_bad_ifg(bad_ifg_cand2, good_ifg2)

bad_ifgfile = os.path.join(loopdir, 'bad_ifg_loopref_corrected.txt')
with open(bad_ifgfile, 'w') as f:
    for i in bad_ifg2:
        print('{}'.format(i), file=f)


#%% Output all bad ifg list and identify remaining candidate of bad ifgs
### Merge bad ifg, bad_ifg2, noref_ifg
bad_ifg_all = list(set(bad_ifg2)) # Remove multiple
bad_ifg_all.sort()

ifgdates_good = list(set(ifgdates)-set(bad_ifg_all))
ifgdates_good.sort()

bad_ifgfile = os.path.join(infodir, '12bad_ifg_corrected.txt')
with open(bad_ifgfile, 'w') as f:
    for i in bad_ifg_all:
        print('{}'.format(i), file=f)

### Identify removed image and output file
imdates_good = tools_lib.ifgdates2imdates(ifgdates_good)
imdates_bad = list(set(imdates)-set(imdates_good))
imdates_bad.sort()

bad_imfile = os.path.join(infodir, '12removed_image_corrected.txt')
with open(bad_imfile, 'w') as f:
    for i in imdates_bad:
        print('{}'.format(i), file=f)

### Remaining candidate of bad ifg
bad_ifg_cand_res = list(set(bad_ifg_cand2)-set(bad_ifg_all))
bad_ifg_cand_res.sort()

bad_ifg_candfile = os.path.join(infodir, '12bad_ifg_cand_corrected.txt')
with open(bad_ifg_candfile, 'w') as f:
    for i in bad_ifg_cand_res:
        print('{}'.format(i), file=f)

pngfile = os.path.join(tsadir, 'network', 'network12cands_postcorrection2.png')
plot_lib.plot_cand_network(ifgdates, bperp, bad_ifg_all, pngfile, bad_ifg_cand_res)


#%% 4th loop to be used to calc n_loop_err and n_ifg_noloop
print('\n4th loop to compute statistics...', flush=True)

res = loop_lib.loop_closure_4th_wrapper(A3loop, length, width, ifgdates, ifgdir, bad_ifg_all, ref_file)
# breakpoint()
# ns_loop_err = np.sum(res[:, :, :,], axis=0)
ns_loop_err = res

#%% Output loop info, move bad_loop_png
loop_info_file = os.path.join(loopdir, 'loop_info_corrected.txt')
f = open(loop_info_file, 'w')
print('# loop_thre: {} rad'.format(thresh), file=f)
print('# ***: Removed w/ ref', file=f)

for i in range(n_loop):
    ### Find index of ifg
    ix_ifg12, ix_ifg23 = np.where(A3loop[i, :] == 1)[0]
    ix_ifg13 = np.where(A3loop[i, :] == -1)[0][0]
    ifgd12 = ifgdates[ix_ifg12]
    ifgd23 = ifgdates[ix_ifg23]
    ifgd13 = ifgdates[ix_ifg13]
    imd1 = ifgd12[:8]
    imd2 = ifgd23[:8]
    imd3 = ifgd23[-8:]

    badloopflag1 = ' '
    badloopflag2 = '  '
    if ifgd12 in bad_ifg2 or ifgd23 in bad_ifg2 or ifgd13 in bad_ifg_all:
        badloopflag2 = '***'

    if type(loop_ph_rms_ifg2[i]) == np.float32:
        str_loop_ph_rms_ifg2 = "{:.2f}".format(loop_ph_rms_ifg2[i])
    else: ## --
        str_loop_ph_rms_ifg2 = loop_ph_rms_ifg2[i]
        
    print('{0} {1} {2}  {3:5s} {4}'.format(imd1, imd2, imd3, str_loop_ph_rms_ifg2, badloopflag2), file=f)

f.close()


#%% Saving coh_avg, n_unw, and n_loop_err only for good ifgs
print('\nSaving coh_avg, n_unw, and n_loop_err...', flush=True)
### Calc coh avg and n_unw
coh_avg = np.zeros((length, width), dtype=np.float32)
n_coh = np.zeros((length, width), dtype=np.int16)
n_unw = np.zeros((length, width), dtype=np.int16)
for ifgd in ifgdates_good:
    ccfile = os.path.join(ifgdir, ifgd, ifgd+'.cc')
    if os.path.getsize(ccfile) == length*width:
        coh = io_lib.read_img(ccfile, length, width, np.uint8)
        coh = coh.astype(np.float32)/255
    else:
        coh = io_lib.read_img(ccfile, length, width)
        coh[np.isnan(coh)] = 0 # Fill nan with 0

    coh_avg += coh
    n_coh += (coh!=0)

    unwfile = os.path.join(ifgdir, ifgd, ifgd+'.unw')
    unw = io_lib.read_img(unwfile, length, width)

    unw[unw == 0] = np.nan # Fill 0 with nan
    n_unw += ~np.isnan(unw) # Summing number of unnan unw

coh_avg[n_coh==0] = np.nan
n_coh[n_coh==0] = 1 #to avoid zero division
coh_avg = coh_avg/n_coh

### Save files
n_unwfile = os.path.join(resultsdir, 'n_unw_corrected')
np.float32(n_unw).tofile(n_unwfile)

coh_avgfile = os.path.join(resultsdir, 'coh_avg_corrected')
coh_avg.tofile(coh_avgfile)

n_loop_errfile = os.path.join(resultsdir, 'n_loop_err_corrected')
np.float32(ns_loop_err).tofile(n_loop_errfile)

cmap_noise = 'viridis'
cmap_noise_r = 'viridis_r'
### Save png
title = 'Average coherence'
plot_lib.make_im_png(coh_avg, coh_avgfile+'.png', cmap_noise, title)
title = 'Number of used unw data'
plot_lib.make_im_png(n_unw, n_unwfile+'.png', cmap_noise, title, n_im)

title = 'Number of unclosed loops'
plot_lib.make_im_png(ns_loop_err, n_loop_errfile+'.png', cmap_noise_r, title)


#%% Plot network
## Read bperp data or dummy
bperp_file = os.path.join(ifgdir, 'baselines')
if os.path.exists(bperp_file):
    bperp = io_lib.read_bperp_file(bperp_file, imdates)
else: #dummy
    bperp = np.random.random(n_im).tolist()

### Network info
## Identify gaps
G = inv_lib.make_sb_matrix(ifgdates_good)
ixs_inc_gap = np.where(G.sum(axis=0)==0)[0]

## Connected network
ix1 = 0
connected_list = []
for ix2 in np.append(ixs_inc_gap, len(imdates_good)-1): #append for last image
    imd1 = imdates_good[ix1]
    imd2 = imdates_good[ix2]
    dyear = (dt.datetime.strptime(imd2, '%Y%m%d').toordinal() - dt.datetime.strptime(imd1, '%Y%m%d').toordinal())/365.25
    n_im_connect = ix2-ix1+1
    connected_list.append([imdates_good[ix1], imdates_good[ix2], dyear, n_im_connect])
    ix1 = ix2+1 # Next connection


#%% Caution about no_loop ifg, remaining large RMS loop and gap
### no_loop ifg

### Remaining candidates of bad ifgs
if len(bad_ifg_cand_res)!=0:
    print("\nThere are {} remaining candidates of bad ifgs but not identified.".format(len(bad_ifg_cand_res)), flush=True)
    print("Check 12bad_ifg_cand_ras and loop/bad_loop_cand_png.", flush=True)
#        for ifgd in bad_ifg_cand_res:
#            print('{}'.format(ifgd))

print('\n{0}/{1} ifgs are discarded from further processing.'.format(len(bad_ifg_all), n_ifg), flush=True)
for ifgd in bad_ifg_all:
    print('{}'.format(ifgd), flush=True)

### Gap
gap_infofile = os.path.join(infodir, '12network_gap_info_correction.txt')
with open(gap_infofile, 'w') as f:
    if ixs_inc_gap.size!=0:
        print("Gaps between:", file=f)
        print("\nGaps in network between:", flush=True)
        for ix in ixs_inc_gap:
            print("{} {}".format(imdates_good[ix], imdates_good[ix+1]), file=f)
            print("{} {}".format(imdates_good[ix], imdates_good[ix+1]), flush=True)

    print("\nConnected network (year, n_image):", file=f)
    print("\nConnected network (year, n_image):", flush=True)
    for list1 in connected_list:
        print("{0}-{1} ({2:.2f}, {3})".format(list1[0], list1[1], list1[2], list1[3]), file=f)
        print("{0}-{1} ({2:.2f}, {3})".format(list1[0], list1[1], list1[2], list1[3]), flush=True)

print('\nIf you want to change the bad ifgs to be discarded, re-run with different thresholds before next step.', flush=True)


    #%% Finish
elapsed_time = time.time()-start
hour = int(elapsed_time/3600)
minite = int(np.mod((elapsed_time/60),60))
sec = int(np.mod(elapsed_time,60))
print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

print('\nSuccessfully finished!!\n')
print('Output directory: {}\n'.format(tsadir))
