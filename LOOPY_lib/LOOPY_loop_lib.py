#!/usr/bin/env python3
"""
========
Overview
========
Python3 library of loop correction functions for LOOPY.
Requires the LiCSBAS libraries

=========
Changelog
=========
v1.0.0 20220615 Jack McGrath, Uni of Leeds
 - Original Implementation
v1.5.2 20210303 Yu Morishita, GSI
 - LiCSBAS_loop_lib
"""
import re
import os
import SCM
import time
import shutil
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import LiCSBAS_io_lib as io_lib
import LiCSBAS_plot_lib as plot_lib
import LiCSBAS_tools_lib as tools_lib

os.environ['QT_QPA_PLATFORM']='offscreen'

import warnings
import matplotlib as mpl
with warnings.catch_warnings(): ## To silence user warning
    warnings.simplefilter('ignore', UserWarning)
    mpl.use('module://matplotlib_inline.backend_inline')

cmap_wrap = tools_lib.get_cmap('SCM.romaO')
cmap_loop = tools_lib.get_cmap('SCM.vik')

#%% Used by lib14
def make_loop_matrix(ifgdates):
    """
    Make loop matrix (containing 1, -1, 0) from ifgdates.

    Inputs:
      ifgdates : Unwrapped phase vector at a point without nan (n_ifg)

    Returns:
      Aloop : Loop matrix with 1 for ifg12/ifg23 and -1 for ifg13
              (n_loop, n_ifg)

    """
    n_ifg = len(ifgdates)
    Aloop = []
    
    for ix_ifg12, ifgd12 in enumerate(ifgdates):
        primary12 = ifgd12[0:8]
        secondary12 = ifgd12[9:17]
        ifgdates23 = [ ifgd for ifgd in ifgdates if
                      ifgd.startswith(secondary12)] # all candidates of ifg23

        for ifgd23 in ifgdates23: # for each candidate of ifg23
            secondary23 = ifgd23[9:17]
            try:
                ## Search ifg13
                ix_ifg13 = ifgdates.index(primary12+'_'+secondary23)
            except: # no loop for this ifg23. Next.
                continue

            ## Loop found
            ix_ifg23 = ifgdates.index(ifgd23)

            Aline = [0]*n_ifg
            Aline[ix_ifg12] = 1
            Aline[ix_ifg23] = 1
            Aline[ix_ifg13] = -1
            Aloop.append(Aline)

    Aloop = np.array(Aloop)
    print('    A3Loops made')
    return Aloop

#%% CHANGED FOR LIB14
def read_unw_loop_ph(Aloop1, ifgdates, ifgdir, length, width, bad_ifg=[], mask=False):
    ### Find index of ifg
    ix_ifg12, ix_ifg23 = np.where(Aloop1 == 1)[0]
    ix_ifg13 = np.where(Aloop1 == -1)[0][0]
    ifgd12 = ifgdates[ix_ifg12]
    ifgd23 = ifgdates[ix_ifg23]
    ifgd13 = ifgdates[ix_ifg13]
    
    if mask:
        fileend = ['.unw_mask','.unw_mask','.unw_mask']
        fileend[bad_ifg-1] = '.unw'
    else:
        fileend = ['.unw','.unw','.unw']
    ### Read unw data
    unw12file = os.path.join(ifgdir, ifgd12, ifgd12+fileend[0])
    try:
        unw12 = io_lib.read_img(unw12file, length, width)
    except:
        unw12 = io_lib.read_img(unw12file[:-5], length, width)
    unw12[unw12 == 0] = np.nan # Fill 0 with nan
    unw23file = os.path.join(ifgdir, ifgd23, ifgd23+fileend[1])
    try:
        unw23 = io_lib.read_img(unw23file, length, width)
    except:
        unw23 = io_lib.read_img(unw23file[:-5], length, width)
    unw23[unw23 == 0] = np.nan # Fill 0 with nan
    unw13file = os.path.join(ifgdir, ifgd13, ifgd13+fileend[2])
    try:
        unw13 = io_lib.read_img(unw13file, length, width)
    except:
        unw13 = io_lib.read_img(unw13file[:-5], length, width)
    unw13[unw13 == 0] = np.nan # Fill 0 with n

    return unw12, unw23, unw13, ifgd12, ifgd23, ifgd13


#%%
def identify_bad_ifg(bad_ifg_cand, good_ifg):
    ### Identify bad ifgs and output text
    good_ifg = list(set(good_ifg))
    good_ifg.sort()
    bad_ifg_cand = list(set(bad_ifg_cand))
    bad_ifg_cand.sort()

    bad_ifg = list(set(bad_ifg_cand)-set(good_ifg)) # difference
    bad_ifg.sort()

    return bad_ifg


#%%
def make_loop_png(unw12, unw23, unw13, loop_ph, png, titles4, cycle):
    # cmap_wrap = tools_lib.get_cmap('SCM.romaO')
    # cmap_loop = tools_lib.get_cmap('SCM.vik')

    ### Settings
    plt.rcParams['axes.titlesize'] = 10
    data = [unw12, unw23, unw13]

    length, width = unw12.shape
    if length > width:
        figsize_y = 10
        figsize_x = int((figsize_y-1)*width/length)
        if figsize_x < 5: figsize_x = 5
    else:
        figsize_x = 10
        figsize_y = int(figsize_x*length/width+1)
        if figsize_y < 3: figsize_y = 3

    ### Plot
    fig = plt.figure(figsize = (figsize_x, figsize_y))

    ## 3 ifgs
    for i in range(3):
        data_wrapped = np.angle(np.exp(1j*(data[i]/cycle))*cycle)
        ax = fig.add_subplot(2, 2, i+1) #index start from 1
        im = ax.imshow(data_wrapped, vmin=-np.pi, vmax=+np.pi, cmap=cmap_wrap,
                  interpolation='nearest')
        ax.set_title('{}'.format(titles4[i]))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        cax = plt.colorbar(im)
        cax.set_ticks([])

    ## loop phase
    ax = fig.add_subplot(2, 2, 4) #index start from 1
    im = ax.imshow(loop_ph, vmin=-np.pi, vmax=+np.pi, cmap=cmap_loop,
              interpolation='nearest')
    ax.set_title('{}'.format(titles4[3]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    cax = plt.colorbar(im)

    plt.tight_layout()
    plt.savefig(png)
    plt.close()
    
#####%%
#%% Functions for Loopy_lib14
#####%%

#%% Used by lib14
def create_loop_dict(A3loop, ifgdates):
    # IFG: Loops
    ifg_dict = {}
    for count,ifg in enumerate(ifgdates):
        ifg_dict[ifg] = list(np.where(A3loop[:,count] != 0)[0])
        
    # Loop: IFGs
    loop_dict = {}
    for i in range(np.shape(A3loop)[0]):
        loop_dict[i] = np.where(A3loop[i,:] != 0)[0].tolist()
    
    print('    Loop Dictionaries made')
    return ifg_dict, loop_dict

#%% Fuction to calculate which of the 3 ifgs in the loop is bad
def calc_bad_ifg_position_single(bad_ifg, Aloop, ifgdates):
    loop_dates = np.where(Aloop != 0)[0]

    for ix, ifg in enumerate(loop_dates):
        if bad_ifg == ifgdates[ifg]:
            break
    if ix == 0:
        ifg_ix = 1
    elif ix ==1:
        ifg_ix = 3
    else:
        ifg_ix = 2
        
    return ifg_ix

#%% Function to correct the bad interferogram using loop closure
def get_corr(loop, bad_ix, thresh, ifgdates, ifgdir, length, width, ref_file = [], multi_prime = True):

    unw12, unw23, unw13, ifgd12, ifgd23, ifgd13 = read_unw_loop_ph(loop, ifgdates, ifgdir, length, width, bad_ifg=bad_ix, mask=True)

    if ref_file:
        refx1, refx2, refy1, refy2 = find_ref_area(ref_file)
        
        ## Skip if no data in ref area in any unw. It is bad data.
        ref_unw12 = np.nanmean(unw12[refy1:refy2, refx1:refx2])
        ref_unw23 = np.nanmean(unw23[refy1:refy2, refx1:refx2])
        ref_unw13 = np.nanmean(unw13[refy1:refy2, refx1:refx2])
    else:
        ref_unw12 = ref_unw23 = ref_unw13 = 0

    ## Calculate loop phase taking into account ref phase
    loop_ph = unw12+unw23-unw13-(ref_unw12+ref_unw23-ref_unw13)
    
    if multi_prime:
        bias = np.nanmedian(loop_ph)
        loop_ph = loop_ph - bias # unbias inconsistent fraction phase

    n_pi_correction = (loop_ph / (2*np.pi)).round() # Modulo division would mean that a pixel out by 6 rads wouldn't be corrected, for example
    loop_correction = n_pi_correction * 2 * np.pi
    
    if bad_ix == 3:
        loop_correction = loop_correction * -1

    return loop_correction

#%% Correct Bad IFG
def correct_bad_ifg(bad_ifg_name,corr,ifgdir, length, width, loop, ifgdates, loopdir, ref_file = [], cycle=3):
    ifgfile = os.path.join(ifgdir, bad_ifg_name, bad_ifg_name + '.unw')
    # print(ifgfile)
    # breakpoint()
    ifg = io_lib.read_img(ifgfile, length, width)
    ifg[ifg == 0] = np.nan # Fill 0 with nan
    corr[np.isnan(corr)] = 0
    corr_ifg = ifg - corr
    corr[np.isnan(ifg)] = np.nan
    #Center 0 for the correction
    corr_max = round(np.nanmax(abs(corr)) / (2*np.pi))
    corr_min = round(-np.nanmax(abs(corr)) / (2*np.pi))
    corr[0,0]= corr_max
    corr[0,1]= corr_min
    # cmap_wrap = tools_lib.get_cmap('SCM.romaO')
    ifg_max = max([np.nanmax(ifg), np.nanmax(corr_ifg)])
    ifg_min = min([np.nanmin(ifg), np.nanmin(corr_ifg)])
    title = '{} ({}pi/cycle)'.format(bad_ifg_name, cycle*2)
    title3 = ['Original {} (RMS: {:.2f})'.format(bad_ifg_name, np.nanstd(ifg)),'Correction (n*2pi)','Corrected {} (RMS: {:.2f})'.format(bad_ifg_name, np.nanstd(corr_ifg))]
    data3=[ifg,corr/(2*np.pi),corr_ifg]
    pngfile = os.path.join(ifgdir, 'Corr_dir', bad_ifg_name + '_compare.unw.png')
    plot_lib.make_3im_png_corr(data3, pngfile, cmap_wrap, title3, vmin=[ifg_min,corr_min,ifg_min], vmax=[ifg_max,corr_max,ifg_max], cbar=True)
    # Backup original unw file and loop png
    shutil.move(os.path.join(ifgdir, bad_ifg_name, bad_ifg_name + '.unw'),os.path.join(ifgdir, bad_ifg_name, bad_ifg_name + '_uncorr.unw'))
    shutil.move(os.path.join(ifgdir, bad_ifg_name, bad_ifg_name + '.unw.png'),os.path.join(ifgdir, bad_ifg_name, bad_ifg_name + '_uncorr.unw.png'))
    plot_lib.make_im_png(np.angle(np.exp(1j*corr_ifg/3)*3), os.path.join(ifgdir, bad_ifg_name, bad_ifg_name + '.unw.png'), SCM.romaO, title, -np.pi, np.pi, cbar=False)
    
    # Make new unw file from corrected data and new loop png
    corr_ifg.tofile(os.path.join(ifgdir, bad_ifg_name, bad_ifg_name + '.unw'))
    
    corr[np.isnan(ifg)] = np.nan
    max_corr = np.nanmax(abs(corr))
    plot_lib.make_im_png(corr/(2*np.pi),os.path.join(ifgdir, bad_ifg_name, bad_ifg_name + '_npi.unw.png'), SCM.vik, bad_ifg_name + ' (2Pi Correction)', vmin = -max_corr/(2*np.pi), vmax = max_corr/(2*np.pi))
    

    for l in range(loop.shape[0]):
        
        re_loop(loop[l], ifgdates, ifgdir, length, width, loopdir, ref_file, cycle=cycle)
        
#%% Function to re-run loops where interferograms have been corrected
def re_loop(loop, ifgdates, ifgdir, length, width, loopdir, ref_file=[], cycle = 3, multi_prime = True):
    
    unw12, unw23, unw13, ifgd12, ifgd23, ifgd13 = read_unw_loop_ph(loop, ifgdates, ifgdir, length, width)
    
    imd1 = ifgd12[:8]
    imd2 = ifgd23[:8]
    imd3 = ifgd23[-8:]
    
    
    if ref_file:
        refx1, refx2, refy1, refy2 = find_ref_area(ref_file)

        ## Skip if no data in ref area in any unw. It is bad data.
        ref_unw12 = np.nanmean(unw12[refy1:refy2, refx1:refx2])
        ref_unw23 = np.nanmean(unw23[refy1:refy2, refx1:refx2])
        ref_unw13 = np.nanmean(unw13[refy1:refy2, refx1:refx2])

    else:
        ref_unw12 = 0
        ref_unw23 = 0
        ref_unw13 = 0

 ## Calculate loop phase taking into account ref phase
    loop_ph = unw12+unw23-unw13-(ref_unw12+ref_unw23-ref_unw13)
    
    if multi_prime:
        bias = np.nanmedian(loop_ph)
        loop_ph = loop_ph - bias # unbias inconsistent fraction phase
    
    rms = np.sqrt(np.nanmean(loop_ph**2))
    
    titles4 = ['{} ({}*2pi/cycle)'.format(imd1+'_'+imd2, cycle),
               '{} ({}*2pi/cycle)'.format(imd2+'_'+imd3, cycle),
               '{} ({}*2pi/cycle)'.format(imd1+'_'+imd3, cycle),]
    if multi_prime:
        titles4.append('Loop (STD={:.2f}rad, bias={:.2f}rad)'.format(rms, bias))
    else:
        titles4.append('Loop phase (RMS={:.2f}rad)'.format(rms))
    
    loop_id = imd1+'_'+imd2+'_'+imd3+'_loop'
    png = os.path.join(loopdir,'bad_loop_png',loop_id)
    uncorr_png = png + '_V0'

    if os.path.exists(png + '.png') or os.path.exists(uncorr_png + '.png'):
        if os.path.exists(png + '.png'):
            shutil.move(png + '.png',uncorr_png + '.png')
            
        while os.path.exists(uncorr_png + '.png'):
            uncorr_png = uncorr_png[:-1] + str(int(uncorr_png[-1])+1)

    elif os.path.exists(os.path.join(loopdir, 'bad_loop_cand_png', loop_id + '.png')) or os.path.exists(os.path.join(loopdir, 'bad_loop_cand_png', loop_id + '_V0.png')):
        png = os.path.join(loopdir,'bad_loop_cand_png', loop_id)
        uncorr_png = png + '_V0'

        if os.path.exists(png + '.png'):
            shutil.move(png + '.png',uncorr_png + '.png')
            
        while os.path.exists(uncorr_png + '.png'):
            uncorr_png = uncorr_png[:-1] + str(int(uncorr_png[-1])+1)
            
    elif os.path.exists(os.path.join(loopdir, 'good_loop_png', loop_id + '.png')) or os.path.exists(os.path.join(loopdir, 'good_loop_png', loop_id + '_V0.png')):
        png = os.path.join(loopdir,'good_loop_png', loop_id)
        uncorr_png = png + '_V0'

        if os.path.exists(png + '.png'):
            shutil.move(png + '.png',uncorr_png + '.png')
            
        while os.path.exists(uncorr_png + '.png'):
            uncorr_png = uncorr_png[:-1] + str(int(uncorr_png[-1])+1)
    
    else:
        png = os.path.join(loopdir,'loop_pngs',loop_id + '_V1')
        
        while os.path.exists(png + '.png'):
            png = png[:-1] + str(int(png[-1])+1)
            
        uncorr_png = png
    
    make_loop_png(unw12, unw23, unw13, loop_ph, uncorr_png, titles4, cycle)
    
    # print('Loop:',imd1+'_'+imd2+'_'+imd3, 'New RMS:',rms)















#####%%
#%% Loopy_lib10 Functions
#####%%
#%% Identify unique loops in loop dict, inc check dict (eg unique bad loops, including candidate)
def identify_unique_loops(ifg_ix, loop_dict, check_dict, loop_info):
    u_loop_dict = {}
    for i in ifg_ix:
        u_loop_dict[i] = []

    unique_loops = pd.DataFrame(ifg_ix,columns = ['IFG_IX'])
    unique_loops['No. Unique'] = 0
    unique_loops['Loop Total'] = 0
    unique_loops['Mean RMS'] = np.inf
    for count, item in enumerate(loop_dict.items()):
        # Add mean RMS of loops to DF
        for ifg in item[1]:
            i = 1
            while not loop_info[int(item[0])].split()[-i][-1].isdigit():
                i += 1
            new_rms = float(loop_info[int(item[0])].split()[-i])
            ix = ifg_ix.index(ifg)
            L_tot = unique_loops.loc[ix, 'Loop Total']
            L_mean = list(np.ones(L_tot) * unique_loops.loc[ix, 'Mean RMS'])
            L_mean.append(new_rms)
            unique_loops.loc[ix, 'Mean RMS'] = round(np.mean(L_mean),2)
            unique_loops.loc[ix, 'Loop Total'] += 1

        # Log unique Loop
        if len(item[1]) == 1:
            # Check to ensure our unique loop is not in the candidate dictionary
            if item[0] not in check_dict.keys():
                unique_loops.loc[ifg_ix.index(item[1][0]),'No. Unique'] += 1
                u_loop_dict[item[1][0]].append(int(item[0]))

    return unique_loops, u_loop_dict

#%% Identify loops that contain only good interferograms
def identify_good_loops(ifg_ix, loop_dict, check_dict, loop_info):
    #Create Dictionary of only good loops
    good_loop_dict = {}
    for i in ifg_ix:
        good_loop_dict[i] = []

    good_loops = pd.DataFrame(ifg_ix,columns = ['IFG_IX'])
    good_loops['No. Good'] = 0
    good_loops['Loop Total'] = 0
    good_loops['Mean RMS'] = np.inf
    for count, item in enumerate(loop_dict.items()):
        # Add mean RMS of loops to DF
        for ifg in item[1]:
            i = 1
            while not loop_info[int(item[0])].split()[-i][-1].isdigit():
                i += 1
            new_rms = float(loop_info[int(item[0])].split()[-i])
            ix = ifg_ix.index(ifg)
            L_tot = good_loops.loc[ix, 'Loop Total']
            L_mean = list(np.ones(L_tot) * good_loops.loc[ix, 'Mean RMS'])
            L_mean.append(new_rms)
            good_loops.loc[ix, 'Mean RMS'] = round(np.mean(L_mean),2)
            good_loops.loc[ix, 'Loop Total'] += 1

        # Log good Loop
        if len(item[1]) == 1:
            # Check to ensure our good loop is not it the candidate dictionary
            if item[0] not in check_dict.keys():
                good_loops.loc[ifg_ix.index(item[1][0]),'No. Good'] += 1
                good_loop_dict[item[1][0]].append(int(item[0]))

    return good_loops, good_loop_dict

#%% Select which IFG to correct
def select_ifg(unique_df):
    # Remove non-unique
    unique_df.drop(unique_df[unique_df['No. Unique'] == 0].index, inplace = True)

    # Select most unique
    unique_df = unique_df[unique_df['No. Unique'] == unique_df['No. Unique'].max()]

    # Select Lowest RMS
    unique_df = unique_df[unique_df['Mean RMS'] == unique_df['Mean RMS'].min()]

    # Select least loops
    unique_df = unique_df[unique_df['Loop Total'] == unique_df['Loop Total'].min()]

    # Get index (Selecting first just in case there are more than 1 left)
    ifg = unique_df.loc[unique_df.index[0], 'IFG_IX']

    return ifg

#%%
def find_ref_area(ref_file):
    with open(ref_file, "r") as f:
        refarea = f.read().split()[0]  #str, x1/x2/y1/y2
    refx1, refx2, refy1, refy2 = [int(s) for s in re.split('[:/]', refarea)]
    
    return refx1, refx2, refy1, refy2

#%% Function to check and correct loop
def check_loop_threshold(loop, thresh, ifgdates, ifgdir, length, width, multi_prime = False, ref_file = []):
    unw12, unw23, unw13, ifgd12, ifgd23, ifgd13 = read_unw_loop_ph(loop, ifgdates, ifgdir, length, width)

    if ref_file:
        refx1, refx2, refy1, refy2 = find_ref_area(ref_file)
        ## Skip if no data in ref area in any unw. It is bad data.
        ref_unw12 = np.nanmean(unw12[refy1:refy2, refx1:refx2])
        ref_unw23 = np.nanmean(unw23[refy1:refy2, refx1:refx2])
        ref_unw13 = np.nanmean(unw13[refy1:refy2, refx1:refx2])

        ## Calculate loop phase taking into account ref phase
        loop_ph = unw12+unw23-unw13-(ref_unw12+ref_unw23-ref_unw13)
    else:
        ## Calculate loop phase and check n bias (2pi*n)
        loop_ph = unw12+unw23-unw13

    if multi_prime:
        bias = np.nanmedian(loop_ph)
        loop_ph = loop_ph - bias # unbias inconsistent fraction phase
    rms = round(np.sqrt(np.nanmean(loop_ph**2)),2)
    # breakpoint()
    bad_loop = (rms > thresh)

    return bad_loop, rms

#%% Fuction to calculate which of the 3 ifgs in the loop is bad
def calc_bad_ifg_position_single3(bad_ifg, Aloop, ifgdates):
    loop_dates = np.where(Aloop != 0)[0]

    for ix, ifg in enumerate(loop_dates):
        if bad_ifg == ifgdates[ifg]:
            break
    if ix == 0:
        ifg_ix = 1
    elif ix ==1:
        ifg_ix = 3
    else:
        ifg_ix = 2
        
    return ifg_ix

#%% Function to correct the bad interferogram using loop closure
def get_corr3(loop, bad_ix, thresh, ifgdates, ifgdir, length, width, ref_file = [], multi_prime = True):

    unw12, unw23, unw13, ifgd12, ifgd23, ifgd13 = read_unw_loop_ph(loop, ifgdates, ifgdir, length, width)
    # if bad_ix == 3:
        # breakpoint()

    if ref_file:
        refx1, refx2, refy1, refy2 = find_ref_area(ref_file)
        
        ## Skip if no data in ref area in any unw. It is bad data.
        ref_unw12 = np.nanmean(unw12[refy1:refy2, refx1:refx2])
        ref_unw23 = np.nanmean(unw23[refy1:refy2, refx1:refx2])
        ref_unw13 = np.nanmean(unw13[refy1:refy2, refx1:refx2])
    else:
        ref_unw12 = ref_unw23 = ref_unw13 = 0

    ## Calculate loop phase taking into account ref phase
    loop_ph = unw12+unw23-unw13-(ref_unw12+ref_unw23-ref_unw13)
    
    if multi_prime:
        bias = np.nanmedian(loop_ph)
        loop_ph = loop_ph - bias # unbias inconsistent fraction phase

    loop_error = loop_ph * (np.absolute(loop_ph) > (thresh))

    n_pi_correction = (loop_error / (2*np.pi)).round() # Modulo division would mean that a pixel out by 6 rads wouldn't be corrected, for example
    loop_correction = n_pi_correction * 2 * np.pi
    
    if bad_ix == 3:
        loop_correction = loop_correction * -1

    return loop_correction

#%%
def read_loop_masks(Aloop1, ifgdates, ifgdir, length, width, bad_ifg=[], mask=False):
    ### Find index of ifg
    ix_ifg12, ix_ifg23 = np.where(Aloop1 == 1)[0]
    ix_ifg13 = np.where(Aloop1 == -1)[0][0]
    ifgd12 = ifgdates[ix_ifg12]
    ifgd23 = ifgdates[ix_ifg23]
    ifgd13 = ifgdates[ix_ifg13]
    

    ### Read unw data
    mask12file = os.path.join(ifgdir, ifgd12, ifgd12+'.mask')
    if os.path.exists(mask12file):
        mask12dat = io_lib.read_img(mask12file, length, width)
        mask12=np.zeros([length,width])
        mask12[mask12dat < 0] = 1 # Fill 0 with nan
        # mask12[mask12 >= 0] = 0 # Fill 0 with nan
    else:
        mask12 = np.zeros([length,width]).astype('float32')
        mask12.tofile(mask12file)
        print('\t\t',mask12file,'not found')
    
    if not os.path.exists(os.path.join(mask12file+'.png')):
        plot_lib.make_im_png(mask12,os.path.join(mask12file + '.png'), SCM.vik, 'Correction Mask', vmin = -np.nanmax(abs(mask12)), vmax = np.nanmax(abs(mask12)))

    mask23file = os.path.join(ifgdir, ifgd23, ifgd23+'.mask')
    if os.path.exists(mask23file):
        mask23dat = io_lib.read_img(mask23file, length, width)
        mask23=np.zeros([length,width])
        mask23[mask23dat < 0] = 1 # Fill 0 with nan
        # mask23[mask23 >= 0] = 0 # Fill 0 with nan
    else:
        mask23 = np.zeros([length,width]).astype('float32')
        mask23.tofile(mask23file)
        print('\t\t',mask23file,'not found')
    
    if not os.path.exists(os.path.join(mask23file+'.png')):
        plot_lib.make_im_png(mask23,os.path.join(mask23file + '.png'), SCM.vik, 'Correction Mask', vmin = -np.nanmax(abs(mask23)), vmax = np.nanmax(abs(mask23)))

    mask13file = os.path.join(ifgdir, ifgd13, ifgd13+'.mask')
    if os.path.exists(mask13file):
        mask13dat = io_lib.read_img(mask13file, length, width)
        mask13=np.zeros([length,width])
        mask13[mask13dat < 0] = 1 # Fill 0 with nan
        # mask13[mask13 >= 0] = 0 # Fill 0 with nan
    else:
        mask13 = np.zeros([length,width]).astype('float32')
        mask13.tofile(mask13file)
        print('\t\t',mask13file,'not found')
    
    if not os.path.exists(os.path.join(mask13file+'.png')):
        plot_lib.make_im_png(mask13,os.path.join(mask13file + '.png'), SCM.vik, 'Correction Mask', vmin = -np.nanmax(abs(mask13)), vmax = np.nanmax(abs(mask13)))

    return mask12, mask23, mask13

#%% Function to correct the good interferogram using loop closure and mask
def get_corr_mask(loop, bad_ix, thresh, ifgdates, ifgdir, length, width, ref_file = [], multi_prime = True):

    unw12, unw23, unw13, ifgd12, ifgd23, ifgd13 = read_unw_loop_ph(loop, ifgdates, ifgdir, length, width)
    mask12, mask23, mask13 = read_loop_masks(loop, ifgdates, ifgdir, length, width)
    
    #Apply mask to the IFGs that we deem to have been good for the purpose of this correction
    if bad_ix != 1:
        unw12[mask12==1] = np.nan
    if bad_ix != 2:
        unw23[mask23==1] = np.nan
    if bad_ix != 3:
        unw13[mask13==1] = np.nan
    
    if ref_file:
        refx1, refx2, refy1, refy2 = find_ref_area(ref_file)
        
        ## Skip if no data in ref area in any unw. It is bad data.
        ref_unw12 = np.nanmean(unw12[refy1:refy2, refx1:refx2])
        ref_unw23 = np.nanmean(unw23[refy1:refy2, refx1:refx2])
        ref_unw13 = np.nanmean(unw13[refy1:refy2, refx1:refx2])
    else:
        ref_unw12 = ref_unw23 = ref_unw13 = 0

    ## Calculate loop phase taking into account ref phase
    loop_ph = unw12+unw23-unw13-(ref_unw12+ref_unw23-ref_unw13)
    
    if multi_prime:
        bias = np.nanmedian(loop_ph)
        loop_ph = loop_ph - bias # unbias inconsistent fraction phase

    loop_error = loop_ph * (np.absolute(loop_ph) > (thresh))

    n_pi_correction = (loop_error / (2*np.pi)).round() # Modulo division would mean that a pixel out by 6 rads wouldn't be corrected, for example
    loop_correction = n_pi_correction * 2 * np.pi
    
    if bad_ix == 3:
        loop_correction = loop_correction * -1

    return loop_correction


    
#%%
def make_corr_loop_png(unw12, unw23, unw13, loop_ph, png, titles4, cycle):
    cmap_loop = tools_lib.get_cmap('SCM.vik')

    ### Settings
    plt.rcParams['axes.titlesize'] = 10
    data = [unw12, unw23, unw13]

    length, width = unw12.shape
    if length > width:
        figsize_y = 10
        figsize_x = int((figsize_y-1)*width/length)
        if figsize_x < 5: figsize_x = 5
    else:
        figsize_x = 10
        figsize_y = int(figsize_x*length/width+1)
        if figsize_y < 3: figsize_y = 3

    ### Plot
    fig = plt.figure(figsize = (figsize_x, figsize_y))

    ## 3 ifgs
    for i in range(3):
        data_wrapped = data[i]
        ax = fig.add_subplot(2, 2, i+1) #index start from 1
        im = ax.imshow(data_wrapped, vmin=-np.pi, vmax=+np.pi, cmap=cmap_loop,
                  interpolation='nearest')
        ax.set_title('{}'.format(titles4[i]))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        cax = plt.colorbar(im)
        cax.set_ticks([])

    ## loop phase
    ax = fig.add_subplot(2, 2, 4) #index start from 1
    im = ax.imshow(loop_ph, vmin=-np.pi, vmax=+np.pi, cmap=cmap_loop,
              interpolation='nearest')
    ax.set_title('{}'.format(titles4[3]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    cax = plt.colorbar(im)

    plt.tight_layout()
    plt.savefig(png)
    plt.close()

#%% 
def create_3loop_dictionaries(A3loop, bad_ifg_ix, bad_ifg_cands_ix, good_ifg_ix=[]):
    #IFG:Bad_Loops
    bad_ifg3_dict = {}
    for ifg in bad_ifg_ix:
        bad_ifg3_dict[str(ifg)] = list(np.where(A3loop[:,ifg] != 0)[0])
        
    bad_ifg3_cands_dict = {}
    for ifg in bad_ifg_cands_ix:
        bad_ifg3_cands_dict[str(ifg)] = list(np.where(A3loop[:,ifg] != 0)[0])

    # Get list of all loops
    bad_loop3_ix = list(np.unique([item for sublist in bad_ifg3_dict.values() for item in sublist]))
    bad_loop3_cands_ix = list(np.unique([item for sublist in bad_ifg3_cands_dict.values() for item in sublist]))
    # Ensure Loops don't appear in multiple dictionaries
    bad_loop3_cands_ix = list(set(bad_loop3_cands_ix)-set(bad_loop3_ix))

    #Bad_Loops:IFG
    bad_loop3_dict = {}
    for i in bad_loop3_ix:
        bad_loop3_dict[str(i)] = []
        for count, ifgs in enumerate(bad_ifg3_dict.values()):
            if i in ifgs:
                bad_loop3_dict[str(i)].append(bad_ifg_ix[count])

    bad_loop3_cands_dict = {}
    for i in bad_loop3_cands_ix:
        bad_loop3_cands_dict[str(i)] = []
        for count, ifgs in enumerate(bad_ifg3_cands_dict.values()):
            if i in ifgs:
                bad_loop3_cands_dict[str(i)].append(bad_ifg_cands_ix[count])
                
    if len(good_ifg_ix) != 0:
        good_ifg3_dict = {}
        for ifg in good_ifg_ix:
            good_ifg3_dict[str(ifg)] = list(np.where(A3loop[:,ifg] != 0)[0])
        good_loop3_ix = list(np.unique([item for sublist in good_ifg3_dict.values() for item in sublist]))
        good_loop3_ix = list(set(good_loop3_ix)-set(bad_loop3_cands_ix)-set(bad_loop3_ix))
        
        good_loop3_dict = {}
        for i in good_loop3_ix:
            good_loop3_dict[str(i)] = []
            for count, ifgs in enumerate(good_ifg3_dict.values()):
                if i in ifgs:
                    good_loop3_dict[str(i)].append(good_ifg_ix[count])
                    
        print('    3Loop Dictionaries made')
        return bad_ifg3_dict, bad_ifg3_cands_dict, good_ifg3_dict, bad_loop3_dict, bad_loop3_cands_dict, good_loop3_dict
                
    else:
        print('    3Loop Dictionaries made')
        return bad_ifg3_dict, bad_ifg3_cands_dict, bad_loop3_dict, bad_loop3_cands_dict

#%%
def remove_fixed_loops(check_loops, bad_ifg_number, bad_ifg_dict, bad_loop_dict, bad_loop_cands_dict, unique_DF):
    edited_loop = []
    for loop in check_loops:
        bad_ifg_dict[str(bad_ifg_number)].remove(loop)
        bad_loop_dict[str(loop)].remove(bad_ifg_number)
        edited_loop = edited_loop + bad_loop_dict[str(loop)] # Variable of all interferograms that may now be in a new unique list
    edited_loop = list(set(edited_loop))
    for ifg in edited_loop:
        df_ix = unique_DF.IFG_IX[unique_DF.IFG_IX == ifg].index.tolist()[0]
        unique_DF.loc[df_ix,'No. Unique'] = 0
        for loop in bad_ifg_dict[str(ifg)]:
            if len(bad_loop_dict[str(loop)]) == 1:
                if loop not in bad_loop_cands_dict.keys(): # Check to ensure our unique loop is not it the candidate dictionary
                    unique_DF.loc[df_ix,'No. Unique'] += 1
                    
    return bad_ifg_dict, bad_loop_dict, unique_DF

#%%
def get_mode(corr_all):
    # Mask all Nan's and all 0's to speed up scipy
    mode_mask = np.all(corr_all==0, axis=2) + np.all(np.isnan(corr_all), axis=2)
    all_tmp = np.copy(corr_all)
    all_tmp[mode_mask,:] = np.nan
    corr,corrcount = stats.mode(all_tmp, axis=2, nan_policy='omit')
    corr = np.array(corr[:,:,0]).astype('float32')
    corrcount = np.array(corrcount[:,:,0])
    corrcount[np.all(corr_all==0, axis=2)] = np.shape(corr_all)[2]
    corrcount[corrcount==0] = np.nan
    
    return corr, corrcount


#%% Script for the event of the modal correction is exactly half the number of half, to see which half is the correct.
def half_check(nloops,corrcount, corr_all):
    halfn = nloops/2 # How many loops is half loops?
    
    # Get all pixels where the corrcount = halfn
    halfcount = corrcount == halfn
    corr_half = np.copy(corr_all)
    
    #Set all pixels where corrcount != halfn to nan
    for i in range(nloops):
        corr_half[halfcount==0,i] = np.nan
    
    # Find the mean error in these areas
    err_mean = np.full(nloops,np.nan)
    
    for l in range(nloops):
        err_mean[l] = np.nanmean(corr_all[:,:,l])
    
    errors = pd.DataFrame(np.linspace(0,nloops-1,nloops).astype('int'), columns=['Loop'])
    errors['Mean'] = err_mean
    errors = errors.sort_values(by=['Mean'],ignore_index=True)
    
    # Find similar corrections
    similarity =[]
    for i in range(int(halfn+1)):
        similarity.append(errors.loc[i+halfn-1,'Mean'] - errors.loc[i,'Mean'])
    sim_ix = similarity.index(min(similarity))
    sim_loops = errors.loc[sim_ix:sim_ix+halfn-1,'Loop'].tolist()
    
    return sim_loops

  
#%%
def create_masks(ifg, nloops, sim_loops, maskloops, masks, corr_all, corr, corrcount, bad_data, good_data, good_loop3_dict, ifgdir, ifgdates):
    # Make masks for the loops
    for count, loop in enumerate(maskloops):
        masks[bad_data,count] -= 1 # Lose a point for no majority solution
        good_corr = corr_all[:,:,count] == corr
        good_corr = good_corr.astype('int') + good_data.astype('int')
        masks[good_corr==2,count] += 2 # Earn two points for being majority solution
        masks[good_corr!=2,count] -= 2 # Lose two points for not being majority solution
        
        if count in sim_loops:
            masks[corrcount == nloops/2, count] += 4
        
        # Write masks to file (always overwrite previous masks)
        # breakpoint()
        for i in good_loop3_dict[str(loop)]:
            if i != ifg:
                maskfile = os.path.join(ifgdir, ifgdates[i], ifgdates[i]+'.mask')
                mask = masks[:,:,count]
                mask.tofile(maskfile)
                
#%% Decide if Loop is Corrected, Good, Cand, or Bad
def classify_loops(corr_loop_df, ix):
    if corr_loop_df.loc[ix,'Bad'] != 0:
        corr_loop_df.loc[ix,'Class'] = 'Bad'
    elif corr_loop_df.loc[ix,'Cand'] != 0:
        corr_loop_df.loc[ix,'Class'] = 'Cand'
    elif corr_loop_df.loc[ix,'Good'] != 0:
        corr_loop_df.loc[ix,'Class'] = 'Good'
    else:
        corr_loop_df.loc[ix,'Class'] = 'Corr'
    
    return corr_loop_df

#%% Update corr_df once IFGs have been corrected
def update_corr_df(corr_loop_df, ifg, ifgname, ifg_dict, good_ifg_list, bad_ifg_cands_list, bad_ifg_list):

    if ifgname in good_ifg_list:
        for i in ifg_dict[ifg]:
            corr_loop_df.loc[i, 'Good'] -= 1
    if ifgname in bad_ifg_cands_list:
        for i in ifg_dict[ifg]:
            corr_loop_df.loc[i, 'Cand'] -= 1
    if ifgname in bad_ifg_list:
        for i in ifg_dict[ifg]:
            corr_loop_df.loc[i, 'Bad'] -= 1
    
    for l in ifg_dict[ifg]:
        corr_loop_df.loc[l,'Corr'] += 1
        corr_loop_df = classify_loops(corr_loop_df,l)
            
    return corr_loop_df

#%% Update loop dictionaries to account for corrected IFGs
def update_loop_dicts(nloops, corr_loop_df, good_loop_dict, bad_loop_cands_dict, bad_loop_dict):
    
    corr = corr_loop_df.loc[corr_loop_df['Class'] == 'Corr', 'Loop'].tolist()
    good = corr_loop_df.loc[corr_loop_df['Class'] == 'Good', 'Loop'].tolist()
    good = good + corr
    cand = corr_loop_df.loc[corr_loop_df['Class'] == 'Cand', 'Loop'].tolist()
    bad = corr_loop_df.loc[corr_loop_df['Class'] == 'Bad', 'Loop'].tolist()
    
    new_good_dict = {}
    new_cand_dict = {}
    new_bad_dict = {}
    
    for loop in range(nloops):
        if str(loop) in good_loop_dict.keys():
            vals = good_loop_dict[str(loop)]
        elif str(loop) in bad_loop_cands_dict.keys():
            vals = bad_loop_cands_dict[str(loop)]
        elif str(loop) in bad_loop_dict.keys():
            vals = bad_loop_dict[str(loop)]
            
        if loop in good:
            new_good_dict[str(loop)] = vals
        elif loop in cand:
            new_cand_dict[str(loop)] = vals
        elif loop in bad:
            new_bad_dict[str(loop)] = vals
    
    return new_good_dict, new_cand_dict, new_bad_dict

#%% Update ifg_loop_df to see how what loops are with what ifg
def update_ifg_df(ifg_loop_df, ifg_dict, corr_loop_df):
    
    n_ifg = ifg_loop_df.shape[0]
    
    ifg_loop_df[['Corr', 'Good', 'Cand', 'Bad']] = 0
    for i in range(n_ifg):
        ix = ifg_loop_df[ifg_loop_df['IFG_IX'] == i].index[0]
        loops = ifg_dict[i]
        for l in loops:
            loop_class = corr_loop_df.loc[l,'Class']
            ifg_loop_df.loc[ix,loop_class] += 1
            
    return ifg_loop_df

#%%
def reset_loops(ifgdir,loopdir, tsadir):
    if os.path.exists(os.path.join(ifgdir, 'Corr_dir')):
        shutil.rmtree(os.path.join(ifgdir, 'Corr_dir'))
    
    for root, dirs, files in os.walk(ifgdir):
        for dir_name in dirs:
            unw = os.path.join(root,dir_name,dir_name + '_unref.unw')
            uncorr = os.path.join(root,dir_name,dir_name + '_uncorr.unw')
            oldmask =  os.path.join(root,dir_name,dir_name + '.unw_maskold')
            # if dir_name=='20200508_20200613': breakpoint()
            # if dir_name[:6]=='202006': print(dir_name), breakpoint()
            if os.path.exists(unw):
                shutil.move(unw, unw[:-10] + '.unw')
                shutil.move(unw +'.png', unw[:-10] + '.unw.png')
                if os.path.exists(uncorr):
                    os.remove(uncorr)
                    os.remove(uncorr + '.png')
                    os.remove(uncorr[:-11] + '_npi.unw.png')
            
            elif os.path.exists(uncorr):
                shutil.move(uncorr, uncorr[:-11] + '.unw')
                shutil.move(uncorr +'.png', uncorr[:-11] + '.unw.png')
                os.remove(uncorr[:-11] + '_npi.unw.png')
                
            elif os.path.exists(oldmask):
                shutil.move(oldmask, oldmask[:-3])

    print(' IFGs Reset')

    for root, dirs, files in os.walk(loopdir):
        for file_name in files:
            if file_name[-6] == 'V':
                if file_name[-5] == '0':
                    shutil.move(os.path.join(root,file_name), os.path.join(root,file_name[:-7] + '.png'))
                else:
                    os.remove(os.path.join(root,file_name))
                    
            if file_name[:5] == 'check':
                os.remove(os.path.join(root,file_name))
    
    for root, dirs, files in os.walk(os.path.join(tsadir,'network')):
        for file_name in files:
            if 'correct' in file_name:
                os.remove(os.path.join(root,file_name))
        
    for root, dirs, files in os.walk(os.path.join(tsadir,'info')):
        for file_name in files:
            if 'correct' in file_name:
                os.remove(os.path.join(root,file_name))
    
    for root, dirs, files in os.walk(os.path.join(tsadir,'12ifg_ras')):
        for file_name in files:
            if 'V1' in file_name:
                os.remove(os.path.join(root,file_name))
            if 'V0' in file_name:
                shutil.move(os.path.join(root,file_name),os.path.join(root,file_name[:-11]+'.unw.png'))
                
    if os.path.exists(os.path.join(loopdir,'loop_pngs')):
        shutil.rmtree(os.path.join(loopdir,'loop_pngs'))

    print(' Loops Reset\nReset Complete\n')
    
#%%
def loop_closure_3rd_wrapper(Aloop, ifgdates, ifgdir, length, width, ref_file):
    refx1, refx2, refy1, refy2 = find_ref_area(ref_file)
    ### Read unw
    unw12, unw23, unw13, ifgd12, ifgd23, ifgd13 = read_unw_loop_ph(Aloop, ifgdates, ifgdir, length, width)

    ## Skip if no data in ref area in any unw. It is bad data.
    ref_unw12 = np.nanmean(unw12[refy1:refy2, refx1:refx2])
    ref_unw23 = np.nanmean(unw23[refy1:refy2, refx1:refx2])
    ref_unw13 = np.nanmean(unw13[refy1:refy2, refx1:refx2])

    ## Calculate loop phase taking into account ref phase
    loop_ph = unw12+unw23-unw13-(ref_unw12+ref_unw23-ref_unw13)
    return np.sqrt(np.nanmean((loop_ph)**2))


#%% Estimated Time to Completion
def etc(start, now, num, n_ifg):
    elapsed = now - start
    end = (elapsed/num) * n_ifg
    hour = int(end/3600)
    minute = int(np.mod((end/60),60))
    sec = int(np.mod(end,60))
    
    return hour, minute, sec


#%%
def loop_closure_4th_wrapper(Aloop, length, width, ifgdates, ifgdir, bad_ifg_all, ref_file):
    n_loop = Aloop.shape[0]
    ns_loop_err1 = np.zeros((length, width), dtype=np.int16)
    
    refx1, refx2, refy1, refy2 = find_ref_area(ref_file)

    for i in range(n_loop):
        if np.mod(i, 100) == 0:
            print("  {0:3}/{1:3}th loop...".format(i, n_loop), flush=True)

        ### Read unw
        unw12, unw23, unw13, ifgd12, ifgd23, ifgd13 = read_unw_loop_ph(Aloop[i, :], ifgdates, ifgdir, length, width)

        ### Skip if bad ifg is included
        if ifgd12 in bad_ifg_all or ifgd23 in bad_ifg_all or ifgd13 in bad_ifg_all:
            continue

        ## Compute ref
        ref_unw12 = np.nanmean(unw12[refy1:refy2, refx1:refx2])
        ref_unw23 = np.nanmean(unw23[refy1:refy2, refx1:refx2])
        ref_unw13 = np.nanmean(unw13[refy1:refy2, refx1:refx2])

        ## Calculate loop phase taking into account ref phase
        loop_ph = unw12+unw23-unw13-(ref_unw12+ref_unw23-ref_unw13)

        ## Count number of loops with suspected unwrap error (>pi)
        loop_ph[np.isnan(loop_ph)] = 0 #to avoid warning
        ns_loop_err1 = ns_loop_err1+(np.abs(loop_ph)>np.pi) #suspected unw error

    return ns_loop_err1
#%%
def plotmask(data,centerz=True,title='',cmap='viridis',vmin=None,vmax=None, interp='antialiased'):
    # cmap_wrap = tools_lib.get_cmap('SCM.romaO')
    plt.figure()
    if centerz:
        vmin=-(np.nanmax(abs(data)))
        vmax=(np.nanmax(abs(data)))

    plt.imshow(data,cmap=cmap,vmin=vmin,vmax=vmax, interpolation=interp)
    plt.colorbar(**{'format': '%.1f'})
    plt.title(title)
    plt.show()
    
    
#%% MASK CORRECTION
def mask_correction(ifgs2fix, ifgdates, ifg_loop_df, corr_loop_df, ifg_dict, n_corr, allowed_loops, length, width, ref_file,n_ifg, start, 
                    fix_total, loop_info, A3loop, thresh, ifgdir, loopdir, tsadir, loop_dict, fixed_ifgs, good_ifg_list, bad_ifg_cands_list, bad_ifg_list, Run=False):

    for ifg in ifgs2fix:
        ifgname = ifgdates[ifg]
        nloops = int(ifg_loop_df.loc[ifg_loop_df['IFG_IX'] == ifg, 'Fixable'])
        loops = ifg_dict[ifg]
        n_corr += 1
        
        # Remove inelligible loops
        if len(loops) != nloops:
            l_use = loops.copy()
            for l in loops:
                if l not in allowed_loops:
                    l_use.remove(l)
            maskloops = l_use
        else:
            maskloops = loops
        
        #Make variables to hold masks and corrections
        corr_all = np.empty((length,width,nloops))
        masks = np.zeros((length,width,nloops), dtype='float32')
        hrs, mins, secs = etc(start, time.time(), n_corr, n_ifg)
        print('({}/{}) Masking'.format(n_corr, fix_total), ifgname + ':', nloops, 'Loops  -  ETC: {} hrs {} mins {} secs'.format(hrs, mins, secs))
     
        if Run: 
                
            # Get correction for each loop
            for count, loop in enumerate(maskloops):
                print('    ',loop_info[loop])
                ifg_position = calc_bad_ifg_position_single3(ifgname, A3loop[loop,:], ifgdates)
                corr_all[:,:,count] = get_corr3(A3loop[loop,:], ifg_position, thresh, ifgdates, ifgdir, length, width, ref_file)
            
            # Find modal correction
            corr, corrcount = get_mode(corr_all)
            
            # FIND CORRECTION MASKS
            mask_thresh = round((nloops+1)/2) # Number required for a majority correction (e.g 4 out of 6)
            good_data = corrcount >= mask_thresh # Where a majority solution has been found
            bad_data = corrcount < mask_thresh # No majority solution found
            
            sim_loops=[] # Reset this variable just in case not needed
            # loop_lib.plotmask(corrcount, centerz=False, title=ifgname)
            
            if nloops % 2 == 0:
                sim_loops = half_check(nloops,corrcount, corr_all)
                #ALSO CONSIDER IF BOTH HALFS ARE EQUALLY GOOD
                    # SKIP AND COME BACK TO IT LATER

            #%% Create correction masks
            # Make masks for the loops
            for count, loop in enumerate(maskloops):
                masks[bad_data,count] -= 1 # Lose a point for no majority solution
                good_corr = corr_all[:,:,count] == corr
                good_corr = good_corr.astype('int') + good_data.astype('int')
                masks[good_corr==2,count] += 2 # Earn two points for being majority solution
                masks[good_corr!=2,count] -= 2 # Lose two points for not being majority solution
                
                if count in sim_loops:
                    masks[corrcount == nloops/2, count] += 4
                
                # Write masks to file (always overwrite previous masks)
                for i in loop_dict[loop]:
                    if i != ifg:
                        maskfile = os.path.join(ifgdir, ifgdates[i], ifgdates[i]+'.mask')
                        mask = masks[:,:,count]
                        mask.tofile(maskfile)
                        
            print('    Correcting....')
            corr_all = np.empty((length,width,nloops))
            
            # Find correction, including the mask
            for count, loop in enumerate(maskloops):
                print('    ',loop_info[loop])
                ifg_position = calc_bad_ifg_position_single3(ifgname, A3loop[loop,:], ifgdates)
                corr_all[:,:,count] = get_corr_mask(A3loop[loop,:], ifg_position, thresh, ifgdates, ifgdir, length, width, ref_file)
            
            # Make correction
            corr, corrcount = get_mode(corr_all)
            
            correct_bad_ifg(ifgname,corr,ifgdir, length, width, A3loop[ifg_dict[ifg],:], ifgdates, loopdir, ref_file)
            if os.path.exists(os.path.join(tsadir,'12ifg_ras',ifgname + '.unw.png')):
                shutil.move(os.path.join(tsadir,'12ifg_ras',ifgname + '.unw.png'),os.path.join(tsadir,'12ifg_ras',ifgname + '_V0.unw.png'))
            shutil.copy(os.path.join(ifgdir,ifgname,ifgname+'.unw.png'),os.path.join(tsadir,'12ifg_ras',ifgname + '_V1.unw.png'))
        
        fixed_ifgs.append(ifg)
        
        # Reclassify loop quality dataframe
        corr_loop_df = update_corr_df(corr_loop_df, ifg, ifgname, ifg_dict, good_ifg_list, bad_ifg_cands_list, bad_ifg_list)
        
    return fixed_ifgs, corr_loop_df, n_corr, fix_total

#%% Secondary correction where IFGs are corrected when they share a loop with 2 corrected IFGs
def fill_corr_loops(fix_df, ifgdates, n_corr, start, n_ifg, fix_total, length, width, corr_loop_df, ifg_dict, loop_info, A3loop, thresh, ref_file,
                    ifgdir, loopdir, tsadir, good_ifg_ix, bad_ifg_cands_ix, bad_ifg_ix, fixed_ifgs, good_ifg_list, bad_ifg_cands_list, bad_ifg_list, loop_dict, Run=False):
    # Iterate through fix_df and fix them, adding in any new IFGs that become eligible
    while np.shape(fix_df)[0] > 0:

        ifg = fix_df.loc[0,'IFG']
        nloops = fix_df.loc[0,'CorrLoops']
        
        # No need for loop lim, as we assume that any corrected ifg is perfect so can use one loop
        ifgname = ifgdates[ifg]
        n_corr += 1
        
        hrs, mins, secs = etc(start, time.time(), n_corr, n_ifg)
        print('({}/{}) Correcting IFG {} {}: {} Loops  -  ETC: {} hrs {} mins {} secs'.format(n_corr, fix_total, ifg, ifgname, nloops, hrs, mins, secs))
        
        loops = list(set(corr_loop_df.index[corr_loop_df['Corr'] == 2].tolist()) & set(ifg_dict[ifg]))
        
        if Run:
            #Make variables to hold corrections
            corr_all = np.empty((length,width,nloops))
            
            # Find correction, including the mask
            for count, loop in enumerate(loops):
                print('    ',loop_info[loop])
                ifg_position = calc_bad_ifg_position_single3(ifgname, A3loop[loop,:], ifgdates)
                corr_all[:,:,count] = get_corr3(A3loop[loop,:], ifg_position, thresh, ifgdates, ifgdir, length, width, ref_file)
            
            # Make correction
            corr, corrcount = get_mode(corr_all)
            
            correct_bad_ifg(ifgname,corr,ifgdir, length, width, A3loop[ifg_dict[ifg],:], ifgdates, loopdir, ref_file)
            
            
            if os.path.exists(os.path.join(tsadir,'12ifg_ras',ifgname + '.unw.png')):
                shutil.move(os.path.join(tsadir,'12ifg_ras',ifgname + '.unw.png'),os.path.join(tsadir,'12ifg_ras',ifgname + '_V0.unw.png'))
            shutil.copy(os.path.join(ifgdir,ifgname,ifgname+'.unw.png'),os.path.join(tsadir,'12ifg_ras',ifgname + '_V1.unw.png'))
        
        fixed_ifgs.append(ifg)
        
        # Reclassify loop quality dataframe
        corr_loop_df = update_corr_df(corr_loop_df, ifg, ifgname, ifg_dict, good_ifg_list, bad_ifg_cands_list, bad_ifg_list)
        
        ## Reset fix_df
        # Get list of all loops now containing 2 corrected IFGs
        loops2fix= corr_loop_df.index[corr_loop_df.Corr == 2].values.tolist()

        # Make list of the IFGs that are not corrected in the 2 Corr loops
        ifgs2fix = list(set(np.array([loop_dict[l] for l in loops2fix]).flatten().tolist()) - set(fixed_ifgs))

        # Make a dataframe to sort the IFGs to fix based on the amount of corr loops available
        fix_df = pd.DataFrame(ifgs2fix,columns = ['IFG'])
        fix_df['CorrLoops'] = 0

        for ix, ifg in enumerate(ifgs2fix):
            fix_df.loc[ix,'CorrLoops'] = sum(corr_loop_df.loc[ifg_dict[ifg]].Corr == 2)

        fix_df = fix_df.sort_values(by=['CorrLoops'], ignore_index=True, ascending=False)
        fix_total = n_corr + len(ifgs2fix)
        
    return n_corr, fix_total, fix_df, corr_loop_df, fixed_ifgs
