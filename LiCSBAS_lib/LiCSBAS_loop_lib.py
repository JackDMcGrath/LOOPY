#!/usr/bin/env python3
"""
========
Overview
========
Python3 library of loop closure check functions for LiCSBAS.

=========
Changelog
=========
v1.5.2 20210303 Yu Morishita, GSI
 - Use get_cmap in make_loop_png
 - Add colorbar in make_loop_png
v1.5.1 20201119 Yu Morishita, GSI
 - Change default cmap for wrapped phase from insar to SCM.romaO
v1.5 20201006 Yu Morishita, GSI
 - Update make_loop_png
v1.4 20200828 Yu Morishita, GSI
 - Update for matplotlib >= 3.3
 - Use nearest interpolation for insar cmap to avoid aliasing
v1.3 20200703 Yu Morioshita, GSI
 - Replace problematic terms
v1.2 20200224 Yu Morioshita, Uni of Leeds and GSI
 - Change color of loop phase
v1.1 20190906 Yu Morioshita, Uni of Leeds and GSI
 - tight_layout for loop png
v1.0 20190708 Yu Morioshita, Uni of Leeds and GSI
 - Original implementation
"""
import re
import os
import numpy as np
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib

os.environ['QT_QPA_PLATFORM']='offscreen'
import warnings
import matplotlib as mpl
with warnings.catch_warnings(): ## To silence user warning
    warnings.simplefilter('ignore', UserWarning)
    mpl.use('Agg')
from matplotlib import pyplot as plt

import SCM
import shutil


#%%
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

    return Aloop

#%%
def read_unw_loop_ph(Aloop1, ifgdates, ifgdir, length, width, bad_ifg=[]):
    ### Find index of ifg
    ix_ifg12, ix_ifg23 = np.where(Aloop1 == 1)[0]
    ix_ifg13 = np.where(Aloop1 == -1)[0][0]
    ifgd12 = ifgdates[ix_ifg12]
    ifgd23 = ifgdates[ix_ifg23]
    ifgd13 = ifgdates[ix_ifg13]

    ### Read unw data
    unw12file = os.path.join(ifgdir, ifgd12, ifgd12+'.unw')
    unw12 = io_lib.read_img(unw12file, length, width)
    unw12[unw12 == 0] = np.nan # Fill 0 with nan
    unw23file = os.path.join(ifgdir, ifgd23, ifgd23+'.unw')
    unw23 = io_lib.read_img(unw23file, length, width)
    unw23[unw23 == 0] = np.nan # Fill 0 with nan
    unw13file = os.path.join(ifgdir, ifgd13, ifgd13+'.unw')
    unw13 = io_lib.read_img(unw13file, length, width)
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
    cmap_wrap = tools_lib.get_cmap('SCM.romaO')
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
#%% New Functions
#####%%
#%% Make loops with 4 IFGS
def make_4loop_matrix(ifgdates):
    """
    Make loop matrix (containing 1, -1, 0) from ifgdates. with 4th ifg

    Inputs:
      ifgdates : Unwrapped phase vector at a point without nan (n_ifg)

    Returns:
      Aloop : Loop matrix with 1 for ifg12/ifg23 and -1 for ifg13
              (n_loop, n_ifg)

    """
    n_ifg = len(ifgdates)
    Aloop = []
    
    # 12 + 23 +34 - 14 Loop
    
    for ix_ifg12, ifgd12 in enumerate(ifgdates):
        primary12 = ifgd12[0:8]
        secondary12 = ifgd12[9:17]
        ifgdates23 = [ ifgd for ifgd in ifgdates if
                      ifgd.startswith(secondary12)] # all candidates of ifg23

        for ifgd23 in ifgdates23: # for each candidate of ifg23
            secondary23 = ifgd23[9:17]
            ifgdates34 = [ ifgd for ifgd in ifgdates if
                          ifgd.startswith(secondary23)] # all candidates of ifg34
            
            for ifgd34 in ifgdates34:
                secondary34 = ifgd34[9:17]
                try:
                    ## Search ifg14
                    ix_ifg14 = ifgdates.index(primary12+'_'+secondary34)
                except: # no loop for this ifg34. Next.
                    continue

                ## Loop found
                ix_ifg23 = ifgdates.index(ifgd23)
                ix_ifg34 = ifgdates.index(ifgd34)
                Aline = [0]*n_ifg
                Aline[ix_ifg12] = 1
                Aline[ix_ifg23] = 1
                Aline[ix_ifg34] = 1
                Aline[ix_ifg14] = -1
                Aloop.append(Aline)
    
    # 12 + 24 - 34 - 13 Loop

    for ix_ifg12, ifgd12 in enumerate(ifgdates):
        primary12 = ifgd12[0:8]
        secondary12 = ifgd12[9:17]
        ifgdates24 = [ ifgd for ifgd in ifgdates if
                      ifgd.startswith(secondary12)] # all candidates of ifg23

        for ifgd24 in ifgdates24: # for each candidate of ifg23
            secondary24 = ifgd24[9:17]
            ifgdates34 = [ ifgd for ifgd in ifgdates if
                          ifgd.endswith(secondary24)] # all candidates of ifg34
            
            for ifgd34 in ifgdates34:
                if ifgd34 != secondary12+'_'+secondary24: # Make sure that ifg 24 and 34 are not the same
                    primary34 = ifgd34[:8]
                    try:
                        ## Search ifg14
                        ix_ifg13 = ifgdates.index(primary12+'_'+primary34)
                    except: # no loop for this ifg34. Next.
                        continue

                    ## Loop found
                    ix_ifg24 = ifgdates.index(ifgd24)
                    ix_ifg34 = ifgdates.index(ifgd34)
                    Aline = [0]*n_ifg
                    Aline[ix_ifg12] = 1
                    Aline[ix_ifg24] = 1
                    Aline[ix_ifg34] = -1
                    Aline[ix_ifg13] = -1
                    Aloop.append(Aline)

    Aloop = np.array(Aloop)

    return Aloop

#%% Function to check and correct loop
def check_loop_threshold(loop, thresh, ifgdates, ifgdir, length, width, multi_prime = True):
    # global ifgdates, ifgdir, tsadir, length, width, refx1, refx2, refy1, refy2
    unw12, unw23, unw13, ifgd12, ifgd23, ifgd13 = read_unw_loop_ph(loop, ifgdates, ifgdir, length, width)

    ## Calculate loop phase and check n bias (2pi*n)
    loop_ph = unw12+unw23-unw13
    if multi_prime:
        bias = np.nanmedian(loop_ph)
        loop_ph = loop_ph - bias # unbias inconsistent fraction phase
    rms = np.sqrt(np.nanmean(loop_ph**2))

    bad_loop = (rms > thresh)

    return bad_loop, rms

#%% Function to check and correct loop
def check_4loop_threshold(loop, thresh, ifgdates, ifgdir, length, width, refx1, refx2, refy1, refy2, multi_prime=True):
    # global ifgdates, ifgdir, tsadir, length, width, refx1, refx2, refy1, refy2
    unw1, unw2, unw3, unw4, ifgd1, ifgd2, ifgd3, ifgd4 = read_unw_4loop_ph(loop, ifgdates, ifgdir, length, width)

    ## Skip if no data in ref area in any unw. It is bad data.
    ref_unw1 = np.nanmean(unw1[refy1:refy2, refx1:refx2])
    ref_unw2 = np.nanmean(unw2[refy1:refy2, refx1:refx2])
    ref_unw3 = np.nanmean(unw3[refy1:refy2, refx1:refx2])
    ref_unw4 = np.nanmean(unw4[refy1:refy2, refx1:refx2])

    ## Calculate loop phase and check n bias (2pi*n)
    if sum(loop == 1) == 2:
        loop_ph = unw1 + unw2 - unw3 - unw4 - (ref_unw1 + ref_unw2 - ref_unw3 - ref_unw4)
    elif sum(loop == 1) == 3:
        loop_ph = unw1 + unw2 + unw3 - unw4 - (ref_unw1 + ref_unw2 + ref_unw3 - ref_unw4)
        
    if multi_prime:
        bias = np.nanmedian(loop_ph)
        loop_ph = loop_ph - bias # unbias inconsistent fraction phase
    rms = round(np.sqrt(np.nanmean(loop_ph**2)),2)

    bad_loop = (rms > thresh)

    return bad_loop, rms

#%%
def read_unw_4loop_ph(Aloop1, ifgdates, ifgdir, length, width, bad_ifg=[]):
    ### Find index of ifg
    if sum(Aloop1 == 1 ) == 3: # Loop 1 + 2 + 3 - 4
        ix_ifg1, ix_ifg2, ix_ifg3 = np.where(Aloop1 == 1)[0]
        ix_ifg4 = np.where(Aloop1 == -1)[0][0]
    elif sum(Aloop1 == 1 ) == 2: # Loop 1 + 2 - 3 - 4
        ix_ifg1, ix_ifg2 = np.where(Aloop1 == 1)[0]
        ix_ifg3, ix_ifg4 = np.where(Aloop1 == -1)[0]
    
    ifgd1 = ifgdates[ix_ifg1]
    ifgd2 = ifgdates[ix_ifg2]
    ifgd3 = ifgdates[ix_ifg3]
    ifgd4 = ifgdates[ix_ifg4]

### Read unw data
    unw1file = os.path.join(ifgdir, ifgd1, ifgd1+'.unw')
    unw1 = io_lib.read_img(unw1file, length, width)
    unw1[unw1 == 0] = np.nan # Fill 0 with nan
    unw2file = os.path.join(ifgdir, ifgd2, ifgd2+'.unw')
    unw2 = io_lib.read_img(unw2file, length, width)
    unw2[unw2 == 0] = np.nan # Fill 0 with nan
    unw3file = os.path.join(ifgdir, ifgd3, ifgd3+'.unw')
    unw3 = io_lib.read_img(unw3file, length, width)
    unw3[unw3 == 0] = np.nan # Fill 0 with n
    unw4file = os.path.join(ifgdir, ifgd4, ifgd4+'.unw')
    unw4 = io_lib.read_img(unw4file, length, width)
    unw4[unw1 == 0] = np.nan # Fill 0 with n

    return unw1, unw2, unw3, unw4, ifgd1, ifgd2, ifgd3, ifgd4

#%% Fuction to calculate which of the 3 ifgs in the loop is bad
def calc_bad_ifg_position_single(bad_ifg, loop_dates):
    bad_dates = bad_ifg.split('_')
    
    if bad_dates[0] == loop_dates[0]:
        if bad_dates[1] == loop_dates[1]:
            ifg_ix = 1
        else:
            ifg_ix = 3
    else:
        ifg_ix = 2
        
    return ifg_ix
        
#%% Function to correct the bad interferogram using loop closure
def get_corr(loop, bad_ix, thresh, ifgdates, ifgdir, length, width, refx1, refx2, refy1, refy2, multi_prime = True):

    unw12, unw23, unw13, ifgd12, ifgd23, ifgd13 = read_unw_loop_ph(loop, ifgdates, ifgdir, length, width)
    # if bad_ix == 3:
        # breakpoint()
    
    ## Skip if no data in ref area in any unw. It is bad data.
    ref_unw12 = np.nanmean(unw12[refy1:refy2, refx1:refx2])
    ref_unw23 = np.nanmean(unw23[refy1:refy2, refx1:refx2])
    ref_unw13 = np.nanmean(unw13[refy1:refy2, refx1:refx2])

    ## Calculate loop phase taking into account ref phase
    loop_ph = unw12+unw23-unw13-(ref_unw12+ref_unw23-ref_unw13)
    
    if multi_prime:
        bias = np.nanmedian(loop_ph)
        loop_ph = loop_ph - bias # unbias inconsistent fraction phase
    
    loop_error = loop_ph * (np.absolute(loop_ph) > (thresh))
    # loop_error[loop_error == 0] = np.nan

    n_pi_correction = (loop_error / (2*np.pi)).round() # Modulo division would mean that a pixel out by 6 rads wouldn't be corrected, for example
    loop_correction = n_pi_correction * 2 * np.pi
    # loop_correction[np.isnan(loop_correction)]=0
    # titles4 = ['1','2','3','PH_Error'+str(bad_ix)]
    # make_loop_png(unw12,unw23,unw13, loop_ph, os.path.join(ifgdir,'Modal_Correction_triple'+str(bad_ix)+'.unw.png'), titles4, 3)
    
    if bad_ix == 3:
        loop_correction = loop_correction * -1
    # loop_correction[loop_correction==0] = np.nan
    # loop_correction = loop_ph
    # plot_lib.make_im_png(loop_correction,os.path.join(ifgdir,'Modal_Correction'+str(bad_ix)+'.unw.png'),SCM.roma,'Correction from one loop')

    return loop_correction

#%% Correct Bad IFG
def correct_bad_ifg(bad_ifg_name,corr,ifgdir, length, width, loop, ifgdates, tsadir, ref_file, cycle=3):
    ifgfile = os.path.join(ifgdir, bad_ifg_name, bad_ifg_name + '.unw')
    # print(ifgfile)
    ifg = io_lib.read_img(ifgfile, length, width)
    ifg[ifg == 0] = np.nan # Fill 0 with nan
    corr[np.isnan(corr)] = 0
    corr_ifg = ifg - corr
    
    title = '{} ({}pi/cycle)'.format(bad_ifg_name, cycle*2)
    
    # Backup original unw file and loop png
    shutil.move(os.path.join(ifgdir, bad_ifg_name, bad_ifg_name + '.unw'),os.path.join(ifgdir, bad_ifg_name, bad_ifg_name + '_uncorr.unw'))
    shutil.move(os.path.join(ifgdir, bad_ifg_name, bad_ifg_name + '.unw.png'),os.path.join(ifgdir, bad_ifg_name, bad_ifg_name + '_uncorr.unw.png'))
    plot_lib.make_im_png(np.angle(np.exp(1j*corr_ifg/3)*3), os.path.join(ifgdir, bad_ifg_name, bad_ifg_name + '.unw.png'), SCM.romaO, title, -np.pi, np.pi, cbar=False)
    
    # Make new unw file from corrected data and new loop png
    corr_ifg.tofile(os.path.join(ifgdir, bad_ifg_name, bad_ifg_name + '.unw'))

    plot_lib.make_im_png(corr_ifg/np.pi,os.path.join(ifgdir,bad_ifg_name,'loop_correction_npi.unw.png'),SCM.hawaii,bad_ifg_name)
    plot_lib.make_im_png(corr_ifg,os.path.join(ifgdir,bad_ifg_name,'loop_correction_rads.unw.png'),SCM.hawaii,bad_ifg_name)


    for l in range(loop.shape[0]):
        
        re_loop(loop[l], ifgdates, ifgdir, length, width, tsadir, ref_file, cycle=cycle)
        
#%% Function to correct the bad interferogram using loop closure
def correct_ifg_from_loop(loop, bad_ix, thresh):
    cycle = 3
    
    unw12, unw23, unw13, ifgd12, ifgd23, ifgd13 = read_unw_loop_ph(loop, ifgdates, ifgdir, length, width)

    # Get Dates
        
    imd1 = ifgd12[:8]
    imd2 = ifgd23[:8]
    imd3 = ifgd23[-8:]

    ## Calculate loop phase
    loop_ph = unw12+unw23-unw13
    
    loop_error = loop_ph * (np.absolute(loop_ph) > thresh)
    loop_error[loop_error == 0] = np.nan

    n_pi_correction = (loop_error // (2*np.pi)) 
    loop_correction = n_pi_correction * 2 * np.pi
    loop_correction[np.isnan(loop_correction)]=0
    
    if bad_ix == 1:
        ifg = imd1 + '_' + imd2
        unw_corr = unw12 - loop_correction
        unw12 = unw12 - loop_correction

    elif bad_ix == 2:
        ifg = imd2 + '_' + imd3
        unw_corr = unw23 - loop_correction
        unw23 = unw23 - loop_correction

    else:
        ifg = imd1 + '_' + imd3
        unw_corr = unw13 + loop_correction
        unw13 = unw13 + loop_correction

    print('\n \n###############\n###############\n CORRECTING', ifg, '\n###############\n###############\n\n')


    print('Correcting',ifg,'from loop',imd1+'_'+imd2+'_'+imd3)
        
    # Backup original unw file and loop png
    shutil.move(os.path.join(ifgdir, ifg, ifg + '.unw'),os.path.join(ifgdir, ifg, ifg + '_uncorr.unw'))
    shutil.move(os.path.join(ifgdir, ifg, ifg + '.unw.png'),os.path.join(ifgdir, ifg, ifg + '_uncorr.unw.png'))
    # Make new unw file from corrected data and new loop png
    unw_corr.tofile(os.path.join(ifgdir, ifg, ifg + '.unw'))
    
    title = '{} ({}pi/cycle)'.format(ifg, cycle*2)
    png = os.path.join(ifgdir,ifg,ifg+'.unw.png')
    plot_lib.make_im_png(np.angle(np.exp(1j*unw_corr/cycle)*cycle), png, SCM.romaO, title, -np.pi, np.pi, cbar=False)
    
    n_pi_correction[n_pi_correction == 0] = np.nan
    loop_correction[loop_correction == 0] = np.nan
    
    plot_lib.make_im_png(n_pi_correction,os.path.join(ifgdir,ifg,'loop_correction_npi.unw.png'),SCM.hawaii,ifg)
    plot_lib.make_im_png(loop_correction,os.path.join(ifgdir,ifg,'loop_correction_rads.unw.png'),SCM.hawaii,ifg)

    ## Calculate loop phase taking into account ref phase
    loop_ph_corr = unw12+unw23-unw13
    
    rms = np.sqrt(np.nanmean(loop_ph_corr**2))

    titles4 = ['{} ({}*2pi/cycle)'.format(imd1+'_'+imd2, cycle),
               '{} ({}*2pi/cycle)'.format(imd2+'_'+imd3, cycle),
               '{} ({}*2pi/cycle)'.format(imd1+'_'+imd3, cycle),]
    titles4.append('Loop phase (RMS={:.2f}rad)'.format(rms))
    
    png = os.path.join(tsadir, '12loop','bad_loop_png',imd1+'_'+imd2+'_'+imd3+'_loop.png')
    shutil.move(png,png[:-4]+'_uncorr.png')
    
    make_loop_png(unw12, unw23, unw13, loop_ph_corr, png, titles4, cycle)
    plot_lib.make_im_png(loop_ph-loop_ph_corr,os.path.join(ifgdir,ifg,'corr_diff.unw.png'),SCM.hawaii,ifg)
    
#%% Function to re-run loops where interferograms have been corrected
def re_loop(loop, ifgdates, ifgdir, length, width, tsadir, ref_file, cycle = 3, multi_prime = True):
    
    unw12, unw23, unw13, ifgd12, ifgd23, ifgd13 = read_unw_loop_ph(loop, ifgdates, ifgdir, length, width)
    
    with open(ref_file, "r") as f:
        refarea = f.read().split()[0]  #str, x1/x2/y1/y2
    refx1, refx2, refy1, refy2 = [int(s) for s in re.split('[:/]', refarea)]
    
    imd1 = ifgd12[:8]
    imd2 = ifgd23[:8]
    imd3 = ifgd23[-8:]
    
    ## Skip if no data in ref area in any unw. It is bad data.
    ref_unw12 = np.nanmean(unw12[refy1:refy2, refx1:refx2])
    ref_unw23 = np.nanmean(unw23[refy1:refy2, refx1:refx2])
    ref_unw13 = np.nanmean(unw13[refy1:refy2, refx1:refx2])

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
    
    png = os.path.join(tsadir, '12loop','bad_loop_png',imd1+'_'+imd2+'_'+imd3+'_loop')
    shutil.move(png + '.png',png+'_uncorr.png')
    
    make_loop_png(unw12, unw23, unw13, loop_ph, png[:-4]+'loop.png', titles4, cycle)
    
    print('Loop:',imd1+'_'+imd2+'_'+imd3, 'New RMS:',rms)
    
#%%
def make_corr_loop_png(unw12, unw23, unw13, loop_ph, png, titles4, cycle):
    cmap_wrap = tools_lib.get_cmap('SCM.romaO')
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
