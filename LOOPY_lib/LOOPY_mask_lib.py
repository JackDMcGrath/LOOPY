#!/usr/bin/env python3
"""
========
Overview
========
Python3 library of unwrap error location functions for LOOPY

=========
Changelog
=========
v1.0 20220608 Jack McGrath, Uni of Leeds
 - Original implementation
"""
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['QT_QPA_PLATFORM']='offscreen'

import warnings
with warnings.catch_warnings(): ## To silence user warning
    warnings.simplefilter('ignore', UserWarning)
    # mpl.use('Agg')
    
#%% Find negihbouring regions
def find_neighbours(ref_region, labels):
    y = labels == ref_region  # convert to Boolean

    rolled = np.roll(y, 1, axis=0)          # shift down
    rolled[0, :] = False             
    z = np.logical_or(y, rolled)

    rolled = np.roll(y, -1, axis=0)         # shift up 
    rolled[-1, :] = False
    z = np.logical_or(z, rolled)

    rolled = np.roll(y, 1, axis=1)          # shift right
    rolled[:, 0] = False
    z = np.logical_or(z, rolled)

    rolled = np.roll(y, -1, axis=1)         # shift left
    rolled[:, -1] = False
    z = np.logical_or(z, rolled)

    neighbours = set(np.unique(np.extract(z, labels))) - set([ref_region])
    neighbours = {x for x in neighbours if x==x} # Drop NaN value

    return neighbours

#%% Classify neighbour regions as good or bad 
def classify_regions(neighbours, similar, different, unclass, ref_val, npi, tol, labels):
    """
    Function to classify regions as based on if an unwrapping error is detected between two regions
    Similar regions have a difference of 1 tolerance between them
    Different regions are at least 2pi different
    Unclass[ified] regions fall between these two
    """
    for neighbour in neighbours:
        if neighbour not in similar['Region'].values and neighbour not in different['Region'].values:
            diff=np.round(np.nanmean(npi[labels == neighbour])) - ref_val
            if abs(diff) >= 2/tol: # Greater or equal to than 2pi jump
                different.loc[len(different.index)]=[int(neighbour),diff,abs(diff),sum(sum(labels==neighbour)),0]
                if neighbour in unclass['Region'].values:
                    unclass = unclass.drop(unclass[unclass['Region']==neighbour].index.values)
            elif abs(diff) <= 1: # Less than or equal to than 1pi jump
                similar.loc[len(similar.index)]=[int(neighbour),diff,abs(diff),sum(sum(labels==neighbour)),0]
                if neighbour in unclass['Region'].values:
                    unclass = unclass.drop(unclass[unclass['Region']==neighbour].index.values)
            elif  neighbour not in unclass['Region'].values:
                unclass.loc[len(unclass.index)]=[int(neighbour),diff,abs(diff),sum(sum(labels==neighbour)),0]

    different = different.sort_values(['Abs Value','Size'], ascending=False, ignore_index=True)

    return similar.astype('int'), different.astype('int'), unclass.astype('int')

#%% Classify neighbour regions as good, bad or candidate, coming from a good region
def ClassifyFromGoodRegions(neighbours, good, errors, unclass, ref_val, npi, tol, labels):
    """
    Function to classify regions as good, bad o unclassified based on if an unwrapping error is detected
    Required starting region to be good
    """
    for neighbour in neighbours:
        if neighbour not in good['Region'].values and neighbour not in errors['Region'].values:
            diff=np.round(np.nanmean(npi[labels == neighbour])) - ref_val
            if abs(diff) >= 2/tol: # Greater or equal to than 2pi jump
                errors.loc[len(errors.index)]=[neighbour,diff,abs(diff),sum(sum(labels==neighbour)),0]
                if neighbour in unclass['Region'].values:
                    unclass = unclass.drop(unclass[unclass['Region']==neighbour].index.values)
            elif abs(diff) <= 1: # Less than or equal to than 1pi jump
                good.loc[len(good.index)]=[neighbour,diff,abs(diff),sum(sum(labels==neighbour)),0]
                if neighbour in unclass['Region'].values:
                    unclass = unclass.drop(unclass[unclass['Region']==neighbour].index.values).reset_index(drop=True)
            elif  neighbour not in unclass['Region'].values:
                unclass.loc[len(unclass.index)]=[int(neighbour),diff,abs(diff),sum(sum(labels==neighbour)),0]

    errors = errors.sort_values(['Abs Value','Size'], ascending=False, ignore_index=True)

    return good.astype('int'), errors.astype('int'), unclass.astype('int')

#%% Classify neighbour regions as good or bad 
def ClassifyFromErrorRegions(neighbours, good, errors, unclass, ref_val, npi, tol, labels):
    """
    Function to classify regions as bad or unclassified based on the detection of unwrapping errors
    Requires starting region to be bad. Regions cannot be classed as good here, as it is unknown if an unwrapping
    error from a bad region is making the data better or worse
    """
    for neighbour in neighbours:
        if neighbour not in good['Region'].values and neighbour not in errors['Region'].values:
            diff=np.round(np.nanmean(npi[labels == neighbour])) - ref_val
            if abs(diff) <= 1: # Less than or equal to than 1pi jump
                errors.loc[len(errors.index)]=[neighbour,diff,abs(diff),sum(sum(labels==neighbour)),0]
                if neighbour in unclass['Region'].values:
                    unclass = unclass.drop(unclass[unclass['Region']==neighbour].index.values).reset_index(drop=True)
            elif neighbour not in unclass['Region'].values:
                unclass.loc[len(unclass.index)]=[int(neighbour),diff,abs(diff),sum(sum(labels==neighbour)),0] 
            

    errors = errors.sort_values(['Abs Value','Size'], ascending=False, ignore_index=True)

    return errors.astype('int'), unclass.astype('int')

#%% Classify candidate as good or bad 
def ClassifyCands(neighbours, good, errors, region, value, labels):
    """
    Function to classify candidate regions as good or bad based on what it touches.
    Assume that to be bad only if it is in contact with an error region
    """

    if any([x for x in neighbours if x in good['Region'].values]):
        good.loc[len(good.index)]=[region,value,abs(value),sum(sum(labels==region)),0]
    else:
        errors.loc[len(errors.index)]=[region,value,abs(value),sum(sum(labels==region)),0]

    return good.astype('int'), errors.astype('int')


#%% Remove previously made masks and pngs
def reset_masks(ifgdir):
    
    for root, dirs, files in os.walk(ifgdir):
        for dir_name in dirs:
            mask = os.path.join(root,dir_name,dir_name + '.unw_mask')
            maskpng = os.path.join(root,dir_name,dir_name + '.mask.png')
          
            if os.path.exists(mask):
                os.remove(mask)
                
            if os.path.exists(maskpng):
                os.remove(maskpng)

    print('Reset Complete\n')

#%%
def make_npi_mask_png(data2, pngfile, cmap, title2):
    """
    Make png with 3 images for comparison.
    data3 and title3 must be list with 3 elements.
    cmap can be 'insar'. To wrap data, np.angle(np.exp(1j*x/cycle)*cycle)
    """
    ### Plot setting
    interp = 'nearest' #'antialiased'

    length, width = data2[0].shape
    figsizex = 12
    xmergin = 4
    figsizey = int((figsizex-xmergin)/2*length/width)+2
    
    fig = plt.figure(figsize = (figsizex, figsizey))

    for i in range(2):
        ax = fig.add_subplot(1, 2, i+1) #index start from 1
        if i == 1:
            im = ax.imshow(data2[i], cmap=cmap[i], interpolation=interp, vmin=-1, vmax=1)
        else:
            im = ax.imshow(data2[i], cmap=cmap[i], interpolation=interp)
        ax.set_title(title2[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if i == 0: 
            fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(pngfile)
    plt.close()
   
    return 

#%%
def make_unw_npi_mask_png(data3, pngfile, cmap, title3):
    """
    Make png with 3 images for comparison.
    data3 and title3 must be list with 3 elements.
    cmap can be 'insar'. To wrap data, np.angle(np.exp(1j*x/cycle)*cycle)
    """
    ### Plot setting    
    interp = 'nearest'
    length, width = data3[0].shape
    figsizex = 12
    xmergin = 4
    figsizey = int((figsizex-xmergin)/2*length/width)+2
    fig = plt.figure(figsize = (figsizex, figsizey))

    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1) #index start from 1
        if i == 2:
            im = ax.imshow(data3[i], cmap=cmap[i], interpolation=interp, vmin=-1, vmax=1)
        else:
            im = ax.imshow(data3[i], cmap=cmap[i], interpolation=interp)
        ax.set_title(title3[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        fig.colorbar(im, ax=ax, location='bottom')
    plt.tight_layout()
    try:
        plt.savefig(pngfile)
    except:
        print('ERROR: Mask Comparison Figure Failed to Save. Error usually\n    MemoryError: Unable to allocate [X] MiB for an array with shape (Y, Z) and data type int64.\nSkipping')
    
    plt.close()
   
    return 
