#!/usr/bin/env python3
"""
========
Overview
========
This script identifies errors in unwrapped interferograms, and creates a mask
that can be applied to these interferograms before they are then used to
correct other IFGs

New modules needed:
    - scipy
    - pandas
    - numba

===============
Input & output files
===============
Inputs in GEOCml*/ :
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw[.png]
   - yyyymmdd_yyyymmdd.cc
 - slc.mli.par

Inputs in TS_GEOCml*/ :
 - results/coh_avg  : Average coherence

Outputs in GEOCml*/ :
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.mask : Boolean Mask
   - yyyymmdd_yyyymmdd.unw_mask : Masked unwrapped IFG
   - yyyymmdd_yyyymmdd.unw_mask.png : png comparison of unw, npi and mask

Outputs in TS_GEOCml*/ :
 - info/
   - mask_info.txt : Basic stats of mask coverage

=====
Usage
=====
LOOPY01_find_errors.py -d ifgdir [-t tsadir] [-m int] [--reset] [--n_para]

 -d       Path to the GEOCml* dir containing stack of unw data.
 -t       Path to the output TS_GEOCml* dir. (Default: TS_GEOCml*)
 -m       Minimum region size to be used in masking (Default: 1 (No filtering))
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
import warnings
import numpy as np
import pandas as pd
import multiprocessing as multi
import LOOPY_mask_lib as mask_lib
import LOOPY_loop_lib as loop_lib
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
from numba import jit
from scipy.ndimage import label
from scipy.interpolate import NearestNDInterpolator


insar = tools_lib.get_cmap('SCM.romaO')

class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg
        
#%% Main
def main(argv=None):

    #%% Check argv
    if argv == None:
        argv = sys.argv

    start = time.time()
    ver="1.1.0"; date=20220615; author="J. McGrath"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)
    
    global plot_figures, tol, min_size, refx1, refx2, refy1, refy2, n_ifg, \
        length, width, ifgdir, ifgdates, coh, i, v, begin
    
    #%% Set default
    ifgdir = []
    tsadir = []
    min_size = 1 # Minimum size of labelled regions
    reset = False
    plot_figures = False
    tol = 0.5 # N value to use for modulo pi division
    v = -1
#    min_size = 100 # Minimum size of labelled regions

    # Parallel Processing options    
    try:
        n_para = len(os.sched_getaffinity(0))
    except:
        n_para = multi.cpu_count()
    
    if sys.platform == "linux" or sys.platform == "linux2":
        q = multi.get_context('fork')
    elif sys.platform == "win32":
        q = multi.get_context('spawn')

    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hd:t:m:v:", ["help", "reset", "n_para="])
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
                min_size = int(a)
            elif o == '-v':
                v = int(a-1)
            elif o == '--reset':
                reset = True
            elif o == '--n_para':
                n_para = int(a)

        if not ifgdir:
            raise Usage('No data directory given, -d is not optional!')
        elif not os.path.isdir(ifgdir):
            raise Usage('No {} dir exists!'.format(ifgdir))
        elif not os.path.exists(os.path.join(ifgdir, 'slc.mli.par')):
                raise Usage('No slc.mli.par file exists in {}!'.format(ifgdir))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2
    
    #%% Directory setting
    ifgdir = os.path.abspath(ifgdir)

    if not tsadir:
        tsadir = os.path.join(os.path.dirname(ifgdir), 'TS_'+os.path.basename(ifgdir))

    if not os.path.exists(tsadir): os.mkdir(tsadir)

    netdir = os.path.join(tsadir, 'network')
    if not os.path.exists(netdir): os.mkdir(netdir)

    infodir = os.path.join(tsadir, 'info')
    if not os.path.exists(infodir): os.mkdir(infodir)
    
    resultsdir = os.path.join(tsadir, 'results')
    if not os.path.exists(resultsdir): os.mkdir(resultsdir)
    
    if reset:
        print('Removing Previous Masks')
        mask_lib.reset_masks(ifgdir)
    else:
        print('Preserving Premade Masks')

    #%% File Setting
    ref_file = os.path.join(infodir,'12ref.txt')
    mlipar = os.path.join(ifgdir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))
    
    cohfile=os.path.join(resultsdir,'coh_avg')
    # If no coh file, use slc
    if not os.path.exists(cohfile): 
        cohfile=os.path.join(ifgdir,'slc.mli')
        print('No Coherence File - using SLC instead')    

    coh=io_lib.read_img(cohfile,length=length, width=width)
    n_px = sum(sum(~np.isnan(coh[:])))
    
    mask_info_file = os.path.join(infodir, 'mask_info.txt')
    f = open(mask_info_file, 'w')
    print('# Size: {0}({1}x{2}), n_valid: {3}'.format(width*length, width, length, n_px), file=f)
    print('# ifg dates         mask_cov', file=f)
    f.close()


    #%% Prepare variables
    # Get ifg dates
    ifgdates = tools_lib.get_ifgdates(ifgdir)
    n_ifg = len(ifgdates)
    mask_cov = []

    # Find reference pixel. If none provided, use highest coherence pixel
    if os.path.exists(ref_file):
        with open(ref_file, "r") as f:
            refarea = f.read().split()[0]  #str, x1/x2/y1/y2
        refx1, refx2, refy1, refy2 = [int(s) for s in re.split('[:/]', refarea)]

        if np.isnan(coh[refy1:refy2,refx1:refx2]):
            print('Ref point = [{},{}] invalid. Using max coherent pixel'.format(refy1,refx1))
        
            refy1,refx1 = np.where(coh==np.nanmax(coh))
            refy1 = refy1[0]
            refy2 = refy1 + 1
            refx1 = refx1[0]
            refx2 = refx1 + 1

    else:
        print('No Reference Pixel provided - using max coherent pixel')

        refy1,refx1 = np.where(coh==np.nanmax(coh))
        refy1 = refy1[0]
        refy2 = refy1 + 1
        refx1 = refx1[0]
        refx2 = refx1 + 1
    
    print('Ref point = [{},{}]'.format(refy1,refx1))
    print('Minumum Region Size = {}'.format(min_size))
    
    #%% Run correction in parallel
    v = 1
    _n_para = n_para if n_para < n_ifg else n_ifg
    print('\nRunning error mapping for all {} ifgs,'.format(n_ifg), flush=True)
    print('with {} parallel processing...'.format(_n_para), flush=True)
    if v >= 0:
        print('In an overly verbose way for IFG {}'.format(v+1))
    ### Parallel processing
    p = q.Pool(_n_para)
    mask_cov = np.array(p.map(mask_unw_errors, range(n_ifg)))
    p.close()
 
    f = open(mask_info_file, 'a')
    for i in range(n_ifg):
        print('{0}  {1:6.2f}'.format(ifgdates[i], mask_cov[i]/n_px), file=f)
    f.close()
        
    #%% Finish
    print('\nCheck network/*, 11bad_ifg_ras/* and 11ifg_ras/* in TS dir.')
    print('If you want to change the bad ifgs to be discarded, re-run with different thresholds or make a ifg list and indicate it by --rm_ifg_list option in the next step.')

    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minute = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minute,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output directory: {}\n'.format(os.path.relpath(ifgdir)))

#%%
def mask_unw_errors(i):
    global begin
    begin=time.time()
    date = ifgdates[i]
    if i==v:
        print('        Starting')#    print('    ({}/{}): {}'.format(i+1, n_ifg, date))
    if os.path.exists(os.path.join(ifgdir,date,date+'.unw_mask')):
        print('    ({}/{}): {}  Mask Exists. Skipping'.format(i+1, n_ifg, date))
        mask_coverage = 0
        return mask_coverage
    else:
        print('    ({}/{}): {}'.format(i+1, n_ifg, date))

    if i==v:
        print('        Loading')
  

    unw=io_lib.read_img(os.path.join(ifgdir,date,date+'.unw'),length=length, width=width)
    if i==v:
        print('        UNW Loaded {:.2f}'.format(time.time()-begin))

    ref = np.nanmean(unw[refy1:refy2, refx1:refx2])
    if np.isnan(ref):
        print('Invalid Ref Value found. Setting to 0')
        ref = 0
        
    ifg=unw.copy()
    ifg = ifg-ref #Maybe no need to use a reference - would be better to subtract 0.5 pi or something, incase IFG is already referenced
    if i==v:
        print('        Reffed {:.2f}'.format(time.time()-begin))
        
    if plot_figures:
        loop_lib.plotmask(ifg,centerz=False,title='UNW')
    
    #%%
    # data = ifg
    filled_ifg = NN_interp(ifg)
    if i==v:
         print('            filled_ifg  {:.2f}'.format(time.time()-begin))
    npi = (filled_ifg/(tol*np.pi)).round()
    
    if plot_figures:
        loop_lib.plotmask(filled_ifg,centerz=False,title='UNW interp')
        loop_lib.plotmask(npi,centerz=True,title='UNW/{:.1f}pi interp'.format(tol),cmap='tab20b')

    #%%
    # Find all unique values of npi
    vals= np.unique(npi)
    vals = vals[~np.isnan(vals)]
    
    # Make tmp array of values where npi == 0, and label (best to start with 0 region for labels)
    tmp=np.zeros((length,width))
    tmp[npi==0] = 1
    if i==v:
        print('        Labelling {:.2f}'.format(time.time()-begin))
    labels, count = label(tmp)
    if i==v:
        print('        ({}/{}): {} regions'.format(i+1, n_ifg, np.nanmax(np.nanmax(labels))))
#    mapped_area = 0
    
    labs, counts = np.unique(labels, return_counts=True)
    counts = counts[np.logical_not(labs==0)] # Remove zero label
    labs = labs[np.logical_not(labs==0)] # Remove zero label
    
    if i==v:
        print('        ({}/{}): unique {:.2f}'.format(i+1, n_ifg, time.time()-begin))
    too_small = np.where(counts < min_size)[0] + 1
    if i==v:
        print('        ({}/{}): too small {:.2f}'.format(i+1, n_ifg, time.time()-begin))
    keep = np.setdiff1d(labs, too_small)
    if i==v:
        print('        ({}/{}): keep {:.2f}'.format(i+1, n_ifg, time.time()-begin))
    
    ID = 0
    # Remove regions smaller than the min size
    labels[np.isin(labels,too_small)] = 0

    if i==v:
        print('        ({}/{}): remove {:.2f}'.format(i+1, n_ifg, time.time()-begin))
    
    # Renumber remaining regions
    labels, ID = renumber(keep, ID, labels)

    if i==v:
        print('        ({}/{}): renumbered {:.2f}'.format(i+1, n_ifg, time.time()-begin))
   
    labels_tmp=np.zeros((length,width,len(vals)),dtype='float32')
    if i==v:
        print('        ({}/{}): labels_tmp frame made {:.2f}'.format(i+1, n_ifg, time.time()-begin))
    for ix, val in enumerate(vals):
        if i==v:
            print('        ({}/{}): Label_tmp val {} {:.2f}'.format(i+1, n_ifg, val, time.time()-begin))
        labels_tmp[:,:,ix] = label((npi==val).astype('int'))[0]

    labels, ID = number_regions(vals, npi, labels, ID, labels_tmp, i)

    if i==v:
        print('        Filling holes {:.2f}'.format(time.time()-begin))        
    labels=labels.astype('float32')
    labels[np.isnan(coh)] = np.nan

    if plot_figures:
        loop_lib.plotmask(labels,centerz=False,title='Labelled Groups')
    
    # Interpolate labels over gaps left by removing regions that are too small (and also apply filter to npi file)
    mask = np.where(labels != 0)
    labels = NN_interp_samedata(labels, mask)
    npi = NN_interp_samedata(npi, mask)
    npi[coh<0.05]=np.nan
    

    if plot_figures:
        loop_lib.plotmask(npi,centerz=True,title='Filtered UNW/{:.1f}pi interp'.format(tol),cmap='tab20b')
        loop_lib.plotmask(labels,centerz=False,title='Filtered Labelled Groups',cmap='tab20b')
    
    #%% All regions are now labelled. Now search for neighbour regions to the reference region
    if i==v:
        print('        Prepping DF {:.2f}'.format(time.time()-begin))

    all_regions = prep_df(ID, npi, labels)
    if i==v:
        print('        DF array {:.2f}'.format(time.time()-begin))

    all_regions=pd.DataFrame(all_regions,columns=['Region','Value','Size','Checked'])
    # # Suppress runtime warning for nanmean and nansum on values with only nans
    # warnings.simplefilter("ignore", category=RuntimeWarning)
    # for region in range(0,ID):
    #     all_regions.loc[region]=[region+1,np.nanmean(npi[labels==region+1]),np.nansum(labels==region+1),0]
    # warnings.simplefilter("default", category=RuntimeWarning)
    all_regions = all_regions.set_index('Region')
    if i==v:
        print('        DF prepped {:.2f}'.format(time.time()-begin))
    #https://stackoverflow.com/questions/38073433/determine-adjacent-regions-in-numpy-array
    ref_region= labels[refy1,refx1] # number of region whose neighbors we want
    
    if i==v:
        print('        Finding Ref Neighbours {:.2f}'.format(time.time()-begin))

    all_regions.loc[ref_region,'Checked'] = 2
    neighbours = mask_lib.find_neighbours(ref_region, labels)
    for neighbour in neighbours:
        all_regions.loc[neighbour,'Checked'] = 1
    if i==v:
        print('        Neighbours found {:.2f}'.format(time.time()-begin))


    #%
    region_class=np.zeros((length,width))*np.nan
    ref_val = all_regions.loc[ref_region,'Value']
    errors = pd.DataFrame([],columns=['Region','Value', 'Abs Value','Size','CheckNeighbour'])
    good = pd.DataFrame([],columns=['Region','Value', 'Abs Value','Size','CheckNeighbour'])
    cands = pd.DataFrame([],columns=['Region','Value', 'Abs Value','Size','CheckNeighbour'])
    good.loc[len(good.index)]=[ref_region,ref_val,abs(ref_val),sum(sum(labels==ref_region)),1]
    if i==v:
        print('        Classifying from ref {:.2f}'.format(time.time()-begin))

    good, errors, cands = mask_lib.ClassifyFromGoodRegions(neighbours, good, errors, cands, ref_val, npi, tol, labels)
    #%
    
    iteration = 1
    while 1 in all_regions['Checked'].values:
        if i==v:
            print('        ({}/{}): {} iterations {:.2f}'.format(i+1, n_ifg, iteration, time.time()-begin))
        errors, cands = class_from_error(errors, all_regions, labels, good, cands, npi)
        # for region in errors['Region'].values:
        #     if all_regions.loc[region,'Checked'] != 2:
        #         all_regions.loc[region,'Checked'] = 2
        #         neighbours = mask_lib.find_neighbours(region, labels)
        #         for neighbour in neighbours:
        #             if all_regions.loc[neighbour,'Checked'] == 0:
        #                 all_regions.loc[neighbour,'Checked'] = 1
        #         errors, cands = mask_lib.ClassifyFromErrorRegions(neighbours, good, errors, cands, all_regions.loc[region,'Value'], npi, tol, labels)

        good, errors, cands = class_from_good(good, all_regions, labels, errors, cands, npi)
        # for region in good['Region'].values:
        #     if all_regions.loc[region,'Checked'] != 2:
        #         all_regions.loc[region,'Checked'] = 2
        #         neighbours = mask_lib.find_neighbours(region, labels)
        #         for neighbour in neighbours:
        #             if all_regions.loc[neighbour,'Checked'] == 0:
        #                 all_regions.loc[neighbour,'Checked'] = 1
        #         good, errors, cands = mask_lib.ClassifyFromGoodRegions(neighbours, good, errors, cands, all_regions.loc[region,'Value'], npi, tol, labels)
    
    
        if plot_figures:
            for region in cands['Region'].values:
                region_class[labels==region] = 0
        
            for region in good['Region'].values:
                region_class[labels==region] = 1
        
            for region in errors['Region'].values:
                region_class[labels==region] = -1
            if np.mod(iteration-1,1) == 0:
                loop_lib.plotmask(region_class,centerz=False,title='Region Classifier {}'.format(iteration))
        iteration += 1
        
        if sum(all_regions['Checked'].values==2) == good.shape[0] + errors.shape[0]:
            if i == v:
                print('        Breaking while loop - only cands left')
            good, errors = class_cand(cands, labels, good, errors)
            # for region in cands['Region'].values:
            #     neighbours = mask_lib.find_neighbours(region, labels)
            #     value = cands[cands['Region']==region].index.values[0]
            #     good, errors = mask_lib.ClassifyCands(neighbours, good, errors, region, value, labels)
            if i == v:
                print('        Cand regions classified')
            break
    #%%
    region_class = make_class_map(all_regions.index.values, labels, good['Region'].values, errors['Region'].values, length, width)
    # region_class=np.zeros((length,width)).astype('float32')*np.nan
    
    # for region in all_regions.index:
    #     if region in good['Region'].values:
    #         region_class[labels==region] = 1
    #     elif region in errors['Region'].values:
    #         region_class[labels==region] = -1
    #     else: # For any isolated regions
    #         region_class[labels==region] = 0
    
    if plot_figures:
        loop_lib.plotmask(region_class,centerz=False,title='Final Region Classifier')
        
    # plot_lib.make_im_png(region_class, os.path.join(ifgdir,date,date+'.mask.png'), 'viridis', 'Unwrapping Error Mask - {} Iterations'.format(iteration), cbar=False)
    # plot_lib.make_im_png(npi, os.path.join(ifgdir,date,date+'.npi.png'), 'tab20b', 'Interpolated unw/{:1}pi'.format(tol), cbar=True)
    title3 = ['Original unw', 'Interpolated unw/{:1}pi'.format(tol),'Unwrapping Error Mask - {} Iterations'.format(iteration)]

    region_class[np.isnan(unw)] = np.nan
    mask_lib.make_unw_npi_mask_png([unw, npi,region_class], os.path.join(ifgdir,date,date+'.mask.png'), [insar,'tab20c','viridis'], title3)
    # mask_lib.make_npi_mask_png([npi,region_class], os.path.join(ifgdir,date,date+'.mask2.png'), ['tab20c','viridis'], title3[1:])
    # region_class.tofile(os.path.join(ifgdir,date,date+'.mask'))
    #%% Save Masked UNW to save time in corrections
    mask_coverage = sum(sum(region_class==-1))
    masked = unw.copy().astype('float32')
    masked[region_class == -1] = np.nan
    region_class[region_class==1]=0
    if i==v:
        print('        Writing maskfile {:.2f}'.format(time.time()-begin))
        print('            ' + os.path.join(ifgdir,date,date+'.mask'))
        print('            ' + os.path.join(ifgdir,date,date+'.unw_mask'))


    region_class.astype('bool').tofile(os.path.join(ifgdir,date,date+'.mask'))
    masked.tofile(os.path.join(ifgdir,date,date+'.unw_mask'))
    print('            ' + os.path.join(ifgdir,date,date+'.unw_mask written'))
    return mask_coverage

#%%
def NN_interp(data):
    mask = np.where(~np.isnan(data))
    interp = NearestNDInterpolator(np.transpose(mask), data[mask])
    interped_data = np.zeros((length,width))*np.nan
    nearest_data = interp(*np.where(~np.isnan(coh)))
    interped_data[np.where(~np.isnan(coh))] = nearest_data
    interped_data[np.isnan(coh)]=np.nan
    interped_data[coh<0.05]=np.nan

    return interped_data

#%%
def NN_interp_samedata(data, mask):
    interp = NearestNDInterpolator(np.transpose(mask), data[mask])
    data_interp = interp(*np.where(~np.isnan(coh)))
    data[np.where(~np.isnan(coh))] = data_interp
    data[coh<0.05]=np.nan
    
    return data
    
#%%
#@jit(nopython=True)
@jit(forceobj=True)
def numba_regions(vals, npi, labels, ID, labels_tmp, i):
    labels = labels.flatten()
    for ix,val in enumerate(vals):
 #       if i==v:
#            print(val)
        if val != 0:
            # tmp=np.zeros((length,width),dtype='float32')
            # tmp[npi==val] = 1
            # labels_tmp, count_tmp = label(tmp)
            # labs, counts = np.unique(labels[:,:,ix], return_counts=True)
            labs_tmp = np.unique(labels_tmp[:,:,ix])
            counts = np.zeros(len(labs_tmp))
            for lab_ix, l in enumerate(labs_tmp):
                counts[lab_ix] = len(np.where(labels_tmp[:,:,ix].flatten()==l))

            too_small = np.where(counts < min_size)[0]
            keep = np.setdiff1d(labs_tmp, too_small)
#            keep = [x for x in labs_tmp if x not in too_small]

            # Remove regions smaller than the min size
            labels[np.isin(labels_tmp[:,:,ix],too_small).flatten()] = 0
#            labels[np.array([c for c,x in enumerate(labels_tmp[:,:,ix].flatten()) if x in too_small])] = 0
#            for region in too_small:
#                labels[labels_tmp==region] = 0

            # Renumber remaining regions
            for region in keep:
                ID += 1
                labels[(labels_tmp[:,:,ix]==region).flatten()] = ID
    
    return labels.reshape(length,width), ID

#%% Because numba regions is too slow
def number_regions(vals, npi, labels, ID, labels_tmp, i):
    for ix,val in enumerate(vals):
        start_num = time.time()
        if i==v:
            print(val,'npi', round(time.time()-begin,2))
        if val != 0:
            tmp=np.zeros((length,width),dtype='float32')
            tmp[npi==val] = 1
            labels_tmp, count_tmp = label(tmp)
            labs, counts = np.unique(labels_tmp, return_counts=True)
            counts = counts[np.logical_not(labs==0)] # Remove zero label
            labs = labs[np.logical_not(labs==0)] # Remove zero label            

#            if val<-0:
#                print(labs)
#                print(counts)

            too_small = np.where(counts < min_size)[0] + 1
            keep = np.setdiff1d(labs, too_small)
  #          if val<-0:
 #               print(too_small)
#                print(keep)



            # Remove regions smaller than the min size
            labels[np.isin(labels_tmp,too_small)] = 0
            # Renumber remaining regions
            if i==v:
                print('Prep Labels...', round(time.time()-start_num,2))
                print(len(keep))

            for region in keep:
                ID += 1
                labels[(labels_tmp==region)] = ID
   #             if val==-3:
  #                  print(ID,':',np.sum(labels_tmp==region))
 #                   print(labels[labels_tmp==region])
#                    print(npi[labels_tmp==region])
            if i==v:
                print('done', round(time.time()-start_num,2))
    return labels, ID

#%%
@jit(nopython=True)
def renumber(keep, ID, data):
    data = data.flatten()
    for region in keep:
        ID += 1
        data[data==region] = ID
    return data.reshape(length,width), ID

#%%
#@jit(nopython=True)
def prep_df(ID, npi, labels):
    npi_flat = npi.flatten()
    labels_flat = labels.flatten()

    all_regions = np.zeros(4*ID).reshape(ID,4)
    for region in range(0,ID):
  #      if np.isnan(np.nanmean(npi_flat[labels_flat==region+1])):
 #           print(npi[labels==region+1])
#            print([region+1,np.nanmean(npi_flat[labels_flat==region+1]),np.nansum(labels_flat==region+1),0])

        all_regions[region] = [region+1,np.nanmean(npi_flat[labels_flat==region+1]),np.nansum(labels_flat==region+1),0]
#        all_regions[region] = [region+1,np.mean(npi_flat[labels_flat==region+1]),np.nansum(labels_flat==region+1),0]
    return all_regions

#%%
@jit(nopython=True)
def make_class_map(all_regions, labels, good, errors, length, width):
    region_class=np.zeros((length*width)).astype('float32')*np.nan
    labels = labels.flatten()

    for region in all_regions:
        if region in good:
            region_class[labels==region] = 1
        elif region in errors:
            region_class[labels==region] = -1
        else: # For any isolated regions
            region_class[labels==region] = 0
    
    return region_class.reshape(length, width)

#%%
def class_from_error(errors, all_regions, labels, good, cands, npi):
    for region in errors['Region'].values:
        if all_regions.loc[region,'Checked'] != 2:
            all_regions.loc[region,'Checked'] = 2
            neighbours = mask_lib.find_neighbours(region, labels)
            for neighbour in neighbours:
               if all_regions.loc[neighbour,'Checked'] == 0:
                    all_regions.loc[neighbour,'Checked'] = 1
            errors, cands = mask_lib.ClassifyFromErrorRegions(neighbours, good, errors, cands, all_regions.loc[region,'Value'], npi, tol, labels)

    return errors, cands
    
#%%
def class_from_good(good, all_regions, labels, errors, cands, npi):
    for region in good['Region'].values:
        if all_regions.loc[region,'Checked'] != 2:
            all_regions.loc[region,'Checked'] = 2
            neighbours = mask_lib.find_neighbours(region, labels)
            for neighbour in neighbours:
                if all_regions.loc[neighbour,'Checked'] == 0:
                    all_regions.loc[neighbour,'Checked'] = 1
            good, errors, cands = mask_lib.ClassifyFromGoodRegions(neighbours, good, errors, cands, all_regions.loc[region,'Value'], npi, tol, labels)

    return good, errors, cands
    
#%%
def class_cand(cands, labels, good, errors):
    for region in cands['Region'].values:
        neighbours = mask_lib.find_neighbours(region, labels)
        value = cands[cands['Region']==region].index.values[0]
        good, errors = mask_lib.ClassifyCands(neighbours, good, errors, region, value, labels)
    
    return good, errors

#%% main
if __name__ == "__main__":
    sys.exit(main())
