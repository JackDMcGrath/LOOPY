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
    - skimage
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
 -v       IFG to give verbose timings for (Development option, Default: -1 (not verbose))
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
import LiCSBAS_plot_lib as plot_lib
from numba import jit, prange
from scipy.ndimage import label
from scipy.interpolate import NearestNDInterpolator
from skimage.segmentation import find_boundaries



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
                v = int(a)-1
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
    _n_para = n_para if n_para < n_ifg else n_ifg
    print('\nRunning error mapping for all {} ifgs,'.format(n_ifg), flush=True)
    print('with {} parallel processing...'.format(_n_para), flush=True)
    if v >= 0:
        print('In an overly verbose way for IFG {}'.format(v+1))
    ### Parallel processing
    p = q.Pool(_n_para)
    mask_cov = np.array(p.map(mask_unw_errors, range(n_ifg)))
    p.close()
    # mask_cov = mask_unw_errors(0)
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
    # print(ID, keep)
    ID_tmp = np.array([*range(ID,len(keep)+ID)]) + 1
    # print(ID_tmp)
    # print('Max Label =', max(labels.flatten()))
    labels, ID = renumber(keep, ID, labels, labels, ID_tmp)
    # print('Max Label2 =', max(labels.flatten()))
    # print('ID =',ID)
#    renumber.parallel_diagnostics(level=4)

    if i==v:
        print('        ({}/{}): renumbered {:.2f}'.format(i+1, n_ifg, time.time()-begin))


    #Labels_tmp only needed for numba regions, not number_regions
    #labels_tmp=np.zeros((length,width,len(vals)),dtype='float32')
 #   if i==v:
#        print('        ({}/{}): labels_tmp frame made with {} vals {:.2f}'.format(i+1, n_ifg, len(vals), time.time()-begin))
    #for ix, val in enumerate(vals):
 #       if i==v:
#            print('        ({}/{}): Label_tmp val {} {:.2f}'.format(i+1, n_ifg, val, time.time()-begin))
        #labels_tmp[:,:,ix] = label((npi==val).astype('int'))[0]

    if i==v:
        commence_num = time.time()
    labels, ID = number_regions(vals, npi, labels, ID, i)
    # print('Max Label3 =', max(labels.flatten()))
    if i==v:
        end_num = time.time()
        print('        ({}/{}): number {:.2f}'.format(i+1, n_ifg, end_num-commence_num))


    if i==v:
        print('        Filling holes {:.2f}'.format(time.time()-begin))
    labels=labels.astype('float32')
    labels[np.isnan(coh)] = np.nan
    # print('Max Label4 =', np.nanmax(labels.flatten()))
    if plot_figures:
        loop_lib.plotmask(labels,centerz=False,title='Labelled Groups')

    # Interpolate labels over gaps left by removing regions that are too small (and also apply filter to npi file)
    # mask = np.where(labels != 0)
    # Only take mask of pixels surrounding areas to be interpolated
    mask = np.where((find_boundaries(labels==0).astype('int') + (labels != 0).astype('int') + (~np.isnan(labels)).astype('int')) == 3)

    labels = NN_interp_samedata(labels, labels, mask)
    # print('Max Label5 =', np.nanmax(labels.flatten()))
    npi = NN_interp_samedata(npi, labels, mask)
    npi[coh<0.05]=np.nan

    if plot_figures:
        loop_lib.plotmask(npi,centerz=True,title='Filtered UNW/{:.1f}pi interp'.format(tol),cmap='tab20b')
        loop_lib.plotmask(labels,centerz=False,title='Filtered Labelled Groups',cmap='tab20b')

    #%%
    start= time.time()
    neighbour_dict = {}
    class_dict = {}
    check_dict = {}
    value_dict = {}
    for r in range(1,ID+1):
        neighbours = mask_lib.find_neighbours(r, labels)
        neighbour_dict[r] = list(neighbours)
        class_dict[r] = 'Unclassified'
        check_dict[r] = 0
        value_dict[r] = np.nanmean(npi[labels==r]).round()

    ref_region= labels[refy1,refx1] # number of region whose neighbors we want

    class_dict[ref_region] = 'Good'
    check_dict[ref_region] = 2

    iterations = 1
    for n in neighbour_dict[ref_region]:
        if class_dict[n] != 'Good' or class_dict[n] != 'Bad':
            if abs(value_dict[n] - value_dict[ref_region]) == 1:
                class_dict[n] = 'Good'
                check_dict[n] = 1
            elif abs(value_dict[n] - value_dict[ref_region]) > 3:
                class_dict[n] = 'Bad'
                check_dict[n] = 1
            else:
                class_dict[n] = 'Cand'

    while 1 in check_dict.values():
        iterations += 1
        error_check = []
        good_check = []

        for r in [k for k,v in check_dict.items() if v == 1]:
            if class_dict[r] == 'Bad':
                error_check.append(r)
            else:
                good_check.append(r)

        for r in error_check:
            check_dict[r] = 2
            for n in neighbour_dict[r]:
                if class_dict[n] == 'Unclassified' or class_dict[n] == 'Cand':
                    if abs(value_dict[n] - value_dict[r]) == 1:
                        class_dict[n] = 'Bad'
                        check_dict[n] = 1
                    elif abs(value_dict[n] - value_dict[r]) > 1 or abs(value_dict[n] - value_dict[r]) < 3:
                        class_dict[n] = 'Cand'
        # breakpoint()
        for r in good_check:
            check_dict[r] = 2
            for n in neighbour_dict[r]:
                if class_dict[n] == 'Unclassified' or class_dict[n] == 'Cand':
                    if abs(value_dict[n] - value_dict[r]) == 1:
                        check_dict[n] = 1
                        class_dict[n] = 'Good'
                    elif abs(value_dict[n] - value_dict[r]) > 3:
                        class_dict[n] = 'Bad'
                        check_dict[n] = 1
                    else:
                        class_dict[n] = 'Cand'
        if i==v:
            print(len([k for k,v in check_dict.items() if v == 2]))
            print(round(time.time()-start,2))



    # Allow isolated regions through
    unclass = [k for k,v in class_dict.items() if v == 'Unclassified']
    for r in unclass:
        if len(neighbour_dict) == 0:
            class_dict[r] = 'Good'
            # Note, isolated unclassified groups still remain. some unclassified may, however, be connected by cands

    # Classify remaining candidates

    cands = [k for k,v in class_dict.items() if v == 'Cand']

    # Classify candidate regions entirely surrounded by a single good or bad class
    for r in cands:
        if all(c == 'Good' for c in [class_dict[v] for v in neighbour_dict[r]]):
            class_dict[r] = 'Good'
        elif all(c == 'Bad' for c in [class_dict[v] for v in neighbour_dict[r]]):
            class_dict[r] = 'Bad'

    cands = [k for k,v in class_dict.items() if v == 'Cand']
    #Tidy up the other candidates
    neutral_check = True

    while len(cands) > 0:
        if i==v:
            print(cands)
        for r in cands:
            y,x = np.where(labels==r)
            labels_trim = labels[y[0]-1:y[-1]+1, x[0]-1:x[-1]+1]
            bound = find_boundaries(labels_trim==r, connectivity=1, mode='outer')
            neighbours, count = np.unique(labels_trim[bound], return_counts=True)
            neighbours = [n for n in neighbours if n == n]
            good_count = 0
            bad_count = 0
            neutral_count = 0
            for ix, n in enumerate(neighbours):
                if class_dict[n] == 'Good':
                    good_count += count[ix]
                elif class_dict[n] == 'Bad':
                    bad_count += count[ix]
                elif neutral_check:
                    neutral_count += count[ix]

            if good_count > bad_count and good_count > neutral_count:
                class_dict[r] = 'Good'
                check_dict[r] = 2
                for n in neighbours:
                    if class_dict[n] == 'Unclassified' or class_dict[n] == 'Cand':
                        if abs(value_dict[n] - value_dict[r]) == 1:
                            check_dict[n] = 1
                            class_dict[n] = 'Good'
                        elif abs(value_dict[n] - value_dict[r]) > 3:
                            class_dict[n] = 'Bad'
                            check_dict[n] = 1
                        else:
                            class_dict[n] = 'Cand'

            elif bad_count > good_count and bad_count > neutral_count:
                class_dict[r] = 'Bad'
                check_dict[r] = 2
                for n in neighbours:
                    if class_dict[n] == 'Unclassified' or class_dict[n] == 'Cand':
                        if abs(value_dict[n] - value_dict[r]) == 1:
                            class_dict[n] = 'Bad'
                            check_dict[n] = 1
                        elif abs(value_dict[n] - value_dict[r]) > 1 or abs(value_dict[n] - value_dict[r]) < 3:
                            class_dict[n] = 'Cand'

        # Redo the check based of the new good values
        while 1 in check_dict.values():
            error_check = []
            good_check = []

            for r in [k for k,v in check_dict.items() if v == 1]:
                if class_dict[r] == 'Bad':
                    error_check.append(r)
                else:
                    good_check.append(r)

            for r in error_check:
                check_dict[r] = 2
                for n in neighbour_dict[r]:
                    if class_dict[n] == 'Unclassified' or class_dict[n] == 'Cand':
                        if abs(value_dict[n] - value_dict[r]) == 1:
                            class_dict[n] = 'Bad'
                            check_dict[n] = 1
                        elif abs(value_dict[n] - value_dict[r]) > 1 or abs(value_dict[n] - value_dict[r]) < 3:
                            class_dict[n] = 'Cand'
            # breakpoint()
            for r in good_check:
                check_dict[r] = 2
                for n in neighbour_dict[r]:
                    if class_dict[n] == 'Unclassified' or class_dict[n] == 'Cand':
                        if abs(value_dict[n] - value_dict[r]) == 1:
                            check_dict[n] = 1
                            class_dict[n] = 'Good'
                        elif abs(value_dict[n] - value_dict[r]) > 3:
                            class_dict[n] = 'Bad'
                            check_dict[n] = 1
                        else:
                            class_dict[n] = 'Cand'

        cands_old = cands
        cands = [k for k,v in class_dict.items() if v == 'Cand']

        if cands_old == cands and neutral_check == False:
            if i==v:
                print('Cant classify remaining Cands. Breaking while loop')
            break
        elif cands_old == cands:
            neutral_check = False
            if i==v:
                print('Removing Neutral Check')
        elif cands_old != cands and neutral_check == False:
            if i==v:
                print('Resetting Neutral Check')
            neutral_check = True


    # Set anything left to be masked

    unclass = [k for k,v in class_dict.items() if v == 'Unclassified' or v == 'Cand']
    for r in unclass:
        class_dict[r] = 'Bad'

    region_class=np.empty((length,width)).astype('float32')*np.nan

    for r in range(1,ID+1):
        if class_dict[r] == 'Good':
            region_class[labels==r] = 2
        elif class_dict[r] == 'Cand':
            region_class[labels==r] = 1
        elif class_dict[r] == 'Bad':
            region_class[labels==r] = -1
        elif class_dict[r] == 'Unclassified':
          region_class[labels==r] = 0

    if i == v:
        print(round(time.time()-start,2))

    title3 = ['Original unw', 'Interpolated unw/{:1}pi'.format(tol),'Unwrapping Error Mask - {} Iterations'.format(iterations)]

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
    interped_data = data.copy()
    interp_to = np.where(((~np.isnan(coh)).astype('int') + np.isnan(data).astype('int')) == 2)
    nearest_data = interp(*interp_to)
    interped_data[interp_to] = nearest_data
    return interped_data

#%%
def NN_interp_samedata(data, mask, mask_ix):
    interp = NearestNDInterpolator(np.transpose(mask_ix), data[mask_ix])
    interp_to = np.where(((~np.isnan(coh)).astype('int') + (mask==0).astype('int')) == 2)
    data_interp = interp(*interp_to)
    data[interp_to] = data_interp
    data[coh<0.05]=np.nan

    return data

#%%
#@jit(nopython=True. parallel=True)
@jit(forceobj=True, parallel=True)
def numba_regions(vals, npi, labels, ID, labels_tmp, i, v):
    if i==v:
        print('Numba Start')
    labels = labels.flatten()
    if i==v:
        print('Labels flat')

    for ix,val in enumerate(vals):
        if i==v:
            print(val)
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
def number_regions(vals, npi, labels, ID, i):
    for ix,val in enumerate(vals):
        start_num = time.time()
        if i==v:
            print(val,'npi', round(time.time()-begin,2))
        if val != 0:
        # if val < -9:
            tmp=np.zeros((length,width),dtype='float32')
            tmp[npi==val] = 1

            labels_tmp, count_tmp = label(tmp)

            # labs, counts = np.unique(labels_tmp, return_counts=True)
            # counts = counts[np.logical_not(labs==0)] # Remove zero label
            # labs = labs[np.logical_not(labs==0)] # Remove zero label
            labs, counts = np.unique(labels_tmp[np.where(labels_tmp!=0)], return_counts=True) #Speed up by ignoring zeros
            too_small = np.where(counts < min_size)[0] + 1
            keep = np.setdiff1d(labs, too_small)

            # Remove regions smaller than the min size
            # labels[np.isin(labels_tmp,too_small)] = 0
            # Speed up again by ignoring 0
            x,y = np.where(labels_tmp != 0)
            drop = np.isin(labels_tmp[np.where(labels_tmp != 0)], too_small)
            x = x[drop]
            y = y[drop]
            labels[x,y] = 0

            ID_tmp = np.array([*range(ID,len(keep)+ID)]) + 1
            labels, ID = renumber(keep, ID, labels, labels_tmp, ID_tmp)

            if i==v:
                print('done', round(time.time()-start_num,2))
    return labels, ID

#%%
@jit(nopython=True, parallel=True)
def renumber(keep, ID, data, data_ref, ID_tmp):
    data = data.flatten()
    data_ref = data_ref.flatten()

    # for region in keep:
    #     ID += 1
    #     data[data_ref==region] = ID

    # Optimised for parallel?
#    ID_tmp = np.array([*range(1+ID,len(keep)+ID)])
    for region in range(0,len(keep)):
        data[data_ref==keep[region]] = ID_tmp[region]
    ID = ID + len(keep)

    return data.reshape(length,width), ID

#%%
@jit(nopython=True, parallel=True)
def prep_df(ID, npi, labels):
    npi_flat = npi.flatten()
    labels_flat = labels.flatten()

    all_regions = np.zeros(4*ID).reshape(ID,4)
    for region in prange(0,ID):
        all_regions[region] = [region+1,round(np.nanmean(npi_flat[labels_flat==region+1])),np.nansum(labels_flat==region+1),0]

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
