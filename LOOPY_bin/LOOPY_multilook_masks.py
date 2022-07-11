#!/usr/bin/env python3
"""
v1.0.0 20220616 Jack McGrath, Uni of Leeds

This script will take masks and multilook them to lower resolutions

====================
Input & output files
====================
Inputs:


Outputs in GEOCml*/ (downsampled if indicated):

=====
Usage
=====
LOOPY_multilook_mask.py -i GEOCdir -o outdir -n out_nlook [-a in_nlook] [--n_para int]

 -i  Path to the input GEOCml dir
 -o  Path to the output GEOCml dir
 -n  Number of downsampling factor (Default: 1, no downsampling)
 -a  Number of downsampling of input data (Default: 1. If not 1, nlooks = int(out/in)
 --n_para  Number of parallel processing (Default: # of usable CPU)

"""
#%% Change log
'''
v1.0.0 20220616 Jack McGrath, Uni of Leeds
 - Mask multilooking implement
v1.7.4 20201119 Yu Morishita, GSI
 - LiCSBAS02_ml_prep.py
'''
#TODO: Make it mask the ml1 data directly
#TODO: Create mask based of .tif rather than ml1 data
#TODO: Automatically work out nlook based of original and final size?

#%% Import
import getopt
import os
import sys
import time
import numpy as np
import multiprocessing as multi
import LOOPY_loop_lib as loop_lib
import LOOPY_mask_lib as mask_lib
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib

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
    ver="1.0.0"; date=20220616; author="J. McGrath"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)

    ### For parallel processing
    global ifgdates, length, width, nlook, n_ifg, geocdir, outdir


    #%% Set default
    geocdir = []
    outdir = []
    inlook = 1
    outlook = 1

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
            opts, args = getopt.getopt(argv[1:], "hi:o:n:a:", ["help", "n_para="])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-i':
                geocdir = a
            elif o == '-o':
                outdir = a
            elif o == '-n':
                outlook = int(a)
            elif o == '-a':
                inlook = int(a)
            elif o == '--n_para':
                n_para = int(a)

        if not geocdir:
            raise Usage('No GEOC directory given, -i is not optional!')
        elif not os.path.isdir(geocdir):
            raise Usage('No {} dir exists!'.format(geocdir))
        elif not outdir:
            raise Usage('No OUTDIR directory given, -o is not optional!')
        elif not os.path.isdir(outdir):
            raise Usage('No {} dir exists!'.format(outdir))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2


    #%% Directory and file setting
    geocdir = os.path.abspath(geocdir)

    mlipar = os.path.join(outdir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))

    no_unw_list = os.path.join(outdir, 'no_unw_list.txt')
    if os.path.exists(no_unw_list): os.remove(no_unw_list)

    nlook = int(outlook / inlook)
    if nlook < 1:
        nlook = 1
        print('Nlook < 1: This is not allowed. Skipping multilooking ')

    #%% Get IFG dates and multilook in parallel
    ifgdates = tools_lib.get_ifgdates(geocdir)
    n_ifg = len(ifgdates)

    print('\nMultilooking Error Maps for {} ifgs,'.format(n_ifg), flush=True)

    if n_para == 1:
        print('with no parallel processing...', flush=True)
        for i in range(n_ifg):
            multi_look_mask(i)

    else:
        _n_para = n_para if n_para < n_ifg else n_ifg

        print('with {} parallel processing...'.format(_n_para), flush=True)

        ### Parallel processing
        p = q.Pool(_n_para)
        p.map(multi_look_mask, range(n_ifg))
        p.close()

    print('\nMask multilooking complete. Run LiCSBAS to produce time-series')

    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minute = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minute,sec))

    print('\nSuccessfully finished!!\n')
    print('Output directory: {}\n'.format(outdir))


#%%
def multi_look_mask(i):
    date = ifgdates[i]

    print('    ({}/{}): {}'.format(i+1, n_ifg, date))

    maskfile = os.path.join(geocdir, date, date+'.mask')
    mask = io_lib.read_img(maskfile, length,width, dtype='bool').astype('float32')
    mask = tools_lib.multilook(mask, nlook, nlook, 0.1).astype('bool')
    mask = (mask > 0.5)

    # Apply mask to multilooked data
    unwfile = os.path.join(outdir, date, date + '.unw')
    unw = io_lib.read_image(unwfile, length, width)

    titles = ['UNW', 'ML{} Mask'.format(nlook)]
    mask_lib.make_npi_mask_png([unw, mask], os.path.join(outdir,date,date+'.mask2.png'), [insar,'viridis'], titles)

    unw[mask] = np.nan
    mask.astype('bool').tofile(os.path.join(outdir,date,date+'.mask'))
    unw.tofile(os.path.join(outdir,date,date+'.unw_mask'))

#%% main
if __name__ == "__main__":
    sys.exit(main())
