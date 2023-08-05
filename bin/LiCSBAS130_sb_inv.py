#!/usr/bin/env python3
"""
v1.5.4 20221020 Qi Ou, Leeds Uni
v1.5.3 20211122 Milan Lazecky, Leeds Uni
v1.5.2 20210311 Yu Morishita, GSI

This script inverts the SB network of unw to obtain the time series and
velocity using NSBAS (López-Quiroz et al., 2009; Doin et al., 2011) approach.
A stable reference point is determined after the inversion. RMS of the time
series wrt median among all points is calculated for each point.
Then the point with minimum RMS and minimum n_gap is selected as new stable
reference point.

===============
Input & output files
===============
Inputs in GEOCml*/ (--comp_cc_dir):
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.cc
 - EQA.dem_par
 - slc.mli.par
 - baselines (may be dummy)
[- [ENU].geo]

Inputs in GEOCml*/ (--cc_dir+suffix):
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw

Inputs in TS_GEOCml*/ :
 - info/
   - 120ref.txt
[-results/]
[  - coh_avg]
[  - hgt]
[  - n_loop_err]
[  - n_unw]
[  - slc.mli]

Outputs in TS_GEOCml*/ :
 - 130cum*.h5             : Cumulative displacement (time-seires) in mm
 - 130results*/
   - vel*[.png]        : Velocity in mm/yr (positive means LOS decrease; uplift)
   - vintercept*[.png] : Constant part of linear velocity (c for vt+c) in mm
   - resid_rms*[.png]  : RMS of residual in mm
   - n_gap*[.png]      : Number of gaps in SB network
   - n_ifg_noloop*[.png] :  Number of ifgs with no loop
   - maxTlen*[.png]    : Max length of continous SB network in year
 - info/
   - 13parameters*.txt : List of used parameters
   - 130used_image*.txt : List of used images
   - 130resid*.txt      : List of RMS of residual for each ifg
   - 130ref*.txt[kml]   : Auto-determined stable ref point
   - 130rms_cum_wrt_med*[.png] : RMS of cum wrt median used for ref selection
 - 130increment*/yyyymmdd_yyyymmdd.increment.png
     : Comparison between unw and inverted incremental displacement
 - 130resid*/yyyymmdd_yyyymmdd.res.png : Residual for each ifg

=====
Usage
=====
LiCSBAS130_sb_inv.py -d ifgdir [-t tsadir] [--inv_alg LS|WLS] [--mem_size float] [--gamma float] [--n_para int] [--n_unw_r_thre float] [--keep_incfile] [--gpu] [--fast] [--only_sb] [--nopngs]

"""
#%% Change log
'''
v1.5.3 20211122 Milan Lazecky, Leeds Uni
 - use fast_nsbas and only_sb to help make processing faster
v1.5.2 20210311 Yu Morishita, GSI
 - Include noise indices and LOS unit vector in cum.h5 file
v1.5.1 20210309 Yu Morishita, GSI
 - Change default --mem_size to 8000
 - Speed up by reading cum data on memory
v1.5 20210305 Yu Morishita, GSI
 - Add GPU option
 - Speed up by activating n_para_inv and OMP_NUM_THREADS=1
v1.4.8 20210127 Yu Morishita, GSI
 - Automatically reduce mem_size if available RAM is small
v1.4.7 20201124 Yu Morishita, GSI
 - Comporess hdf5 file
v1.4.6 20201119 Yu Morishita, GSI
 - Change default cmap for wrapped phase from insar to SCM.romaO
v1.4.5 20201118 Yu Morishita, GSI
 - Again Bug fix of multiprocessing
v1.4.4 20201116 Yu Morishita, GSI
 - Bug fix of multiprocessing in Mac python>=3.8
v1.4.3 20201104 Yu Morishita, GSI
 - Bug fix when n_pt_unnan=0 in a patch
v1.4.2 20201028 Yu Morishita, GSI
 - Update how to get n_para
v1.4.1 20200925 Yu Morishita, GSI
 - Small bug fix in n_para
v1.4 20200909 Yu Morishita, GSI
 - n_core -> n_para
 - Parallelize making png
v1.3 20200902 Yu Morishita, GSI
 - Parallelize calculation of n_gap and n_ifg_noloop
 - Change n_core default to # of usable CPU
 - Fix n_core_inv=1 for inversion because it already uses multicore
v1.2 20200225 Yu Morishita, Uni of Leeds and GSI
 - Not output network pdf
 - Change color of png
 - Change name of parameters.txt to 13parameters.txt
 - Deal with cc file in uint8 format
 - Automatically find stable reference point
v1.1 20190829 Yu Morishita, Uni of Leeds and GSI
 - Remove cum.h5 if exists before creation
v1.0 20190730 Yu Morishita, Uni of Leeds and GSI
 - Original implementation
'''
# - network/network13*.png : Figures of the network

#%% Import
import getopt
import os
os.environ["OMP_NUM_THREADS"] = "1"
# Because np.linalg.lstsq use full CPU but not much faster than 1CPU.
# Instead parallelize by multiprocessing. Here because can cause CPU issues
# if imported before numpy
import sys
import re
import time
import psutil
import h5py as h5
import numpy as np
import datetime as dt
import multiprocessing as multi
import SCM
import LiCSBAS_io_lib as io_lib
import LiCSBAS_inv_lib as inv_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_loop_lib as loop_lib
import LiCSBAS_plot_lib as plot_lib
import argparse
import shutil


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    '''
    Use a multiple inheritance approach to use features of both classes.
    The ArgumentDefaultsHelpFormatter class adds argument default values to the usage help message
    The RawDescriptionHelpFormatter class keeps the indentation and line breaks in the ___doc___
    '''
    pass


class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg


def init_args():
    # read inputs
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=CustomFormatter)
    parser.add_argument('-f', dest='frame_dir', default="./", help="directory of LiCSBAS output of a particular frame")
    parser.add_argument('-c', dest='cc_dir', default="GEOCml10GACOS", help="folder containing connected cc files")
    # parser.add_argument('-d', dest='unw_dir', help="folder containing unw input to be corrected")
    parser.add_argument('-t', dest='ts_dir', default="TS_GEOCml10GACOS", help="folder containing time series")
    parser.add_argument('-m', dest='memory_size', default=2048, type=float, help="Max memory size for each patch in MB")
    parser.add_argument('-g', dest='gamma', default=0.0001, type=float, help="Gamma value for NSBAS inversion")
    parser.add_argument('-l', dest='ifg_list', default=None, type=str, help="text file containing a list of ifgs")
    parser.add_argument('--suffix', default="", type=str, help="suffix of the output")
    parser.add_argument('--inv_alg', metavar="[LS|WLS]", default="WLS", help="Inversion algolism \n    LS :       NSBAS Least Square with no weight\n   WLS:       NSBAS Weighted Least Square (not well tested)\n              Weight (variance) is calculated by (1-coh**2)/(2*coh**2)")
    parser.add_argument('--n_unw_r_thre', metavar="THRE", type=float, help="Threshold of n_unw (number of used unwrap data) \n (Note this value is ratio to the number of images; i.e., 1.5*n_im) \n Larger number (e.g. 2.5) makes processing faster but result sparser. \n (Default: 1 and 0.5 for C- and L-band, respectively)")
    parser.add_argument('--n_para', type=int, help="Number of parallel processing (Default: # of usable CPU)")
    parser.add_argument('--keep_incfile', default=False, action='store_true', help="Remove inc and resid files if False")
    parser.add_argument('--nopngs', default=False, action='store_true', help="Avoid generating some (unnecessary) PNG previews of increment residuals etc.")
    parser.add_argument('--gpu', default=False, action='store_true', help="Use GPU (Need cupy module)")
    parser.add_argument('--fast', default=False, action='store_true', help="Use more economic NSBAS computation (should be faster and less demanding, may bring errors in points with many gaps)")
    parser.add_argument('--only_sb', default=False, action='store_true', help="Perform only SB processing (skipping points with NaNs)")
    parser.add_argument('--null_noloop', default=False, action='store_true', help="Don't include any IFG pixel that isn't associated with a good loop")
    parser.add_argument('--no_inversion', default=False, action='store_true', help="Don't do velocity inverison, just do noise indicies")
    parser.add_argument('--no_loops_removed', default=False, action='store_true', help="If no loop pixels have been removed, don't write out mask file")
    args = parser.parse_args()

    return args


def main():

    start = time.time()
    ver="1.0"; date=20220929; author="Q. Ou"
    print("\n{} ver{} {} {}".format(os.path.basename(sys.argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)

    ## For parallel processing
    global n_para_gap, G, Aloop, unwpatch, imdates, incdir, ccdir, ifgdir, length, width,\
        coef_r2m, ifgdates, ref_unw, cycle, keep_incfile, resdir, restxtfile, \
        cmap_vel, cmap_wrap, wavelength, args

    try:
        n_para = len(os.sched_getaffinity(0))
    except:
        n_para = multi.cpu_count()

    os.environ["OMP_NUM_THREADS"] = "1"
    # Because np.linalg.lstsq use full CPU but not much faster than 1CPU.
    # Instead parallelize by multiprocessing

    cmap_vel = SCM.roma.reversed()
    cmap_noise = 'viridis'
    cmap_noise_r = 'viridis_r'
    cmap_wrap = SCM.romaO
    q = multi.get_context('fork')
    compress = 'gzip'

    args = init_args()

    # Make sure requested n_para is not larger than the available
    if args.n_para < n_para:
        n_para = args.n_para

    if args.gpu:
        print("\nGPU option is activated. Need cupy module.\n")
        import cupy as cp

    # define input directories
    ccdir = os.path.abspath(os.path.join(args.frame_dir, args.cc_dir))  # to read .cc
    if args.suffix == "1":
        ifgdir = os.path.abspath(os.path.join(args.frame_dir, args.cc_dir))  # to read .unw
    else:
        ifgdir = os.path.abspath(os.path.join(args.frame_dir, args.cc_dir+args.suffix))  # to read .unw
    tsadir = os.path.abspath(os.path.join(args.frame_dir, args.ts_dir))   # to read 120.ref, to write cum.h5
    infodir = os.path.join(tsadir, 'info')  # to read 120.ref
    reffile = os.path.join(infodir, '120ref.txt')

    # define output directories and files
    resultsdir = os.path.join(tsadir, '130results'+args.suffix)  # to save vel, vintercept, rms etc
    if os.path.exists(resultsdir): shutil.rmtree(resultsdir)
    os.mkdir(resultsdir)

    resdir = os.path.join(tsadir, '130resid'+args.suffix)  # to save .res
    if os.path.exists(resdir): shutil.rmtree(resdir)
    os.mkdir(resdir)

    incdir = os.path.join(tsadir,'130increment'+args.suffix)
    if os.path.exists(incdir): shutil.rmtree(incdir)
    os.mkdir(incdir)

    cumh5file = os.path.join(tsadir, '130cum{}.h5'.format(args.suffix))
    restxtfile = os.path.join(infodir, '130resid{}.txt'.format(args.suffix))
    imfile = os.path.join(infodir, '130used_image{}.txt'.format(args.suffix))
    rms_cum_wrt_med_file = os.path.join(infodir, '130rms_cum_wrt_med{}'.format(args.suffix))
    rms_cum_png = os.path.join(infodir, '130rms_cum_wrt_med{}.png'.format(args.suffix))
    # refsfile = os.path.join(infodir, '130ref{}.txt'.format(args.suffix))
    # refkml = os.path.join(infodir, '130ref{}.kml'.format(args.suffix))

    if n_para > 32:
        # Emprically >32 does not make much faster despite using large resource
        n_para_inv = 32
    else:
        n_para_inv = n_para

    #%% Set preliminaly reference
    with open(reffile, "r") as f:
        refarea = f.read().split()[0]  #str, x1/x2/y1/y2
    refx1, refx2, refy1, refy2 = [int(s) for s in re.split('[:/]', refarea)]

    #%% Read data information
    ### Get size
    mlipar = os.path.join(ccdir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))
    speed_of_light = 299792458 #m/s
    radar_frequency = float(io_lib.get_param_par(mlipar, 'radar_frequency')) #Hz
    wavelength = speed_of_light/radar_frequency #meter
    coef_r2m = -wavelength/4/np.pi*1000 #rad -> mm, positive is -LOS

    ### Calc pixel spacing depending on IFG or GEOC, used in later spatial filter
    dempar = os.path.join(ccdir, 'EQA.dem_par')
    width_geo = int(io_lib.get_param_par(dempar, 'width'))
    length_geo = int(io_lib.get_param_par(dempar, 'nlines'))
    dlat = float(io_lib.get_param_par(dempar, 'post_lat')) #negative
    dlon = float(io_lib.get_param_par(dempar, 'post_lon')) #positive
    lat1 = float(io_lib.get_param_par(dempar, 'corner_lat'))
    lon1 = float(io_lib.get_param_par(dempar, 'corner_lon'))
    if width == width_geo and length == length_geo: ## Geocoded
        print('\nIn geographical coordinates', flush=True)
        centerlat = lat1+dlat*(length/2)
        ra = float(io_lib.get_param_par(dempar, 'ellipsoid_ra'))
        recip_f = float(io_lib.get_param_par(dempar, 'ellipsoid_reciprocal_flattening'))
        rb = ra*(1-1/recip_f) ## polar radius
        pixsp_a = 2*np.pi*rb/360*abs(dlat)
        pixsp_r = 2*np.pi*ra/360*dlon*np.cos(np.deg2rad(centerlat))
    else:
        print('\nIn radar coordinates', flush=True)
        pixsp_r_org = float(io_lib.get_param_par(mlipar, 'range_pixel_spacing'))
        pixsp_a = float(io_lib.get_param_par(mlipar, 'azimuth_pixel_spacing'))
        inc_agl = float(io_lib.get_param_par(mlipar, 'incidence_angle'))
        pixsp_r = pixsp_r_org/np.sin(np.deg2rad(inc_agl))

    ### Set n_unw_r_thre and cycle depending on L- or C-band
    if wavelength > 0.2: ## L-band
        if args.n_unw_r_thre is None: n_unw_r_thre = 0.5
        else: n_unw_r_thre = args.n_unw_r_thre
        cycle = 1.5 # 2pi/cycle for comparison png
    elif wavelength <= 0.2: ## C-band
        if args.n_unw_r_thre is None: n_unw_r_thre = 1.0
        else: n_unw_r_thre = args.n_unw_r_thre
        cycle = 3 # 3*2pi/cycle for comparison png

    bad_ifg11file = os.path.join(infodir, '11bad_ifg.txt')
    bad_ifg12file = os.path.join(infodir, '12bad_ifg.txt')

    ### Read bad_ifg11 and 12
    if os.path.exists(bad_ifg11file):
        bad_ifg11 = io_lib.read_ifg_list(bad_ifg11file)
    else:
        bad_ifg11 = []
    if os.path.exists(bad_ifg12file):
        bad_ifg12 = io_lib.read_ifg_list(bad_ifg12file)
    else:
        bad_ifg12 = []
    bad_ifg_all = list(set(bad_ifg11+bad_ifg12))
    bad_ifg_all.sort()

    #%% Read date and network information
    ### Get all ifgdates in ifgdir
    if args.ifg_list:
        ifgdates = io_lib.read_ifg_list(args.ifg_list)
    else:
        ifgdates_all = tools_lib.get_ifgdates(ifgdir)
        ### Remove bad ifgs and images from list
        ifgdates = list(set(ifgdates_all)-set(bad_ifg_all))
        ifgdates.sort()

    imdates = tools_lib.ifgdates2imdates(ifgdates)
    n_ifg = len(ifgdates)
    n_im = len(imdates)
    n_unw_thre = int(n_unw_r_thre*n_im)

    ### Make 13used_image.txt
    with open(imfile, 'w') as f:
        for i in imdates:
            print('{}'.format(i), file=f)

    ### Calc dt in year
    imdates_dt = ([dt.datetime.strptime(imd, '%Y%m%d').toordinal() for imd in imdates])
    dt_cum = np.float32((np.array(imdates_dt)-imdates_dt[0])/365.25)

    ### Construct G and Aloop matrix for increment and n_gap
    G = inv_lib.make_sb_matrix(ifgdates)
    Aloop = loop_lib.make_loop_matrix(ifgdates)

    #%% Plot network
    ## Read bperp data or dummy
    bperp_file = os.path.join(ccdir, 'baselines')
    if os.path.exists(bperp_file):
        bperp = io_lib.read_bperp_file(bperp_file, imdates)
    else: #dummy
        bperp = np.random.random(n_im).tolist()


    #%% Get patch row number
    ### Check RAM
    mem_avail = (psutil.virtual_memory().available)/2**20 #MB
    if args.memory_size > mem_avail/2:
        print('\nNot enough memory available compared to mem_size ({} MB).'.format(args.memory_size))
        print('Reduce mem_size automatically to {} MB.'.format(int(mem_avail/2)))
        memory_size = int(mem_avail/2)
    else:
        memory_size = args.memory_size

    ### Determine if read cum on memory (fast) or hdf5 (slow)
    cum_size = int(n_im*length*width*4/2**20) #MB
    if memory_size > cum_size*2:
        print('Read cum data on memory (fast but need memory).')
        save_mem = False # read on memory
        memory_size_patch = memory_size - cum_size
    else:
        print('Read cum data in HDF5 (save memory but slow).')
        save_mem = True # read on hdf5
        memory_size_patch = memory_size

    if args.inv_alg == 'WLS':
        n_store_data = n_ifg*3+n_im*2+n_im*0.3 #
    else:
        n_store_data = n_ifg*2+n_im*2+n_im*0.3 #not sure

    n_patch, patchrow = tools_lib.get_patchrow(width, length, n_store_data, memory_size_patch)


    #%% Display and output settings & parameters
    print('')
    print('Size of image (w,l)    : {}, {}'.format(width, length))
    print('# of images to be used : {}'.format(n_im))
    print('# of ifgs to be used   : {}'.format(n_ifg))
    print('Threshold of used unw  : {}'.format(n_unw_thre))
    print('')
    print('Reference area (X/Y)   : {}:{}/{}:{}'.format(refx1, refx2, refy1, refy2))
    print('Allowed memory size    : {} MB'.format(memory_size))
    print('Number of patches      : {}'.format(n_patch))
    print('Inversion algorism     : {}'.format(args.inv_alg))
    print('Gamma value            : {}'.format(args.gamma), flush=True)

    with open(os.path.join(infodir, '13parameters.txt'), "w") as f:
        print('range_samples:  {}'.format(width), file=f)
        print('azimuth_lines:  {}'.format(length), file=f)
        print('wavelength:     {}'.format(wavelength), file=f)
        print('n_im:           {}'.format(n_im), file=f)
        print('n_ifg:          {}'.format(n_ifg), file=f)
        print('n_unw_thre:     {}'.format(n_unw_thre), file=f)
        print('ref_area:       {}:{}/{}:{}'.format(refx1, refx2, refy1, refy2), file=f)
        print('memory_size:    {} MB'.format(memory_size), file=f)
        print('n_patch:        {}'.format(n_patch), file=f)
        print('inv_alg:        {}'.format(args.inv_alg), file=f)
        print('gamma:          {}'.format(args.gamma), file=f)
        print('pixel_spacing_r: {:.2f} m'.format(pixsp_r), file=f)
        print('pixel_spacing_a: {:.2f} m'.format(pixsp_a), file=f)


    #%% Ref phase for inversion
    lengththis = refy2-refy1
    countf = width*refy1
    countl = width*lengththis # Number to be read
    ref_unw = []
    for i, ifgd in enumerate(ifgdates):
        unwfile = os.path.join(ifgdir, ifgd, ifgd+'.unw')
        f = open(unwfile, 'rb')
        f.seek(countf*4, os.SEEK_SET) #Seek for >=2nd path, 4 means byte

        ### Read unw data (mm) at ref area
        unw = np.fromfile(f, dtype=np.float32, count=countl).reshape((lengththis, width))*coef_r2m
        unw[unw == 0] = np.nan
        if np.isnan(unw[:, refx1:refx2]).all():
            if not os.path.exists(os.path.join(tsadir,'12nan_reference')):
                os.mkdir(os.path.join(tsadir,'12nan_reference'))
            preview_png = os.path.join(tsadir,'12nan_reference', ifgd+'.png')
            print('Median reference due to NaN ref in {}.'.format(ifgd))
            plot_lib.make_im_png(unw, preview_png, cmap_vel, ifgd, -wavelength / 2 * 1000, wavelength / 2 * 1000, ref_window=[refx1, refx2, refy1, refy2])

            unw = np.nanmedian(unw)
        else:
            unw = unw[:, refx1:refx2]

        ref_unw.append(np.nanmean(unw))

        f.close()


    #%% Open cum.h5 for output
    if os.path.exists(cumh5file): os.remove(cumh5file)
    cumh5 = h5.File(cumh5file, 'w')
    cumh5.create_dataset('imdates', data=[np.int32(imd) for imd in imdates])
    if not np.all(np.abs(np.array(bperp))<=1):# if not dummy
        cumh5.create_dataset('bperp', data=bperp)
    gap = cumh5.require_dataset('gap', (n_im-1, length, width),
                                dtype=np.int8, compression=compress)
    if save_mem:
        cum = cumh5.require_dataset('cum', (n_im, length, width),
                                    dtype=np.float32, compression=compress)
        vel = cumh5.require_dataset('vel', (length, width),
                                    dtype=np.float32, compression=compress)
        vconst = cumh5.require_dataset('vintercept', (length, width),
                                       dtype=np.float32, compression=compress)
    else:
        cum = np.zeros((n_im, length, width), dtype=np.float32)
        vel = np.zeros((length, width), dtype=np.float32)
        vconst = np.zeros((length, width), dtype=np.float32)

    if width == width_geo and length == length_geo: ## if geocoded
        cumh5.create_dataset('corner_lat', data=lat1)
        cumh5.create_dataset('corner_lon', data=lon1)
        cumh5.create_dataset('post_lat', data=dlat)
        cumh5.create_dataset('post_lon', data=dlon)


    #%% For each patch
    for i_patch, rows in enumerate(patchrow):
        print('\nProcess {0}/{1}th line ({2}/{3}th patch)...'.format(rows[1], patchrow[-1][-1], i_patch+1, n_patch), flush=True)
        start2 = time.time()

        #%% Read data
        ### Allocate memory
        lengththis = rows[1] - rows[0]
        n_pt_all = lengththis*width
        unwpatch = np.zeros((n_ifg, lengththis, width), dtype=np.float32)

        if args.inv_alg == 'WLS':
            cohpatch = np.zeros((n_ifg, lengththis, width), dtype=np.float32)

        ### For each ifg
        print("  Reading {0} ifg's unw data...".format(n_ifg), flush=True)
        countf = width*rows[0]
        countl = width*lengththis
        for i, ifgd in enumerate(ifgdates):
            unwfile = os.path.join(ifgdir, ifgd, ifgd+'.unw')
            f = open(unwfile, 'rb')
            f.seek(countf*4, os.SEEK_SET) #Seek for >=2nd patch, 4 means byte

            ### Read unw data (mm) at patch area
            unw = np.fromfile(f, dtype=np.float32, count=countl).reshape((lengththis, width))*coef_r2m
            unw[unw == 0] = np.nan # Fill 0 with nan
            unw = unw - ref_unw[i]
            unwpatch[i] = unw
            f.close()

            ### Read coh file at patch area for WLS
            if args.inv_alg == 'WLS':
                cohfile = os.path.join(ccdir, ifgd, ifgd+'.cc')
                f = open(cohfile, 'rb')

                if os.path.getsize(cohfile) == length*width: ## uint8 format
                    f.seek(countf, os.SEEK_SET) #Seek for >=2nd patch
                    cohpatch[i, :, :] = (np.fromfile(f, dtype=np.uint8, count=countl).reshape((lengththis, width))).astype(np.float32)/255
                else: ## old float32 format
                    f.seek(countf*4, os.SEEK_SET) #Seek for >=2nd patch, 4 means byte
                    cohpatch[i, :, :] = np.fromfile(f, dtype=np.float32, count=countl).reshape((lengththis, width))
                cohpatch[cohpatch==0] = np.nan

        unwpatch = unwpatch.reshape((n_ifg, n_pt_all)).transpose() #(n_pt_all, n_ifg)

        ### Calc variance from coherence for WLS
        if args.inv_alg == 'WLS':
            cohpatch = cohpatch.reshape((n_ifg, n_pt_all)).transpose() #(n_pt_all, n_ifg)
            cohpatch[cohpatch<0.01] = 0.01 ## because negative value possible due to geocode
            cohpatch[cohpatch>0.99] = 0.99 ## because >1 possible due to geocode
            varpatch = (1-cohpatch**2)/(2*cohpatch**2)
            del cohpatch


        #%% Remove points with less valid data than n_unw_thre
        ix_unnan_pt = np.where(np.sum(~np.isnan(unwpatch), axis=1) > n_unw_thre)[0]
        n_pt_unnan = len(ix_unnan_pt)
        unwpatch = unwpatch[ix_unnan_pt,:] ## keep only unnan data
        if args.inv_alg == 'WLS':
            varpatch = varpatch[ix_unnan_pt,:] ## keep only unnan data

        print('  {}/{} points removed due to not enough ifg data...'.format(n_pt_all-n_pt_unnan, n_pt_all), flush=True)


        #%% Compute number of gaps, ifg_noloop, maxTlen point-by-point
        if n_pt_unnan != 0:
            ns_gap_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan
            gap_patch = np.zeros((n_im-1, n_pt_all), dtype=np.int8)
            ns_ifg_noloop_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan
            maxTlen_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan


            if args.null_noloop:
                print('\n  Not including IFG pixels with no loops in the inversion, but not nullifying original dataset')
                ### Determine n_para
                n_pt_patch_min = 1000
                if n_pt_patch_min*n_para > n_pt_unnan:
                    ## Too much n_para
                    n_para_gap = int(np.floor(n_pt_unnan/n_pt_patch_min))
                    if n_para_gap == 0: n_para_gap = 1
                else:
                    n_para_gap = n_para

                print('  with {} parallel processing...'.format(n_para_gap),
                      flush=True)

                ### Devide unwpatch by n_para for parallel processing
                p = q.Pool(n_para_gap)
                _result = np.array(p.map(null_noloop_wrapper, range(n_para_gap)),
                                   dtype=object)
                p.close()

                print('Cropping Velocity Inversion to Only Pixels in Loops')
                no_loop_log = np.hstack(_result[:, 3]).T
                del _result

                unwpatch[np.where(no_loop_log == 1)] = np.nan
                print('Removed {} IFG no_loop pixels'.format(np.sum((no_loop_log == 1).flatten() > 0)))

                ix_unnan_pt_noLoop = np.where(np.sum(~np.isnan(unwpatch), axis=1) > 0)[0]
                n_pt_unnan = len(ix_unnan_pt_noLoop)
                unwpatch = unwpatch[ix_unnan_pt_noLoop, :] ## keep only unnan data
                varpatch = varpatch[ix_unnan_pt_noLoop, :]
                ix_unnan_pt = ix_unnan_pt[ix_unnan_pt_noLoop]


            print('\n  Identifing gaps, and counting n_gap and n_ifg_noloop,')

            if args.gpu:
                print('  using GPU...', flush=True)
                n_loop, _ = Aloop.shape
                unwpatch_cp = cp.asarray(unwpatch)
                G_cp = cp.asarray(G)
                Aloop_cp = cp.asarray(Aloop)

                ns_unw_unnan4inc = cp.array(
                    [(G_cp[:, i]*(~cp.isnan(unwpatch_cp))).sum(
                        axis=1, dtype=cp.int16) for i in range(n_im-1)])
                # n_ifg*(n_pt,n_ifg) -> (n_im-1,n_pt)
                ns_gap_patch[ix_unnan_pt] = cp.asnumpy(
                    (ns_unw_unnan4inc==0).sum(axis=0)) #n_pt
                gap_patch[:, ix_unnan_pt] = cp.asnumpy(ns_unw_unnan4inc==0)

                del ns_unw_unnan4inc
                del G_cp

                ### n_ifg_noloop
                # n_ifg*(n_pt,n_ifg)->(n_loop,n_pt)
                # Number of ifgs for each loop at eath point.
                # 3 means complete loop, 1 or 2 means broken loop.
                ns_ifg4loop = cp.array([
                        (cp.abs(Aloop_cp[i, :])*(~cp.isnan(unwpatch_cp))).sum(axis=1)
                        for i in range(n_loop)])
                bool_loop = (ns_ifg4loop==3)
                #(n_loop,n_pt) identify complete loop only

                # n_loop*(n_loop,n_pt)*n_pt->(n_ifg,n_pt)
                # Number of loops for each ifg at eath point.
                ns_loop4ifg = cp.array([(
                        (cp.abs(Aloop_cp[:, i])*bool_loop.T).T*
                        (~cp.isnan(unwpatch_cp[:, i]))
                        ).sum(axis=0) for i in range(n_ifg)]) #

                ns_ifg_noloop_tmp = (ns_loop4ifg==0).sum(axis=0) #n_pt
                ns_nan_ifg = cp.isnan(unwpatch_cp).sum(axis=1) #n_pt, nan ifg count
                ns_ifg_noloop_patch[ix_unnan_pt] = cp.asnumpy(
                    ns_ifg_noloop_tmp - ns_nan_ifg)

                del bool_loop, ns_ifg4loop, ns_loop4ifg
                del ns_ifg_noloop_tmp, ns_nan_ifg
                del unwpatch_cp, Aloop_cp

            else:
                ### Determine n_para
                n_pt_patch_min = 1000
                if n_pt_patch_min*n_para > n_pt_unnan:
                    ## Too much n_para
                    n_para_gap = int(np.floor(n_pt_unnan/n_pt_patch_min))
                    if n_para_gap == 0: n_para_gap = 1
                else:
                    n_para_gap = n_para

                print('  with {} parallel processing...'.format(n_para_gap),
                      flush=True)

                ### Devide unwpatch by n_para for parallel processing
                p = q.Pool(n_para_gap)
                _result = np.array(p.map(count_gaps_wrapper, range(n_para_gap)),
                                   dtype=object)
                p.close()

                ns_gap_patch[ix_unnan_pt] = np.hstack(_result[:, 0]) #n_pt
                gap_patch[:, ix_unnan_pt] = np.hstack(_result[:, 1]) #n_im-1, n_pt
                ns_ifg_noloop_patch[ix_unnan_pt] = np.hstack(_result[:, 2])


            ### maxTlen
            _maxTlen = np.zeros((n_pt_unnan), dtype=np.float32) #temporaly
            _Tlen = np.zeros((n_pt_unnan), dtype=np.float32) #temporaly
            for imx in range(n_im-1):
                _Tlen = _Tlen + (dt_cum[imx+1]-dt_cum[imx]) ## Adding dt
                _Tlen[gap_patch[imx, ix_unnan_pt]==1] = 0 ## reset to 0 if gap
                _maxTlen[_maxTlen<_Tlen] = _Tlen[_maxTlen<_Tlen] ## Set Tlen to maxTlen
            maxTlen_patch[ix_unnan_pt] = _maxTlen


            #%% Time series inversion
            if not args.no_inversion:
                print('\n  Small Baseline inversion by {}...\n'.format(args.inv_alg), flush=True)
                if args.inv_alg == 'WLS':
                    inc_tmp, vel_tmp, vconst_tmp = inv_lib.invert_nsbas_wls(
                        unwpatch, varpatch, G, dt_cum, args.gamma, n_para_inv)
                else:
                    inc_tmp, vel_tmp, vconst_tmp = inv_lib.invert_nsbas(
                        unwpatch, G, dt_cum, args.gamma, n_para_inv, args.gpu, fast=args.fast, only_sb=args.only_sb)

                ### Set to valuables
                inc_patch = np.zeros((n_im-1, n_pt_all), dtype=np.float32)*np.nan
                vel_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan
                vconst_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan

                inc_patch[:, ix_unnan_pt] = inc_tmp
                vel_patch[ix_unnan_pt] = vel_tmp
                vconst_patch[ix_unnan_pt] = vconst_tmp

                ### Calculate "Ideal" Interferogram (Jack)
                ifg_patch = np.zeros((n_ifg, n_pt_all), dtype=np.float32)*np.nan
                ifg_patch[:, ix_unnan_pt] = np.dot(G, inc_tmp)

                ### Calculate residuals
                res_patch = np.zeros((n_ifg, n_pt_all), dtype=np.float32)*np.nan
                res_patch[:, ix_unnan_pt] = unwpatch.T-np.dot(G, inc_tmp)

                res_sumsq = np.nansum(res_patch**2, axis=0)
                res_n = np.float32((~np.isnan(res_patch)).sum(axis=0))
                res_n[res_n==0] = np.nan # To avoid 0 division
                res_rms_patch = np.sqrt(res_sumsq/res_n)

                ### Cumulative displacememt
                cum_patch = np.zeros((n_im, n_pt_all), dtype=np.float32)*np.nan
                cum_patch[1:, :] = np.cumsum(inc_patch, axis=0)

                ## Fill 1st image with 0 at unnan points from 2nd images
                bool_unnan_pt = ~np.isnan(cum_patch[1, :])
                cum_patch[0, bool_unnan_pt] = 0

                ## Drop (fill with nan) interpolated cum by 2 continuous gaps
                for i in range(n_im-2): ## from 1->n_im-1
                    gap2 = gap_patch[i, :]+gap_patch[i+1, :]
                    bool_gap2 = (gap2==2) ## true if 2 continuous gaps for each point
                    cum_patch[i+1, :][bool_gap2] = np.nan

                ## Last (n_im th) image. 1 gap means interpolated
                cum_patch[-1, :][gap_patch[-1, :]==1] = np.nan

            else:
                print('\n  Not running SB inversion...\n', flush=True)
                cum_patch = np.zeros((n_im, n_pt_all), dtype=np.float32)*np.nan
                vel_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan
                vconst_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan
                inc_patch = np.zeros((n_im-1, n_pt_all), dtype=np.float32)*np.nan
                res_patch = np.zeros((n_ifg, n_pt_all), dtype=np.float32)*np.nan
                ifg_patch = np.zeros((n_ifg, n_pt_all), dtype=np.float32)*np.nan
                res_rms_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan


        #%% Fill by np.nan if n_pt_unnan == 0
        else:
            cum_patch = np.zeros((n_im, n_pt_all), dtype=np.float32)*np.nan
            vel_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan
            vconst_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan
            gap_patch = np.zeros((n_im-1, n_pt_all), dtype=np.int8)
            inc_patch = np.zeros((n_im-1, n_pt_all), dtype=np.float32)*np.nan
            res_patch = np.zeros((n_ifg, n_pt_all), dtype=np.float32)*np.nan
            ifg_patch = np.zeros((n_ifg, n_pt_all), dtype=np.float32)*np.nan
            res_rms_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan
            ns_gap_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan
            ns_ifg_noloop_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan
            maxTlen_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan


        #%% Output data and image
        ### cum.h5 file
        cum[:, rows[0]:rows[1], :] = cum_patch.reshape((n_im, lengththis, width))
        vel[rows[0]:rows[1], :] = vel_patch.reshape((lengththis, width))
        vconst[rows[0]:rows[1], :] = vconst_patch.reshape((lengththis, width))
        gap[:, rows[0]:rows[1], :] = gap_patch.reshape((n_im-1, lengththis, width))

        ### Others
        openmode = 'w' if rows[0] == 0 else 'a' #w only 1st patch

        ## For each imd. cum and inc
        for imx, imd in enumerate(imdates):
            ## Incremental displacement
            if imd == imdates[-1]: continue #skip last
            incfile = os.path.join(incdir, '{0}_{1}.inc'.format(imd, imdates[imx+1]))
            with open(incfile, openmode) as f:
                inc_patch[imx, :].tofile(f)

        ## For each ifgd. resid
        for i, ifgd in enumerate(ifgdates):
            resfile = os.path.join(resdir, '{0}.res'.format(ifgd))
            with open(resfile, openmode) as f:
                res_patch[i, :].tofile(f)
            ifgfile = os.path.join(resdir, '{0}.ifg'.format(ifgd))
            with open(ifgfile, openmode) as f:
                ifg_patch[i, :].tofile(f)

        ## velocity and noise indecies in results dir
        if not args.no_loops_removed:
            names = ['vel', 'vintercept', 'resid_rms', 'n_gap', 'n_ifg_noloop', 'maxTlen']
        else:
            names = ['vel', 'vintercept', 'resid_rms', 'n_gap', 'n_ifg_noloop_postNullNoLoop', 'maxTlen']
        data = [vel_patch, vconst_patch, res_rms_patch, ns_gap_patch, ns_ifg_noloop_patch, maxTlen_patch]
        for i in range(len(names)):
            file = os.path.join(resultsdir, names[i])
            with open(file, openmode) as f:
                data[i].tofile(f)


        #%% Finish patch
        elapsed_time2 = int(time.time()-start2)
        hour2 = int(elapsed_time2/3600)
        minite2 = int(np.mod((elapsed_time2/60),60))
        sec2 = int(np.mod(elapsed_time2,60))
        print("  Elapsed time for {0}th patch: {1:02}h {2:02}m {3:02}s".format(i_patch+1, hour2, minite2, sec2), flush=True)


    # #%% Find stable ref point
    # print('\nFind stable reference point...', flush=True)
    # ### Compute RMS of time series with reference to all points
    # sumsq_cum_wrt_med = np.zeros((length, width), dtype=np.float32)
    # for i in range(n_im):
    #     sumsq_cum_wrt_med = sumsq_cum_wrt_med + (cum[i, :, :]-np.nanmedian(cum[i, :, :]))**2
    # rms_cum_wrt_med = np.sqrt(sumsq_cum_wrt_med/n_im)
    #
    # ### Mask by minimum n_gap
    # n_gap = io_lib.read_img(os.path.join(resultsdir, 'n_gap'), length, width)
    # min_n_gap = np.nanmin(n_gap)
    # mask_n_gap = np.float32(n_gap==min_n_gap)
    # mask_n_gap[mask_n_gap==0] = np.nan
    # rms_cum_wrt_med = rms_cum_wrt_med*mask_n_gap
    #
    # ### Save image
    # with open(rms_cum_wrt_med_file, 'w') as f:
    #     rms_cum_wrt_med.tofile(f)
    # plot_lib.make_im_png(rms_cum_wrt_med, rms_cum_png, cmap_noise_r, 'RMS of cum wrt median (mm)', np.nanpercentile(rms_cum_wrt_med, 1), np.nanpercentile(rms_cum_wrt_med, 99))
    #
    # ### Find stable reference
    # min_rms = np.nanmin(rms_cum_wrt_med)
    # refy1s, refx1s = np.where(rms_cum_wrt_med==min_rms)
    # refy1s, refx1s = refy1s[0], refx1s[0] ## Only first index
    # refy2s, refx2s = refy1s+1, refx1s+1
    # print('Selected ref: {}:{}/{}:{}'.format(refx1s, refx2s, refy1s, refy2s), flush=True)
    # if width == width_geo and length == length_geo: ## Geocoded
    #     ### Make ref_stable.kml
    #     reflat = lat1+dlat*refy1s
    #     reflon = lon1+dlon*refx1s
    #     io_lib.make_point_kml(reflat, reflon, refkml)

    # ### Save ref
    # cumh5.create_dataset('refarea', data='{}:{}/{}:{}'.format(refx1s, refx2s, refy1s, refy2s))
    # with open(refsfile, 'w') as f:
    #     print('{}:{}/{}:{}'.format(refx1s, refx2s, refy1s, refy2s), file=f)

    # ### Referencing cumulative displacement and vel to new stable ref
    # for i in range(n_im):
    #     cum[i, :, :] = cum[i, :, :] - cum[i, refy1s, refx1s]
    # vel = vel - vel[refy1s, refx1s]
    # vconst = vconst - vconst[refy1s, refx1s]

    ### Save ref
    cumh5.create_dataset('refarea', data='{}:{}/{}:{}'.format(refx1, refx2, refy1, refy2))

    #%% Close h5 file
    if not save_mem:
        print('\nWriting to HDF5 file...')
        cumh5.create_dataset('cum', data=cum, compression=compress)
        cumh5.create_dataset('vel', data=vel, compression=compress)
        cumh5.create_dataset('vintercept', data=vconst, compression=compress)
    cumh5.close()


    #%% Output png images
    ### Incremental displacement
    if args.nopngs or args.no_inversion:
        print('skipping generating additional png images of increments and residuals - as sometimes taking too long (tutorial purposes)')
    else:
        _n_para = n_im-1 if n_para > n_im-1 else n_para
        print('\nOutput increment png images with {} parallel processing...'.format(_n_para), flush=True)
        p = q.Pool(_n_para)
        p.map(inc_png_wrapper, range(n_im-1))
        p.close()

        ### Residual for each ifg. png and txt.
        with open(restxtfile, "w") as f:
            print('# RMS of residual (mm)', file=f)
        _n_para = n_ifg if n_para > n_ifg else n_para
        print('\nOutput residual png images with {} parallel processing...'.format(_n_para), flush=True)
        p = q.Pool(_n_para)
        p.map(resid_png_wrapper, range(n_ifg))
        p.close()

    ### Velocity and noise indices
    cmins = [None, None, None, None, None, None]
    cmaxs = [None, None, None, None, None, None]
    cmaps = [cmap_vel, cmap_vel, cmap_noise_r, cmap_noise_r, cmap_noise_r, cmap_noise]
    titles = ['Velocity (mm/yr)', 'Intercept of velocity (mm)', 'RMS of residual (mm)', 'Number of gaps in SB network', 'Number of ifgs with no loops', 'Max length of connected SB network (yr)']

    print('\nOutput noise png images...', flush=True)
    for i in range(len(names)):
        file = os.path.join(resultsdir, names[i])
        data = io_lib.read_img(file, length, width)

        pngfile = file+'.png'

        ## Get color range if None
        if cmins[i] is None:
            cmins[i] = np.nanpercentile(data, 1)
        if cmaxs[i] is None:
            cmaxs[i] = np.nanpercentile(data, 99)
        if cmins[i] == cmaxs[i]: cmins[i] = cmaxs[i]-1
        # print(pngfile)
        plot_lib.make_im_png(data, pngfile, cmaps[i], titles[i], cmins[i], cmaxs[i])

    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(sys.argv[0])))
    print('Output directory: {}\n'.format(os.path.relpath(tsadir)))


def count_gaps_wrapper(i):
    print("    Running {:2}/{:2}th patch...".format(i+1, n_para_gap), flush=True)
    n_pt_patch = int(np.ceil(unwpatch.shape[0]/n_para_gap))
    n_im = G.shape[1]+1
    n_loop, n_ifg = Aloop.shape

    if i*n_pt_patch >= unwpatch.shape[0]:
        # Nothing to do
        return

    ### n_gap and gap location
    ns_unw_unnan4inc = np.array([(G[:, j]*
                          (~np.isnan(unwpatch[i*n_pt_patch:(i+1)*n_pt_patch])))
                         .sum(axis=1, dtype=np.int16) for j in range(n_im-1)])
                    #n_ifg*(n_pt,n_ifg) -> (n_im-1,n_pt)
    _ns_gap_patch = (ns_unw_unnan4inc==0).sum(axis=0) #n_pt
    _gap_patch = ns_unw_unnan4inc==0

    del ns_unw_unnan4inc

    ### n_ifg_noloop
    # n_ifg*(n_pt,n_ifg)->(n_loop,n_pt)
    # Number of ifgs for each loop at eath point.
    # 3 means complete loop, 1 or 2 means broken loop.
    ns_ifg4loop = np.array([(np.abs(Aloop[j, :])*
                         (~np.isnan(unwpatch[i*n_pt_patch:(i+1)*n_pt_patch])))
                            .sum(axis=1) for j in range(n_loop)])
    bool_loop = (ns_ifg4loop==3) #(n_loop,n_pt) identify complete loop only
    bad_bool_loop = (ns_ifg4loop != 3) # Identify incomplete loops (Loop x Pixel)
    del ns_ifg4loop

    # n_loop*(n_loop,n_pt)*n_pt->(n_ifg,n_pt)
    # Number of loops for each ifg at eath point.
    ns_loop4ifg = np.array([(
            (np.abs(Aloop[:, j])*bool_loop.T).T*
            (~np.isnan(unwpatch[i*n_pt_patch:(i+1)*n_pt_patch, j]))
            ).sum(axis=0) for j in range(n_ifg)]) #    <= This is the variable that contains the number of loops for each IFG for each pixel
    del bool_loop

    ns_ifg_noloop_tmp = (ns_loop4ifg==0).sum(axis=0) # Number of incomplete loops per pixel

    del ns_loop4ifg

    ns_nan_ifg = np.isnan(unwpatch[i*n_pt_patch:(i+1)*n_pt_patch, :]).sum(axis=1)
    #n_pt, nan ifg count
    _ns_ifg_noloop_patch = ns_ifg_noloop_tmp - ns_nan_ifg # IFGs with no loop = IFGs with no loop - IFGs that are NaNs anyway (number no_loop per pixel)

    return _ns_gap_patch, _gap_patch, _ns_ifg_noloop_patch

def null_noloop_wrapper(i):
    print("    Running {:2}/{:2}th patch...".format(i+1, n_para_gap), flush=True)
    n_pt_patch = int(np.ceil(unwpatch.shape[0]/n_para_gap))
    n_im = G.shape[1]+1
    n_loop, n_ifg = Aloop.shape

    if i*n_pt_patch >= unwpatch.shape[0]:
        # Nothing to do
        return

    ### n_gap and gap location
    ns_unw_unnan4inc = np.array([(G[:, j]*
                          (~np.isnan(unwpatch[i*n_pt_patch:(i+1)*n_pt_patch])))
                         .sum(axis=1, dtype=np.int16) for j in range(n_im-1)])
                    #n_ifg*(n_pt,n_ifg) -> (n_im-1,n_pt)
    _ns_gap_patch = (ns_unw_unnan4inc==0).sum(axis=0) #n_pt
    _gap_patch = ns_unw_unnan4inc==0

    del ns_unw_unnan4inc

    ### n_ifg_noloop
    # n_ifg*(n_pt,n_ifg)->(n_loop,n_pt)
    # Number of ifgs for each loop at eath point.
    # 3 means complete loop, 1 or 2 means broken loop.
    ns_ifg4loop = np.array([(np.abs(Aloop[j, :])*
                         (~np.isnan(unwpatch[i*n_pt_patch:(i+1)*n_pt_patch])))
                            .sum(axis=1) for j in range(n_loop)])
    bool_loop = (ns_ifg4loop==3) #(n_loop,n_pt) identify complete loop only
    bad_bool_loop = (ns_ifg4loop != 3) # Identify incomplete loops (Loop x Pixel)
    # if i == 0: print(bool_loop.shape)
    # if i == 0:
    #     print(bad_bool_loop)
    del ns_ifg4loop

    # n_loop*(n_loop,n_pt)*n_pt->(n_ifg,n_pt)
    # Number of loops for each ifg at eath point.
    ns_loop4ifg = np.array([(
            (np.abs(Aloop[:, j])*bool_loop.T).T*
            (~np.isnan(unwpatch[i*n_pt_patch:(i+1)*n_pt_patch, j]))
            ).sum(axis=0) for j in range(n_ifg)]) #    <= This is the variable that contains the number of loops for each IFG for each pixel
    del bool_loop
    # if i == 0: print('ns_loop4ifg', ns_loop4ifg.shape)
    # if i == 0: print('ns_loop4ifg', np.unique(ns_loop4ifg, return_counts=True))
    ns_ifg_noloop_tmp = (ns_loop4ifg==0).sum(axis=0) # Number of incomplete loops per pixel
    # if i == 0: print('ns_ifg_noloop_tmp', ns_ifg_noloop_tmp.shape)
    ns_ifg_noloop_ix = (ns_loop4ifg == 0).astype('int') # Int array of n_ifg x n_px of px with no loops
    # if i == 0: print('ns_ifg_noloop_ix', ns_ifg_noloop_ix.shape)
    #del ns_loop4ifg

    ns_nan_ifg = np.isnan(unwpatch[i*n_pt_patch:(i+1)*n_pt_patch, :]).sum(axis=1)
    # if i == 0: print(ns_nan_ifg)
    #n_pt, nan ifg count
    _ns_ifg_noloop_patch = ns_ifg_noloop_tmp - ns_nan_ifg # IFGs with no loop = IFGs with no loop - IFGs that are NaNs anyway (number no_loop per pixel)

    return _ns_gap_patch, _gap_patch, _ns_ifg_noloop_patch, ns_ifg_noloop_ix


def inc_png_wrapper(imx):
    imd = imdates[imx]
    if imd == imdates[-1]:
        return #skip last for increment

    ## Comparison of increment and daisy chain pair
    ifgd = '{}_{}'.format(imd, imdates[imx+1])
    incfile = os.path.join(incdir, '{}.inc'.format(ifgd))
    unwfile = os.path.join(ifgdir, ifgd, '{}.unw'.format(ifgd))
    pngfile = os.path.join(incdir, '{}.inc_comp.png'.format(ifgd))

    inc = io_lib.read_img(incfile, length, width)

    try:
        unw = io_lib.read_img(unwfile, length, width)*coef_r2m
        ix_ifg = ifgdates.index(ifgd)
        # unw = unw - ref_unw[ix_ifg]
    except:
        unw = np.zeros((length, width), dtype=np.float32)*np.nan

    ### Output png for comparison
    data3 = [np.angle(np.exp(1j*(data/coef_r2m/cycle))*cycle) for data in [unw, inc, inc-unw]]
    title3 = ['Daisy-chain IFG ({}pi/cycle)'.format(cycle*2), 'Inverted ({}pi/cycle)'.format(cycle*2), 'Difference ({}pi/cycle)'.format(cycle*2)]
    pngfile = os.path.join(incdir, '{}.increment.png'.format(ifgd))
    plot_lib.make_3im_png(data3, pngfile, cmap_wrap, title3, vmin=-np.pi, vmax=np.pi, cbar=False)

    if not args.keep_incfile:
        os.remove(incfile)


def resid_png_wrapper(i):
    ifgd = ifgdates[i]
    infile = os.path.join(resdir, '{}.res'.format(ifgd))
    resid = io_lib.read_img(infile, length, width)
    resid_rms = np.sqrt(np.nanmean(resid**2))
    with open(restxtfile, "a") as f:
        print('{} {:5.2f}'.format(ifgd, resid_rms), file=f)

    pngfile = infile+'.png'
    title = 'Residual (mm) of {} (RMS:{:.2f}mm)'.format(ifgd, resid_rms)
    # plot_lib.make_im_png(resid, pngfile, cmap_vel, title, -wavelength/2*1000, wavelength/2*1000)
    if resid_rms < 20:
        plot_lib.make_im_png(resid, pngfile, cmap_vel, title, -wavelength/4*1000, wavelength/4*1000)
    else:
        vlim = np.nanmax([abs(np.nanpercentile(resid, 95)), abs(np.nanpercentile(resid, 5))])
        plot_lib.make_im_png(resid, pngfile, cmap_vel, title, -vlim, vlim)

    ifgfile = os.path.join(resdir, '{}.ifg'.format(ifgd))
    ifg = io_lib.read_img(ifgfile, length, width)
    pngfile = ifgfile+'.png'
    title = 'Inverted {} IFG'.format(ifgd)
    vlim = np.nanmax([abs(np.nanpercentile(ifg, 95)), abs(np.nanpercentile(ifg, 5))])

    plot_lib.make_im_png(ifg, pngfile, cmap_vel, title, -vlim, vlim)

    if not args.keep_incfile:
        os.remove(infile)
        os.remove(ifgfile)


if __name__ == "__main__":
    sys.exit(main())
