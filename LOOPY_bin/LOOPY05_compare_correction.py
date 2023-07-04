#!/usr/bin/env python3
"""
========
Overview
========
Compare Corrections

===============
Input & output files
===============

=====
Exception
=====
LOOPY05_compare_corrections.py [-h] [-f FRAME_DIR] [-i ORIGINAL_DIR] [-c CORRECTED_DIR] [-o COMPARISON_DIR] [-n n_para] [--reset]
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import glob
import argparse
import sys
import time
from pathlib import Path
from matplotlib import cm
import SCM
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib
import shutil
import multiprocessing as multi


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    '''
    Use a multiple inheritance approach to use features of both classes.
    The ArgumentDefaultsHelpFormatter class adds argument default values to the Exception help message
    The RawDescriptionHelpFormatter class keeps the indentation and line breaks in the ___doc___
    '''
    pass


def init_args():
    global args

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=CustomFormatter)
    parser.add_argument('-f', dest='frame_dir', default="./", help="directory of LiCSBAS output of a particular frame")
    parser.add_argument('-i', dest='orig_dir', default="GEOCml10GACOS", help="folder containing the original data")
    parser.add_argument('-c', dest='corr_dir', default="GEOCml10GACOSmerged", help="folder containing the corrected data")
    parser.add_argument('-o', dest='comp_dir', default="comp_GEOCml10GACOS ", help="folder containing the comparisons")
    parser.add_argument('-n', dest='n_para', type=int, help="number of processes for parallel processing")
    parser.add_argument('--reset', dest='reset_corr', default=False, action='store_true', help="overwrite previous comparisons")
    args = parser.parse_args()

def start():
    global start_time
    # intialise and print info on screen
    start_time = time.time()
    ver="1.0"; date=20221020; author="Qi Ou"
    print("\n{} ver{} {} {}".format(os.path.basename(sys.argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)


def finish():
    #%% Finish
    elapsed_time = time.time() - start_time
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))
    print("\n{} {} finished!".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)
    print('Output directory: {}\n'.format(os.path.relpath(compdir)))


def set_input_output():
    global origdir, corrdir, compdir, TSorig, TScorr, comp_dates, uncorr_dates, noorig_dates, compCorr, compTS

    origdir = os.path.join(args.frame_dir, args.orig_dir)
    corrdir = os.path.join(args.frame_dir, args.corr_dir)
    compdir = os.path.join(args.frame_dir, args.comp_dir)

    TSorig = os.path.join(args.frame_dir, 'TS_' + args.orig_dir, 'results')
    TScorr = os.path.join(args.frame_dir, 'TS_' + args.corr_dir, 'results')

    compCorr = True
    compTS = True

    for dir in [origdir, corrdir]:
        if not os.path.exists(dir):
            if compCorr:
                compCorr = False
                print('Not comparing IFGs: ')
            print('\t{} does not exist!'.format(dir))

    for dir in [TSorig, TScorr]:
        if not os.path.exists(dir):
            if compTS:
                compTS = False
                print('Not comparing Time Series: ')
            print('\t{} does not exist!'.format(dir))

    if not compCorr and not compTS:
        raise Exception('Too many missing directories....')

    if os.path.exists(compdir):
        if args.reset_corr:
            print('Removing previous comparison...')
            shutil.rmtree(compdir)
        else:
            raise Exception('{} exists! Use --reset to overwrite'.format(args.comp_dir))

    os.mkdir(compdir)
    if compCorr:
        os.mkdir(os.path.join(compdir, 'IFGs'))
    if compTS:
        os.mkdir(os.path.join(compdir, 'results'))

    origdates = tools_lib.get_ifgdates(origdir)
    corrdates = tools_lib.get_ifgdates(corrdir)

    if len(origdates) == 0:
        raise Exception('No IFGs in {}'.format(args.orig_dir))

    if len(corrdates) == 0:
        raise Exception('No IFGs in {}'.format(args.corr_dir))

    uncorr_dates = list(set(origdates) - set(corrdates))
    noorig_dates = list(set(corrdates) - set(origdates))
    comp_dates = list(set(origdates) - set(uncorr_dates))

    if len(uncorr_dates) == len(origdates):
        raise Exception('No IFGs in {} have corrections in {}'.format(args.orig_dir, args.corr_dir))

    if len(noorig_dates) == len(corrdates):
        raise Exception('No Corrected IFGs in {} have originals in {}'.format(args.corr_dir, args.orig_dir))

    compinfoFile = os.path.join(compdir, 'compInfo.txt')
    with open(compinfoFile, 'w') as f:
        if compCorr:
            print('Comparing IFGs:')
            print('IFG Dataset 1: {}'.format(os.path.basename(origdir)), file=f)
            print('IFG Dataset 2: {}'.format(os.path.basename(corrdir)), file=f)
        if compTS:
            print('Comparing Timeseries:')
            print('TS Dataset 1: {}'.format('TS_' + os.path.basename(origdir)), file=f)
            print('TS Dataset 2: {}'.format('TS_' + os.path.basename(corrdir)), file=f)


def get_para():
    global width, length, n_para

    # read ifg size and satellite frequency
    mlipar = os.path.join(origdir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))

    # multi-processing
    if not args.n_para:
        try:
            n_para = min(len(os.sched_getaffinity(0)), 8) # maximum use 8 cores
        except:
            n_para = multi.cpu_count()
    else:
        n_para = args.n_para

def even_split(a, n):
    """ Divide a list, a, in to n even parts"""
    n = min(n, len(a)) # to avoid empty lists
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def compare_corrections():
    if compCorr:
        print('Comparing Interferograms...')
        # parallel processing
        if n_para > 1 and len(comp_dates) > 20:
            pool = multi.Pool(processes=n_para)
            pool.map(comp_ifg, even_split(comp_dates, n_para))
        else:
            comp_ifg(comp_dates)

    if compTS:
        resOrig = glob.glob(os.path.join(TSorig, '*.png'))

        print('Comparing Results....')
        print('\tSearching {} for results files'.format(TSorig))
        skip_files = ['hgt', 'slc.mli']
        vel_files = ['vel', 'vel.mskd', 'vintercept', 'vintercept.mskd']

        for res in resOrig:
            file = os.path.basename(res)[:-4]
            if file in skip_files:
                continue
            origfile = os.path.join(TSorig, file)
            corrfile = os.path.join(TScorr, file)
            if os.path.exists(origfile) and os.path.exists(corrfile):
                print('\t\t{}'.format(file))
                pngfile = os.path.join(compdir, 'results', file + '.png')
                data1 = io_lib.read_img(origfile, length, width)
                data2 = io_lib.read_img(corrfile, length, width)
                if file in vel_files:
                    plot_comparison(file, data1, data2, pngfile, SCM.roma, ifg=False)
                else:
                    plot_comparison(file, data1, data2, pngfile, 'viridis', ifg=False)
            else:
                print('\t\t{} exists in uncorrected but not in TS_{}'.format(file, args.corr_dir))



def comp_ifg(pairlist):
    for pair in pairlist:
        ogIFG = io_lib.read_img(os.path.join(origdir, pair, pair + '.unw'), length, width)
        ogIFG = ogIFG
        corrIFG = io_lib.read_img(os.path.join(corrdir, pair, pair + '.unw'), length, width)
        corrIFG = corrIFG
        pngfile = os.path.join(compdir, 'IFGs', pair + '.png')

        plot_comparison(pair, ogIFG, corrIFG, pngfile)



def plot_comparison(title, data1, data2, png_path, cmap=cm.RdBu, ifg=True):
    fig, ax = plt.subplots(1, 3, figsize=(9, 5))
    fig.suptitle(title)
    for x in ax:
        x.axes.xaxis.set_ticklabels([])
        x.axes.yaxis.set_ticklabels([])

    if ifg:
        data1ref = np.nanmedian(data1)
        data2ref = np.nanmedian(data2)
        data1 = data1 - data1ref
        data2 = data2 - data2ref
        diff = data1 - data2
        vmin_tmp = np.nanmin([np.nanpercentile(data1, 0.5),np.nanpercentile(data2, 0.5)])
        vmax_tmp = np.nanmax([np.nanpercentile(data1, 99.5),np.nanpercentile(data2, 99.5)])
        vmax = np.nanmax([abs(vmin_tmp), abs(vmax_tmp)])
        vmin = -vmax
        vlim = 2 * np.pi
    else:
        diff = data1 - data2
        vlim = np.nanpercentile(abs(diff), 99.5)
        vmin = np.nanmin([np.nanpercentile(data1, 0.5),np.nanpercentile(data2, 0.5)])
        vmax = np.nanmax([np.nanpercentile(data1, 99.5),np.nanpercentile(data2, 99.5)])

    im1 = ax[0].imshow(data1, vmin=vmin, vmax=vmax, cmap=cmap, interpolation='nearest')
    im2 = ax[1].imshow(data2, vmin=vmin, vmax=vmax, cmap=cmap, interpolation='nearest')
    im3 = ax[2].imshow(diff, vmin=-vlim, vmax=vlim, cmap=cm.RdBu, interpolation='nearest')


    ax[0].set_title("Original")
    ax[1].set_title("Corrected")
    ax[2].set_title("Difference")
    fig.colorbar(im1, ax=ax[0], location='right', shrink=0.8)
    fig.colorbar(im2, ax=ax[1], location='right', shrink=0.8)
    fig.colorbar(im3, ax=ax[2], location='right', shrink=0.8)
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    start()
    init_args()
    set_input_output()
    get_para()

    compare_corrections()

    finish()


if __name__ == "__main__":
    main()
