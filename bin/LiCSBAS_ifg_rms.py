#!/usr/bin/env python3
"""
20220427 Andrew Watson

Calculates the RMS for all interferograms in the provided GEOCml directory

===============
Input & output files
===============
Inputs in GEOCml*/ :
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw

Outputs in GEOCml*/
 - ifg_rms.txt

=====
Usage
=====
LiCSBAS_ifg_rms.py -i in_dir

 -i  Path to the GEOCml* dir containing stack of unw data
"""
#%% Change log
'''
v1.0 20220427 Andrew Watson, Uni of Leeds
 - Original implementation
'''

#%% Import
import getopt
import os
import sys
import time
import shutil
import glob
import numpy as np
from osgeo import gdal
import multiprocessing as multi
import SCM
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib

class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg

#%% Main
def main(argv=None):

    #%% Check argv
    if argv == None:
        argv = sys.argv

    #%% Set default
    in_dir = []

    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:")
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-i':
                in_dir = a

        if not in_dir:
            raise Usage('No input directory given, -i is not optional!')
    
    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2

    #%% Read data information
    ### Directory
    in_dir = os.path.abspath(in_dir)

    ### Get size
    mlipar = os.path.join(in_dir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))
    print("\nSize         : {} x {}".format(width, length), flush=True)

    ### Get ifgdates and imdates
    ifgdates = tools_lib.get_ifgdates(in_dir)
    imdates = tools_lib.ifgdates2imdates(ifgdates)
    n_ifg = len(ifgdates)
    n_im = len(imdates)

    #%% Calculate RMS
    rmsfile = os.path.join(in_dir, 'ifg_rms.txt')
    if os.path.exists(rmsfile): os.remove(rmsfile)
    
    ### Read data and calculate
    for ifgix, ifgd in enumerate(ifgdates):
        if np.mod(ifgix,100) == 0:
            print("Calculating RMS for {0:3}/{1:3}th unw.".format(ifgix, n_ifg), flush=True)
        unwfile = os.path.join(in_dir, ifgd, ifgd+'.unw')
        unw = io_lib.read_img(unwfile, length, width)

        rms = np.sqrt(np.nanmean(np.square(unw)))
        with open(rmsfile, "a") as f:
            print('{0}  {1:4.1f}'.format(ifgd, rms), file=f)

    f.close()

#%% main
if __name__ == "__main__":
    sys.exit(main())

