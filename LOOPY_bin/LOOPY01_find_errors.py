#!/usr/bin/env python3
"""
========
Overview
========
This script identifies errors in unwrapped interferograms, and aims to correct
them. It must be run before the application of GACOS and LOOPY03 if being used
on the ful resolution data (as recommended)

1) Read in IFG, interpolate to full area, and find modulo 2pi values
2) Carry out modal filtering to reduce noise
3) Find difference between adjacent pixels
4) Classify an error boundary as anywhere between 2 pixels with a > modulo pi
    difference
5) Add unwrapping errors back into original IFG, and re-interpolate to create
    an IFG seperated into distinct regions by unwrapping errors
6) All pixels in the same region as the reference are to be considered good.
    Correct all regions bordering this with a static correction to reduce the
    difference to < 2pi

Limitations:
    Can't identify regions that are correctly unwrapped, but isolated from main
    pixel, either by being an island, or because it's cut-off by another
    unwrapping error (Latter option would require iterations)
    Needs an unwrapping error to be complete enclosed in order to be masked
    ('Twist' errors can't be identified')

Additional modules to LiCSBAS needed:
- scipy
- skimage

===============
Input & output files
===============
If not working at full res:
Inputs in GEOCml*/:
- yyyymmdd_yyyymmdd/
 - yyyymmdd_yyyymmdd.unw
 - yyyymmdd_yyyymmdd.cc
- slc.mli.par

Inputs in TS_GEOCml */:
- results/coh_avg  : Average coherence

If working at full res:
Inputs in GEOCml*/:
- yyyymmdd_yyyymmdd/

Inputs in GEOC/:
- yyyymmdd_yyyymmdd/
 - yyyymmdd_yyyymmdd.geo.unw.tif
 - yyyymmdd_yyyymmdd.geo.cc.tif
- frame.geo.[E, N, U, hgt, mli]
- slc.mli

Outputs in GEOCml*LoopMask/:
- yyyymmdd_yyyymmdd/
  - yyyymmdd_yyyymmdd.unw[.png] : Corrected unw
  - yyyymmdd_yyyymmdd.cc : Coherence file
  - yyyymmdd_yyyymmdd.errormap.png : png of identified error boundaries
  - yyyymmdd_yyyymmdd.npicorr.png : png of original + corrected unw, with npi maps
  - yyyymmdd_yyyymmdd.maskcorr.png : png of original + corrected unw, original npi map, and correction
- known_errors.png : png of the input know error mask
- other metafiles produced by LiCSBAS02_ml_prep.py

=====
Usage
=====
LOOPY01_find_errors.py -d ifgdir [-t tsadir] [-c corrdir] [-m int] [-f int] [-e errorfile] [-v int] [--fullres] [--reset] [--n_para] [--onlylisted] [--autoerror]

-d        Path to the GEOCml* dir containing stack of unw data
-t        Path to the output TS_GEOCml* dir. (Default: TS_GEOCml*)
-c        Path to the correction dierectory (Default: GEOCml*LoopMask)
-m        Output multilooking factor (Default: No multilooking of mask, INCOMPTIBLE WITH FULL RES)
-f        Minimum size of error to correct (Default: 10 pixels at final ML size)
-e        Text file, where each row is a known error location, in form lon1,lat1,....,lonn,latn
-v        IFG to give verbose timings for (Development option, Default: -1 (not verbose))
--fullres Create masks from full res data (ie. orginal geotiffs) (Assume in folder called GEOC)
--reset   Remove previous corrections
--n_para  Number of parallel processing (Default: # of usable CPU)
--autoerror Try and automatically guess where errors will be based on coherence
--onlylisted Use only errors that have been user defined

=========
Changelog
=========
v1.3 20230201 Jack McGrath, Uni of Leeds
- Allow option to mask geotiffs directly
v1.2 20230131 Jack McGrath, Uni of Leeds
- Change masking method to edge detection
v1.1 20220615 Jack McGrath, Uni of Leeds
- Edit to run from command line
v1.0 20220608 Jack McGrath, Uni of Leeds
- Original implementation
"""

import os
import re
import sys
import SCM
import time
import glob
import getopt
import shutil
import numpy as np
import multiprocessing as multi
import LiCSBAS_io_lib as io_lib
import LiCSBAS_plot_lib as plot_lib
import LiCSBAS_tools_lib as tools_lib
import LOOPY_lib as loopy_lib
from osgeo import gdal
from scipy.stats import mode
from scipy.ndimage import label
from scipy.ndimage import binary_dilation, binary_closing
from scipy.interpolate import NearestNDInterpolator
from skimage import filters
from skimage.filters.rank import modal
from skimage.morphology import skeletonize

insar = tools_lib.get_cmap('SCM.romaO')

class Usage(Exception):
    """Usage context manager"""

    def __init__(self, msg):
        self.msg = msg


# %% Main
def main(argv=None):

    # %% Check argv
    if argv is None:
        argv = sys.argv

    start = time.time()
    ver = "1.3.0"; date = 20230201; author = "J. McGrath"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)

    global plot_figures, tol, ml_factor, refx1, refx2, refy1, refy2, n_ifg, \
        length, width, ifgdir, ifgdates, coh, i, v, begin, fullres, geocdir, \
        corrdir, bool_mask, cycle, n_valid_thre, min_error, coh_thresh, onlylisted, \
        poly_strings_all, lon, lat

    # %% Set default
    ifgdir = []  # GEOCml* dir
    tsadir = []  # TS_GEOCml* dir
    corrdir = []  # Directory to hold the corrections
    ml_factor = []  # Amount to multilook the resulting masks
    errorfile = []  # File to hold lines containing known errors
    autoerror = False
    coh_thresh = []  # Coherence threshold for autoerror
    fullres = False
    reset = False
    plot_figures = False
    onlylisted = False
    v = -1
    cycle = 3
    n_valid_thre = 0.5
    min_error = 10  # Minimum size of error region (in pixels at final resolution)

    # Parallel Processing options
    try:
        n_para = len(os.sched_getaffinity(0))
    except:
        n_para = multi.cpu_count()

    if sys.platform == "linux" or sys.platform == "linux2":
        q = multi.get_context('fork')
    elif sys.platform == "win32":
        q = multi.get_context('spawn')

    # %% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hd:t:c:m:e:v:f:", ["help", "reset", "n_para=", "fullres", "autoerror", "coh_thresh=", "onlylisted"])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == ' --help':
                print(__doc__)
                return 0
            elif o == '-d':
                ifgdir = a
            elif o == '-t':
                tsadir = a
            elif o == '-c':
                corrdir = a
            elif o == '-m':
                ml_factor = int(a)
            elif o == '-f':
                min_error = int(a)
            elif o == '-e':
                errorfile = a
            elif o == '-v':
                v = int(a) - 1
            elif o == '--reset':
                reset = True
            elif o == '--n_para':
                n_para = int(a)
            elif o == '--fullres':
                fullres = True
            elif o == '--autoerror':
                autoerror = True
            elif o == '--coh_thresh':
                coh_thresh = float(a)
            elif o == '--onlylisted':
                onlylisted = True

        if not ifgdir:
            raise Usage('No data directory given, -d is not optional!')
        elif not os.path.isdir(ifgdir):
            raise Usage('No {} dir exists!'.format(ifgdir))
        elif not os.path.exists(os.path.join(ifgdir, 'slc.mli.par')):
            raise Usage('No slc.mli.par file exists in {}!'.format(ifgdir))

        if fullres:
            if ml_factor:
                raise Usage('Multilooking Factor given - not permitted with --fullres (will work to size GEOCml*)')
            elif autoerror:
                raise Usage('Fullres and Autoerror not permitted (havent yet implemented oversampling of coh_avg to fullres, or put in read cc.tif)')
            elif coh_thresh:
                raise Usage('Fullres and coh_thresh not permitted (havent yet implemented reading in *.cc.tif)')
            else:
                mlIx = os.path.basename(ifgdir).find('ml')
                mlIn = [ii for ii in os.path.basename(ifgdir)[mlIx + 2:]]
                search = True
                ml_factor = []
                while search:
                    for alpha in mlIn:
                        if alpha.isnumeric():
                            ml_factor.append(alpha)
                        else:
                            break
                    search = False
                ml_factor = int("".join(ml_factor))
        elif not ml_factor:
            ml_factor = 1

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  " + str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2

    # %% Directory setting
    ifgdir = os.path.abspath(ifgdir)

    if not tsadir:
        tsadir = os.path.join(os.path.dirname(ifgdir), 'TS_' + os.path.basename(ifgdir))

    if not corrdir:
        if ml_factor == 1 or fullres:
            corrdir = os.path.join(os.path.dirname(ifgdir), os.path.basename(ifgdir) + 'LoopMask')
        else:  # In the event you are working with already multilooked data (GEOCml*) find what true output ml_factor will be
            mlIx = os.path.basename(ifgdir).find('ml')
            mlIn = [ii for ii in os.path.basename(ifgdir)[mlIx + 2:]]
            search = True
            ml_inFactor = []
            while search:
                for alpha in mlIn:
                    if alpha.isnumeric():
                        ml_inFactor.append(alpha)
                    else:
                        break
                search = False
            ml_inFactor = int("".join(ml_inFactor))
            ml_outFactor = ml_factor * ml_inFactor
            corrdir = os.path.join(os.path.dirname(ifgdir), os.path.basename(ifgdir) + 'LoopMaskml{}'.format(ml_outFactor))

    if not os.path.exists(corrdir):
        os.mkdir(corrdir)

    infodir = os.path.join(tsadir, 'info')

    resultsdir = os.path.join(tsadir, 'results')

    if reset:
        print('Removing Previous Masks')
        if os.path.exists(corrdir):
            shutil.rmtree(corrdir)
    else:
        print('Preserving Premade Masks')

    loopy_lib.prepOutdir(corrdir, ifgdir)

    # %% File Setting
    ref_file = os.path.join(infodir, '12ref.txt')
    mlipar = os.path.join(ifgdir, 'slc.mli.par')

    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))

    # %% Prepare variables
    # Get ifg dates
    ifgdates = tools_lib.get_ifgdates(ifgdir)
    n_ifg = len(ifgdates)

    # Find how far to interpolate IFG to
    if fullres:
        geocdir = os.path.abspath(os.path.join(ifgdir, '..', 'GEOC'))
        print('Processing full resolution masks direct from tifs in {}'.format(geocdir))

        # Create full res mli
        print('\nCreate slc.mli', flush=True)
        cohfile = os.path.join(ifgdir, ifgdates[0], ifgdates[0])
        mlitif = glob.glob(os.path.join(geocdir, '*.geo.mli.tif'))
        if len(mlitif) > 0:
            mlitif = mlitif[0]  # First one
            coh = gdal.Open(mlitif).ReadAsArray()  # Coh due to previous use of coherence to find IFG limits
            if isinstance(coh, type(None)):
                print('Full Res Coherence == NoneType. Using hgt')
                coh = gdal.Open(glob.glob(os.path.join(geocdir, '*.geo.hgt.tif'))[0]).ReadAsArray()
            coh[coh == 0] = np.nan
            mlifile = os.path.join(corrdir, 'slc.mli')
            coh.tofile(mlifile)
            mlipngfile = mlifile + '.png'
            mli = np.log10(coh)
            vmin = np.nanpercentile(mli, 5)
            vmax = np.nanpercentile(mli, 95)
            plot_lib.make_im_png(mli, mlipngfile, 'gray', 'MLI (log10)', vmin=vmin, vmax=vmax, cbar=True)
            print('  slc.mli[.png] created', flush=True)
            ref_type = 'MLI'
        else:
            print('  No *.geo.mli.tif found in {}'.format(os.path.basename(geocdir)), flush=True)

    else:
        cohfile = os.path.join(resultsdir, 'coh_avg')
        ref_type = 'coherence'
        # If no coh file, use slc
        if not os.path.exists(cohfile):
            cohfile = os.path.join(ifgdir, 'slc.mli')
            print('No Coherence File - using MLI instead')
            ref_type = 'MLI'

        coh = io_lib.read_img(cohfile, length=length, width=width)

    if fullres:
        geotiff = gdal.Open(mlitif)
        widthtiff = geotiff.RasterXSize
        lengthtiff = geotiff.RasterYSize
        bool_mask = np.zeros((lengthtiff, widthtiff))
    else:
        bool_mask = np.zeros((length, width))

    if errorfile:
        print('Reading known errors')
        with open(errorfile) as f:
            poly_strings_all = f.readlines()

        if fullres:
            lon_w_p, postlon, _, lat_n_p, _, postlat = geotiff.GetGeoTransform()
            # lat lon are in pixel registration. dlat is negative
            lon1 = lon_w_p + postlon / 2
            lat1 = lat_n_p + postlat / 2
            lat2 = lat1 + postlat * (lengthtiff - 1)  # south
            lon2 = lon1 + postlon * (widthtiff - 1)  # east
            lon, lat = np.linspace(lon1, lon2, widthtiff), np.linspace(lat1, lat2, lengthtiff)
        else:
            dempar = os.path.join(ifgdir, 'EQA.dem_par')
            lat1 = float(io_lib.get_param_par(dempar, 'corner_lat'))  # north
            lon1 = float(io_lib.get_param_par(dempar, 'corner_lon'))  # west
            postlat = float(io_lib.get_param_par(dempar, 'post_lat'))  # negative
            postlon = float(io_lib.get_param_par(dempar, 'post_lon'))  # positive
            lat2 = lat1 + postlat * (length - 1)  # south
            lon2 = lon1 + postlon * (width - 1)  # east
            lon, lat = np.linspace(lon1, lon2, width), np.linspace(lat1, lat2, length)


        # %% Run correction in parallel

        _n_para = n_para if n_para < len(poly_strings_all) else len(poly_strings_all)
        print('\nMapping Likely Errors, '.format(n_ifg), flush=True)

        if n_para == 0:
            print('with no parallel processing...', flush=True)
            strnum = 0
            for poly_str in poly_strings_all:
                start = time.time()
                strnum += 1
                bool_mask = bool_mask + tools_lib.poly_mask(poly_str, lon, lat, radius=2)
                print('Plotted Likely Error {}/{}\t({:.3f}s)'.format(i, len(poly_strings_all), time.time() - start))
            bool_mask[np.where(bool_mask != 0)] = 1

        else:
            print('with {} parallel processing...'.format(_n_para), flush=True)
            # Parallel processing
            p = q.Pool(_n_para)
            bool_all = np.array(p.map(plotKnown, range(len(poly_strings_all))))
            p.close()
            bool_mask[np.where(np.sum(bool_all, axis=0) != 0)] = 1

# %%
    if autoerror:
        statsfile = os.path.join(infodir, '11ifg_stats.txt')
        if not coh_thresh:
            if os.path.exists(statsfile):
                with open(statsfile) as f:
                    param = f.readlines()
                coh_thresh = float([val.split()[4] for val in param if 'coh_thre' in val][0])
            else:
                coh_thresh = 0.15

        print('Using {} average coherence threshold to find likely errors'.format(coh_thresh))
        cohavgfile = os.path.join(tsadir, 'results', 'coh_avg')
        if os.path.exists(cohavgfile):
            print('Found existing coh_avg file in TSdir')
            coh_avg = io_lib.read_img(cohavgfile, length, width)
        else:
            print('No coh_avg file. Calculating now...')
            coh_avg = np.zeros((length, width), dtype=np.float32)
            n_coh = np.zeros((length, width), dtype=np.int16)
            n_unw = np.zeros((length, width), dtype=np.int16)


#            # %% Run correction in parallel
#            _n_para = n_para if n_para < n_ifg else n_ifg
#            print('\nCalculating Mean Coherence from {} ifgs, '.format(n_ifg), flush=True)
#
#            if n_para == 1:
#                print('with no parallel processing...', flush=True)
#                if v >= 0:
#                    print('In an overly verbose way for IFG {}'.format(v + 1))
#
#               for ifgd in ifgdates:
#                    print(ifgd)
#                    ccfile = os.path.join(ifgdir, ifgd, ifgd + '.cc')
#                    if os.path.getsize(ccfile) == length * width:
#                        coh1 = io_lib.read_img(ccfile, length, width, np.uint8)
#                        coh1 = coh1.astype(np.float32) / 255
#                  else:
#                        coh1 = io_lib.read_img(ccfile, length, width)
#                        coh1[np.isnan(coh)] = 0  # Fill nan with 0
#
#                   coh_avg += coh1
#                    n_coh += (coh1 != 0)
#
#                   unwfile = os.path.join(ifgdir, ifgd, ifgd + '.unw')
#                   unw = io_lib.read_img(unwfile, length, width)
#
#                   unw[unw == 0] = np.nan  # Fill 0 with nan
#                   n_unw += ~np.isnan(unw)  # Summing number of unnan unw
#
#            else:
#                print('with {} parallel processing...'.format(_n_para), flush=True)
#
#                # Parallel processing
#                p = q.Pool(_n_para)
#                p.map(mask_unw_errors, range(n_ifg))
#                p.close()



            for ifgd in ifgdates:
                print(ifgd)
                ccfile = os.path.join(ifgdir, ifgd, ifgd + '.cc')
                if os.path.getsize(ccfile) == length * width:
                    coh1 = io_lib.read_img(ccfile, length, width, np.uint8)
                    coh1 = coh1.astype(np.float32) / 255
                else:
                    coh1 = io_lib.read_img(ccfile, length, width)
                    coh1[np.isnan(coh)] = 0  # Fill nan with 0

                coh_avg += coh1
                n_coh += (coh1 != 0)

                unwfile = os.path.join(ifgdir, ifgd, ifgd + '.unw')
                unw = io_lib.read_img(unwfile, length, width)

                unw[unw == 0] = np.nan  # Fill 0 with nan
                n_unw += ~np.isnan(unw)  # Summing number of unnan unw

            coh_avg[n_coh == 0] = np.nan
            n_coh[n_coh == 0] = 1  # to avoid zero division
            print('New Version - divide by n_ifg not n_unw for each pixel')
            #coh_avg = coh_avg / n_coh
            coh_avg = coh_avg / n_ifg

        # loopy_lib.plotim(coh_avg, title='Coherence Average', centerz=False)
        errs = np.zeros((length, width))
        errs[np.where(coh_avg < coh_thresh)] = 1
        # loopy_lib.plotim(errs, title='Coherence < {}'.format(coh_thresh), centerz=False)
        labels = label(errs)[0]
        label_id, label_size = np.unique(labels, return_counts=True)
        label_id = label_id[np.where(label_size < min_error)]
        # loopy_lib.plotim(errs, centerz=False)
        errs[np.isin(labels, label_id)] = 0  # Drop any incoherent areas smaller than min_corr_size
        # loopy_lib.plotim(errs, title='Coherence trimmed', centerz=False)
        errs = binary_closing(errs, iterations=2)  # Fill in any holes in the incoherent regions
        # loopy_lib.plotim(errs, title='Coherence filled', centerz=False)

        errs2 = skeletonize(errs)
        # loopy_lib.plotim(errs2, title='Coherence skeletonized', centerz=False)

        bool_mask[np.where(errs2 == 1)] = 1

    if errorfile or autoerror:
        bool_plot = bool_mask.copy()
        bool_plot[np.where(np.isnan(coh))] = np.nan
        # loopy_lib.plotim(bool_plot, title='Coherence Thresh: {}'.format(coh_thresh), centerz=False)
        if fullres:
            bool_plot = tools_lib.multilook(bool_plot, ml_factor, ml_factor, n_valid_thre=0.1)
        title = 'Known UNW error Locations)'
        plot_lib.make_im_png(bool_plot, os.path.join(corrdir, 'known_errors.png'), 'viridis', title, vmin=0, vmax=1, cbar=False)
        print('Map of known error locations made')



    # Find reference pixel. If none provided, use highest coherence pixel
    if os.path.exists(ref_file):
        with open(ref_file, "r") as f:
            refarea = f.read().split()[0]  # str, x1/x2/y1/y2
        refx1, refx2, refy1, refy2 = [int(s) for s in re.split('[:/]', refarea)]

        # Change reference pixel in case working with fullres data
        if fullres:
            refx1 = refx1 * ml_factor
            refx2 = refx2 * ml_factor
            refy1 = refy1 * ml_factor
            refy2 = refy2 * ml_factor

        if np.isnan(np.nanmean(coh[refy1:refy2, refx1:refx2])):
            print('Ref point = [{}, {}] invalid. Using max {} pixel'.format(refy1, refx1, ref_type))
            refy1, refx1 = np.where(coh == np.nanmax(coh))
            refy1 = refy1[0]
            refy2 = refy1 + 1
            refx1 = refx1[0]
            refx2 = refx1 + 1

    else:
        print('No Reference Pixel provided - using max {} pixel'.format(ref_type))

        refy1, refx1 = np.where(coh == np.nanmax(coh))
        refy1 = refy1[0]
        refy2 = refy1 + 1
        refx1 = refx1[0]
        refx2 = refx1 + 1

    print('Ref point = [{}, {}]'.format(refy1, refx1))
    print('Mask Multilooking Factor = {}'.format(ml_factor))
    print('Minimum Output Correction Size = {} pixels'.format(min_error))

    # %% Create new EQA and mli_par files if multilooking
    if not fullres and ml_factor != 1:
        if not os.path.exists(tsadir):
          os.mkdir(tsadir)

        mlipar = os.path.join(corrdir, 'slc.mli.par')
        dempar = os.path.join(corrdir, 'EQA.dem_par')
        eqapar = os.path.join(ifgdir, 'EQA.dem_par')
        if os.path.exists(mlipar):
            os.remove(mlipar)

        radar_freq = 5.405e9

        print('\nCreate slc.mli.par', flush=True)

        with open(mlipar, 'w') as f:
            print('range_samples:   {}'.format(int(width / ml_factor)), file=f)
            print('azimuth_lines:   {}'.format(int(length / ml_factor)), file=f)
            print('radar_frequency: {} Hz'.format(radar_freq), file=f)

        if os.path.exists(dempar):
            os.remove(dempar)
        print('\nCreate EQA.dem_par', flush=True)

        lat1 = float(io_lib.get_param_par(eqapar, 'corner_lat'))  # north
        lon1 = float(io_lib.get_param_par(eqapar, 'corner_lon'))  # west
        postlat = float(io_lib.get_param_par(eqapar, 'post_lat'))  # negative
        postlon = float(io_lib.get_param_par(eqapar, 'post_lon'))  # positive
        lat2 = lat1 + postlat * (length - 1)  # south
        lon2 = lon1 + postlon * (width - 1)  # east
        lon, lat = np.linspace(lon1, lon2, width), np.linspace(lat1, lat2, length)

        text = ["Gamma DIFF&GEO DEM/MAP parameter file",
                "title: DEM",
                "DEM_projection:     EQA",
                "data_format:        REAL*4",
                "DEM_hgt_offset:          0.00000",
                "DEM_scale:               1.00000",
                "width: {}".format(int(width / ml_factor)),
                "nlines: {}".format(int(length / ml_factor)),
                "corner_lat:     {}  decimal degrees".format(lat1),
                "corner_lon:    {}  decimal degrees".format(lon1),
                "post_lat: {} decimal degrees".format(postlat * ml_factor),
                "post_lon: {} decimal degrees".format(postlon * ml_factor),
                "",
                "ellipsoid_name: WGS 84",
                "ellipsoid_ra:        6378137.000   m",
                "ellipsoid_reciprocal_flattening:  298.2572236",
                "",
                "datum_name: WGS 1984",
                "datum_shift_dx:              0.000   m",
                "datum_shift_dy:              0.000   m",
                "datum_shift_dz:              0.000   m",
                "datum_scale_m:         0.00000e+00",
                "datum_rotation_alpha:  0.00000e+00   arc-sec",
                "datum_rotation_beta:   0.00000e+00   arc-sec",
                "datum_rotation_gamma:  0.00000e+00   arc-sec",
                "datum_country_list: Global Definition, WGS84, World\n"]

        with open(dempar, 'w') as f:
            f.write('\n'.join(text))

    # %% Run correction in parallel
    _n_para = n_para if n_para < n_ifg else n_ifg
    print('\nRunning error mapping for all {} ifgs, '.format(n_ifg), flush=True)

    if n_para == 1:
        print('with no parallel processing...', flush=True)
        if v >= 0:
            print('In an overly verbose way for IFG {}'.format(v + 1))

        for i in range(n_ifg):
            mask_unw_errors(i)

    else:
        print('with {} parallel processing...'.format(_n_para), flush=True)
        if v >= 0:
            print('In an overly verbose way for IFG {}'.format(v + 1))

        # Parallel processing
        p = q.Pool(_n_para)
        p.map(mask_unw_errors, range(n_ifg))
        p.close()


    if ml_factor != 1:
        print('Multilooking the metadata')

        for file in ['E.geo', 'N.geo', 'U.geo', 'slc.mli', 'hgt']:
            data = io_lib.read_img(os.path.join(tsadir, file), length, width)
            data = tools_lib.multilook(data, ml_factor, ml_factor)
            outfile = os.path.join(tsadir, file)
            data.tofile(outfile)
            print('  {} multilooked'.format(file), flush=True)

    # %% Finish
    elapsed_time = time.time() - start
    hour = int(elapsed_time / 3600)
    minute = int(np.mod((elapsed_time / 60), 60))
    sec = int(np.mod(elapsed_time, 60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour, minute, sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output directory: {}\n'.format(os.path.relpath(corrdir)))


# %% Function to mask unwrapping errors
def mask_unw_errors(i):
    global begin
    begin = time.time()
    date = ifgdates[i]
    if not os.path.exists(os.path.join(corrdir, date)):
        os.mkdir(os.path.join(corrdir, date))
    if os.path.exists(os.path.join(corrdir, date, date + '.unw')):
        print('    ({}/{}): {}  Mask Exists. Skipping'.format(i + 1, n_ifg, date))
        return
    else:
        print('    ({}/{}): {}'.format(i + 1, n_ifg, date))
    if i == v:
        print('        Starting {} verbosely'.format(date))
    if i == v:
        print('        Loading')

    # Read in IFG
    if fullres:
        unw = gdal.Open(os.path.join(geocdir, date, date + '.geo.unw.tif')).ReadAsArray()
        unw[unw == 0] = np.nan

    else:
        unw = io_lib.read_img(os.path.join(ifgdir, date, date + '.unw'), length=length, width=width)

    if i == v:
        print('        UNW Loaded {:.2f}'.format(time.time() - begin))

    if coh_thresh:
        cc_map = io_lib.read_img(os.path.join(ifgdir, date, date + '.cc'), length=length, width=width, dtype=np.uint8).astype(np.float32) / 255
        unw[np.where(cc_map < coh_thresh)] = np.nan

    # Find Reference Value, and reference all IFGs to same value
    ifg = unw.copy()
    if np.all(np.isnan(unw[refy1:refy2, refx1:refx2])):
        print('Invalid Ref Value found for IFG {}. Setting to 0'.format(date))
        ref = 0
    else:
        ref = np.nanmean(unw[refy1:refy2, refx1:refx2])
        ifg = ifg - ref  # Maybe no need to use a reference - would be better to subtract 0.5 pi or something, incase IFG is already referenced
    if i == v:
        print('        Reffed {:.2f}'.format(time.time() - begin))
    # %%


    # Interpolate IFG to entire frame
    filled_ifg = NN_interp(ifg)

    if i == v:
        print('        Interpolated {:.2f}'.format(time.time() - begin))

    # Find modulo 2pi values for original IFG, and after adding and subtracting 1pi
    # Use round rather than // to account for slight noise
    nPi = 1
    npi = (filled_ifg / (nPi * np.pi)).round()

    if i == v:
        print('        npi_calculated {:.2f}'.format(time.time() - begin))

    # Modal filtering of npi images
    start = time.time()
    npi = mode_filter(npi, filtSize=21)

    if i == v:
        print('        Scipy filtered {:.2f} ({:.2f} s)'.format(time.time() - begin, time.time() - start))

    if not onlylisted:
        if i == v:
            print('        Searching for errors')
        # %%
        errors = np.zeros(npi.shape) * np.nan
        errors[np.where(~np.isnan(npi))] = 0

        # Compare with 1 row below
        error_rows, error_cols = np.where((np.abs(npi[:-1, :] - npi[1:, :]) > 1))
        errors[error_rows, error_cols] = 1

        # Compare with 1 row above
        error_rows, error_cols = np.where((np.abs(npi[1:, :] - npi[:-1, :]) > 1))
        errors[error_rows + 1, error_cols] = 1

        # Compare to column to the left
        error_rows, error_cols = np.where((np.abs(npi[:, 1:] - npi[:, :-1]) > 1))
        errors[error_rows, error_cols] = 1

        # Compare to column to the right
        error_rows, error_cols = np.where((np.abs(npi[:, :-1] - npi[:, 1:]) > 1))
        errors[error_rows, error_cols + 1] = 1

        # Add known error locations
        errors[np.where(bool_mask == 1)] = 1

        if i == v:
            print('        Boundaries Classified {:.2f}'.format(time.time() - begin))

    else:
        if i == v:
            print('        Using pre-defined errors')
        errors = np.zeros(unw.shape) * np.nan
        errors[np.where(~np.isnan(unw))] = 0
        # Add known error locations
        errors[np.where(bool_mask == 1)] = 1
        if i == v:
            print('        Only defined errors used {:.2f}'.format(time.time() - begin))

    # %%
    # Add error lines to the original IFG, and interpolate with these values to
    # create IFG split up by unwrapping error boundaries
    ifg2 = unw.copy()
    if i == v:
        print('        Copied unw {:.2f}'.format(time.time() - begin))
    err_val = 10 * np.nanmax(ifg2)
    if i == v:
        print('        err_val set {:.2f}'.format(time.time() - begin))

    ifg2[np.where(errors == 1)] = err_val
    if i == v:
        print('        Boundaries added {:.2f}'.format(time.time() - begin))
    filled_ifg2 = NN_interp(ifg2)

    if i == v:
        print('        IFG2 interpolated {:.2f}'.format(time.time() - begin))

    # Binarise the IFG
    filled_ifg2[np.where(filled_ifg2 == err_val)] = np.nan
    filled_ifg2[~np.isnan(filled_ifg2)] = 1
    filled_ifg2[np.isnan(filled_ifg2)] = 0

    if i == v:
        print('        Binarised {:.2f}'.format(time.time() - begin))

    # Label the binary IFG into connected regions
    regions, count = label(filled_ifg2)
    regions = regions.astype('float32')
    regionId, regionSize = np.unique(regions, return_counts=True)

    if i == v:
        print('        Added to IFG {:.2f}'.format(time.time() - begin))

    # Set a minimum size of region to be corrected
    min_corr_size = min_error * ml_factor ** 2  # 10 pixels at final ml size

    # Region IDs to small to corrected
    drop_regions = regionId[np.where(regionSize < min_corr_size)]
    timer = time.time()
    regions[np.where(np.isin(regions, np.append(drop_regions, 0)))] = np.nan

    # Cease if there are no regions left to be checked
    if drop_regions.shape[0] == regionId.shape[0]:
        correction = np.zeros(unw.shape)
    else:
        # Remove error areas and tiny regions from NPI so they match
        npi_corr = npi.copy()
        npi_corr[np.where(np.isnan(regions))] = np.nan
        # Reinterpolate without tiny regions
        if i == v:
            print('interp prep in {:.2f} secs'.format(time.time() - timer))
        timer = time.time()
        npi_corr = NN_interp(npi_corr)
        regions = NN_interp(regions)
        if i == v:
            print('interp in {:.2f} secs'.format(time.time() - timer))

        # Find region number of reference pixel. All pixels in this region to be
        # considered unw error free. Mask where 1 == good pixel, 0 == bad
        # Use mode incase ref area is > 1 pixel (eg if working at full res)
        ref_region = mode(regions[refy1:refy2, refx1:refx2].flatten(), keepdims=True)[0][0]
        mask = regions == ref_region

        if i == v:
            print('        Mask made {:.2f}'.format(time.time() - begin))
        # breakpoint()
        # Make an array exclusively holding the good values
        good_vals = np.zeros(mask.shape) * np.nan
        good_vals[mask] = npi_corr[mask]

        # Make an array to hold the correction
        correction = np.zeros(mask.shape)

        # Boolean array of the outside boundary of the good mask
        good_border = filters.sobel(mask).astype('bool')
        corr_regions = np.unique(regions[good_border])

        if np.any(corr_regions == ref_region):
            corr_regions = np.delete(corr_regions, np.where(corr_regions == ref_region)[0][0])

        if np.any(np.isnan(corr_regions)):
            corr_regions = np.delete(corr_regions, np.where(np.isnan(corr_regions))[0][0])

        if i == v:
            print('        Preparing Corrections {:.2f}'.format(time.time() - begin))
    # %%
        for ii, corrIx in enumerate(corr_regions):
            # Make map only of the border regions
            start = time.time()
            border_regions = np.zeros(mask.shape)
            border_regions[good_border] = regions[good_border]
            # Plot boundary in isolation
            border = np.zeros(mask.shape).astype('int')
            border[np.where(border_regions == corrIx)] = 1
            # Dilate boundary so it crosses into both regions
            border_dil = binary_dilation(border).astype('int')
            
            error_loc = np.where(border)
            
            corr_val = []
            for ix, erry in enumerate(error_loc[0]):
              errx = error_loc[1][ix]
              errxmin = errx - 1 if errx > 0 else 0
              errxmax = errx + 2 if errx < (width - 1) else width
              errymin = erry - 1 if erry > 0 else 0
              errymax = erry + 2 if erry < (length - 1) else length
              
              err_val = npi_corr[erry, errx]
              good_val = good_vals[errymin:errymax, errxmin:errxmax]
              corr_val.append(((np.nanmedian(good_val) - err_val) * (nPi / 2)).round() * 2 * np.pi)

            if i == v:
                print('corr_val')
                print(np.unique(corr_val, return_counts=True))
            
            corr_val = np.median(corr_val)
            correction[np.where(regions == corrIx)] = corr_val
            if i == v:
                print('            Done {:.0f}/{:.0f}: {:.2f} rads {:.2f} secs'.format(ii + 1, len(corr_regions), corr_val, time.time() - start))
        if i == v:
            print('        Correction Calculated {:.2f}'.format(time.time() - begin))

    # Apply correction to original version of IFG
    if coh_thresh:
        unw = io_lib.read_img(os.path.join(ifgdir, date, date + '.unw'), length=length, width=width)
    corr_unw = unw.copy()
    if i == v:
        print('        UNW copied {:.2f}'.format(time.time() - begin))
    corr_unw[np.where(~np.isnan(corr_unw))] = corr_unw[np.where(~np.isnan(corr_unw))] + correction[np.where(~np.isnan(corr_unw))]
    if i == v:
        print('        Correction Applied {:.2f}'.format(time.time() - begin))

    # %% Multilook mask if required
    if ml_factor != 1:
        unw = tools_lib.multilook(unw, ml_factor, ml_factor, n_valid_thre=n_valid_thre)
        if i == v:
            print('        Original IFG multilooked {:.2f}'.format(time.time() - begin))
        mask = tools_lib.multilook(mask, ml_factor, ml_factor, 0.1).astype('bool').astype('int')
        if i == v:
            print('        Mask multilooked {:.2f}'.format(time.time() - begin))
        npi = tools_lib.multilook((filled_ifg / (np.pi)).round(), ml_factor, ml_factor, n_valid_thre=0.1)
        if i == v:
            print('        Modulo NPI multilooked {:.2f}'.format(time.time() - begin))
        correction = tools_lib.multilook(correction, ml_factor, ml_factor, n_valid_thre=0.1)
        if i == v:
            print('        Correction multilooked {:.2f}'.format(time.time() - begin))
        corr_unw = tools_lib.multilook(corr_unw, ml_factor, ml_factor, n_valid_thre=n_valid_thre)
        if i == v:
            print('        Corrected IFG multilooked {:.2f}'.format(time.time() - begin))
        errors = tools_lib.multilook(errors, ml_factor, ml_factor, n_valid_thre=0.1)
        if i == v:
            print('        Error map multilooked {:.2f}'.format(time.time() - begin))

        # Multilook coherence files as well if needed
        if not fullres:
            cohfile = os.path.join(ifgdir, date, date + '.cc')
            cc = io_lib.read_img(cohfile, length=length, width=width, dtype=np.uint8).astype(np.float32)
            cc[cc == 0] = np.nan
            cc = tools_lib.multilook(cc, ml_factor, ml_factor, n_valid_thre=n_valid_thre).astype(np.uint8)
            cc.tofile(os.path.join(corrdir, date, date + '.cc'))
            if i == v:
                print('        Coherence multilooked {:.2f}'.format(time.time() - begin))

    # %% Make PNGs
    # Flip round now, so 1 = bad pixel, 0 = good pixel
    # mask = (mask == 0).astype('int')
    # mask[np.where(np.isnan(unw))] = 0
    title = '{} ({}pi/cycle)'.format(date, cycle * 2)
    plot_lib.make_im_png(np.angle(np.exp(1j * corr_unw / cycle) * cycle), os.path.join(corrdir, date, date + '.unw.png'), SCM.romaO, title, vmin=-np.pi, vmax=np.pi, cbar=False)
    # Make new unw file from corrected data and new loop png
    corr_unw.tofile(os.path.join(corrdir, date, date + '.unw'))
    # mask.astype('bool').tofile(os.path.join(corrdir, date, date + '.mask'))
    # Create correction png image (UnCorr_unw, npi, correction, Corr_unw)
    corrcomppng = os.path.join(corrdir, date, date + '.maskcorr.png')
    titles4 = ['{} Uncorrected'.format(ifgdates[i]),
               '{} Corrected'.format(ifgdates[i]),
               'Modulo nPi',
               'Mask Correction (n * 2Pi)']
    npi[np.where(np.isnan(unw))] = np.nan
    correction[np.where(np.isnan(unw))] = np.nan
    loopy_lib.make_compare_png(unw, corr_unw, npi, correction / (2 * np.pi), corrcomppng, titles4, 3)

    title = 'Error Map'
    #if coh_thresh:
    #    errs_found = errors.copy()
    #    errors[np.where(cc < coh_thresh)] = 0.5
    #    errors[np.where(errs_found)] = 1

    plot_lib.make_im_png(errors, os.path.join(corrdir, date, date + '.errormap.png'), 'viridis', title, vmin=0, vmax=1, cbar=False)

    if i == v:
        print('        pngs made {:.2f}'.format(time.time() - begin))

    # Link to the cc file
    if not os.path.exists(os.path.join(corrdir, date, date + '.cc')):
        shutil.copy(os.path.join(ifgdir, date, date + '.cc'), os.path.join(corrdir, date, date + '.cc'))

    if i == v:
        print('        Saved {:.2f}'.format(time.time() - begin))

    return


# %% Function to carry out nearest neighbour interpolation
def NN_interp(data):
    mask = np.where(~np.isnan(data))
    interp = NearestNDInterpolator(np.transpose(mask), data[mask])
    interped_data = data.copy()
    interp_to = np.where(((~np.isnan(coh)).astype('int') + np.isnan(data).astype('int')) == 2)
    try:
        nearest_data = interp(*interp_to)
        interped_data[interp_to] = nearest_data
    except IndexError:
        print('IndexError: Interpolation Failed - check IFG data coverage')

    return interped_data


# %% Function to modally filter arrays using Scikit
def mode_filter(data, filtSize=21):
    npi_min = np.nanmin(data) - 1
    npi_range = np.nanmax(data) - npi_min

    # Convert array into 0-255 range
    greyscale = ((data - npi_min) / npi_range) * 255

    # Filter image, convert back to np.array, and repopulate with nans
    im_mode = modal(greyscale.astype('uint8'), np.ones([filtSize, filtSize]))
    dataMode = ((np.array(im_mode, dtype='float32') / 255) * npi_range + npi_min).round()
    dataMode[np.where(np.isnan(data))] = np.nan
    dataMode[np.where(dataMode == npi_min)] = np.nan

    return dataMode


def plotKnown(i):
    poly_str = poly_strings_all[i]
    start = time.time()
    poly_mask = tools_lib.poly_mask(poly_str, lon, lat, radius=2)
    if np.mod(i + 1, 25) == 0:
        print('Plotting Likely Error {}/{}\t({:.3f}s)'.format(i + 1, len(poly_strings_all), time.time() - start))
    return poly_mask

# %% main
if __name__ == "__main__":
    sys.exit(main())
