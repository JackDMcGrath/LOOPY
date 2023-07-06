import os
import numpy as np
import argparse
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
from osgeo import gdal
from scipy.io import savemat

class OpenTif:
    """ a Class that stores the band array and metadata of a Gtiff file."""
    def __init__(self, filename):
        self.ds = gdal.Open(filename)
        self.basename = os.path.splitext(os.path.basename(filename))[0]
        self.band = self.ds.GetRasterBand(1)
        self.data = self.band.ReadAsArray()
        self.xsize = self.ds.RasterXSize
        self.ysize = self.ds.RasterYSize
        self.left = self.ds.GetGeoTransform()[0]
        self.top = self.ds.GetGeoTransform()[3]
        self.xres = self.ds.GetGeoTransform()[1]
        self.yres = self.ds.GetGeoTransform()[5]
        self.right = self.left + self.xsize * self.xres
        self.bottom = self.top + self.ysize * self.yres
        self.projection = self.ds.GetProjection()
        pix_lin, pix_col = np.indices((self.ds.RasterYSize, self.ds.RasterXSize))
        self.lat, self.lon = self.top + self.yres*pix_lin, self.left+self.xres*pix_col

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    '''
    Use a multiple inheritance approach to use features of both classes.
    The ArgumentDefaultsHelpFormatter class adds argument default values to the usage help message
    The RawDescriptionHelpFormatter class keeps the indentation and line breaks in the ___doc___
    '''
    pass

# reading arguments
parser = argparse.ArgumentParser(description=__doc__, formatter_class=CustomFormatter)
parser.add_argument('-f', dest="frame_dir", default="./", help="directory of LiCSBAS output of a particular frame")
parser.add_argument('-c', dest='cc_dir', default="GEOCml10GACOS", help="folder containing cc input")
parser.add_argument('-d', dest='unw_dir', default="GEOCml10GACOS", help="folder containing unw input")
parser.add_argument('-o', dest='outfile', default="output.mat", help="output .mat file")
parser.add_argument('-l', dest='ifg_list', type=str, help="text file containing a list of ifgs")
parser.add_argument('-t', dest='tif', type=str, help="tif file containing coordinates info")
args = parser.parse_args()

### Define input directories
ccdir = os.path.abspath(os.path.join(args.frame_dir, args.cc_dir))
ifgdir = os.path.abspath(os.path.join(args.frame_dir, args.unw_dir))

### Get size
mlipar = os.path.join(ccdir, 'slc.mli.par')
width = int(io_lib.get_param_par(mlipar, 'range_samples'))
length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))
print("\nSize         : {} x {}".format(width, length), flush=True)

### Get_ifgdate
if args.ifg_list:
    ifgdates = io_lib.read_ifg_list(args.ifg_list)
else:
    ### Get dates
    ifgdates = tools_lib.get_ifgdates(ifgdir)
ifgdates.sort()
imdates = tools_lib.ifgdates2imdates(ifgdates)

### Intialise arrays
data = np.zeros((width*length, len(ifgdates)))
coherence = np.zeros((width*length, len(ifgdates)))
lonlat = np.zeros((width*length, 2))
ifg_id = np.zeros((len(ifgdates), 2))

### Accumulate through network (1)unw pixel counts, (2) coherence and (3) size of connected components
for i, ifgd in enumerate(ifgdates):
    # turn ifg into ones and zeros for non-nan and nan values
    unwfile = os.path.join(ifgdir, ifgd, ifgd + '.unw')
    unw = io_lib.read_img(unwfile, length, width)

    # coherence values from 0 to 1
    ccfile = os.path.join(ccdir, ifgd, ifgd + '.cc')
    coh = io_lib.read_img(ccfile, length, width, np.uint8)
    coh = coh.astype(np.float32) / 255  # keep 0 as 0 which represent nan values

    data[:, i] = unw.flatten()
    coherence[:, i] = coh.flatten()


mask = np.isnan(np.prod(data, axis=1))
# coherence[mask] = 0
# data[mask] = 0
# data = data % (2*np.pi)
coherence = coherence[~mask]
data = data[~mask]

coord = OpenTif(args.tif)
lat = coord.lat.flatten()
lon = coord.lon.flatten()
lat = lat[~mask]
lon = lon[~mask]
lonlat = np.vstack((lon, lat)).T

primarylist = []
secondarylist = []
for ifgd in ifgdates:
    primarylist.append(imdates.index(ifgd[:8]))
    secondarylist.append(imdates.index(ifgd[-8:]))

indices = np.vstack((primarylist, secondarylist)).T
mdic = {"data":data, "coherence":coherence, "lonlat":lonlat , "indices":indices}
savemat(args.outfile, mdic)

