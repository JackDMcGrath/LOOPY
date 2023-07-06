# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:52:58 2023

@author: jdmcg
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
from scipy.ndimage import label, sobel
from scipy.ndimage import binary_dilation, binary_closing
from scipy.interpolate import NearestNDInterpolator
from skimage import filters
from skimage.filters.rank import modal
from skimage.morphology import skeletonize

insar = tools_lib.get_cmap('SCM.romaO')

# %% Set default
ifgdir = os.path.join('D:', 'LiCSBAS_singleFrames', '035D_Iran', 'GEOCml10GACOS')
tsadir = os.path.join(os.path.dirname(ifgdir), 'TS_' + os.path.basename(ifgdir))
corrdir = []  # Directory to hold the corrections
ml_factor = 10  # Amount to multilook the resulting masks
errorfile = []  # File to hold lines containing known errors
fullres = False
reset = False
plot_figures = False
v = -1
cycle = 3
min_corr_size = 10
# %% File Setting
infodir = os.path.join(tsadir, 'info')

resultsdir = os.path.join(tsadir, 'results')
ref_file = os.path.join(infodir, '12ref.txt')
mlipar = os.path.join(ifgdir, 'slc.mli.par')

width = int(io_lib.get_param_par(mlipar, 'range_samples'))
length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))

cohfile = os.path.join(resultsdir, 'coh_avg')
coh_avg = io_lib.read_img(cohfile, length, width)
coh_thres= 0.15

coh_avg = np.zeros((length, width), dtype=np.float32)
n_coh = np.zeros((length, width), dtype=np.int16)
n_unw = np.zeros((length, width), dtype=np.int16)

ifgdates = tools_lib.get_ifgdates(ifgdir)
for ifgd in ifgdates:
    ccfile = os.path.join(ifgdir, ifgd, ifgd + '.cc')
    if os.path.getsize(ccfile) == length * width:
        coh = io_lib.read_img(ccfile, length, width, np.uint8)
        coh = coh.astype(np.float32) / 255
    else:
        coh = io_lib.read_img(ccfile, length, width)
        coh[np.isnan(coh)] = 0  # Fill nan with 0

    coh_avg += coh
    n_coh += (coh != 0)

    unwfile = os.path.join(ifgdir, ifgd, ifgd + '.unw')
    unw = io_lib.read_img(unwfile, length, width)

    unw[unw == 0] = np.nan  # Fill 0 with nan
    n_unw += ~np.isnan(unw)  # Summing number of unnan unw

coh_avg[n_coh == 0] = np.nan
n_coh[n_coh == 0] = 1  # to avoid zero division
coh_avg = coh_avg / n_coh

statsfile = os.path.join(infodir, '11ifg_stats.txt')
if os.path.exists(statsfile):
    with open(statsfile) as f:
        param = f.readlines()
    coh_thres = float([val.split()[4] for val in param if 'coh_thre' in val][0])
# %%
errs = np.zeros((length, width))
errs[np.where(coh_avg < coh_thres)] = 1
labels = label(errs)[0]
label_id, label_size = np.unique(labels, return_counts=True)
label_id = label_id[np.where(label_size < min_corr_size)]
# loopy_lib.plotim(errs, centerz=False)
errs[np.isin(labels, label_id)] = 0  # Drop any incoherent areas smaller than min_corr_size
loopy_lib.plotim(errs, centerz=False)
errs = binary_closing(errs, iterations=2)  # Fill in any holes in the incoherent regions
loopy_lib.plotim(errs, centerz=False)

errs_bound = (binary_dilation(errs) != errs).astype('int')
loopy_lib.plotim(errs_bound, centerz=False)
# %%
errs2 = skeletonize(errs)
loopy_lib.plotim(errs2, centerz=False)
labels = label(errs2.astype('bool').astype('int'), structure=np.ones((3, 3)))[0]
label_id, label_size = np.unique(labels, return_counts=True)
label_id = label_id[np.where(label_size < 0)]
errs2[np.isin(labels, label_id)] = 0
loopy_lib.plotim(errs2, centerz=False)
# %%

coh_avg[np.where(errs2 == 1)] = 1
loopy_lib.plotim(coh_avg, centerz=False)

coh_full = np.repeat(np.repeat(coh_avg, ml_factor, axis=0), ml_factor, axis=1)
