#!/usr/bin/env python3
"""
========
Overview
========
Python3 library of unwrap error correction functions for LOOPY

=========
Changelog
=========
v1.0 20220608 Jack McGrath, Uni of Leeds
 - Original implementation
"""
import os
import sys
import glob
import shutil
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import LiCSBAS_tools_lib as tools_lib

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


with warnings.catch_warnings():  # To silence user warning
    warnings.simplefilter('ignore', UserWarning)
    if sys.platform == 'win32':
        mpl.use('module://matplotlib_inline.backend_inline')

cmap_wrap = tools_lib.get_cmap('SCM.romaO')
cmap_corr = tools_lib.get_cmap('SCM.vik')


# %% Inline plotting of the
def plotim(data, centerz=True, title='', cmap='viridis', vmin=None, vmax=None, interp='antialiased', cbar=True):
    # cmap_wrap = tools_lib.get_cmap('SCM.romaO')
    plt.figure()
    if centerz:
        vmin = -(np.nanmax(abs(data)))
        vmax = (np.nanmax(abs(data)))

    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interp)
    if cbar:
        plt.colorbar(**{'format': '%.0f'})
    plt.title(title)
    plt.show()


# %%
def prepOutdir(out_dir, in_dir):
    """
    Script to create the new GEOC dir and move the correct files to it at the
    start of LOOPY
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    files = glob.glob(os.path.join(in_dir, '*'))
    for file in files:
        if not os.path.isdir(file):  # not copy directory, only file
            print('Copy {}'.format(os.path.basename(file)), flush=True)
            shutil.copy(file, out_dir)

    print('{} prepared...'.format(os.path.basename(out_dir)))


# %%
def make_compare_png(uncorr, corrunw, npi, corr, png, titles4, cycle):
    """
    Make 4 panel png to compare uncorrected and corrected unw, also showing
    original modulo npi and correction
    """

    # Settings
    plt.rcParams['axes.titlesize'] = 10
    ifg = [uncorr, corrunw]

    length, width = uncorr.shape
    if length > width:
        figsize_y = 10
        figsize_x = int((figsize_y - 1) * width / length)
        if figsize_x < 5:
            figsize_x = 5
    else:
        figsize_x = 10
        figsize_y = int(figsize_x * length / width + 1)
        if figsize_y < 3:
            figsize_y = 3

    # Plot
    fig = plt.figure(figsize=(figsize_x, figsize_y))

    # Original and Corrected unw
    for i in range(2):
        data_wrapped = np.angle(np.exp(1j * (ifg[i] / cycle)) * cycle)
        ax = fig.add_subplot(2, 2, i + 1)  # index start from 1
        im = ax.imshow(data_wrapped, vmin=-np.pi, vmax=+np.pi, cmap=cmap_wrap,
                       interpolation='nearest')
        ax.set_title('{}'.format(titles4[i]))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        cax = plt.colorbar(im)
        cax.set_ticks([])

    # npi
    ax = fig.add_subplot(2, 2, 3)  # index start from 1
    im = ax.imshow(npi, cmap='tab20c', interpolation='nearest')
    ax.set_title('{}'.format(titles4[2]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    cax = plt.colorbar(im)

    # Correction
    ax = fig.add_subplot(2, 2, 4)  # index start from 1
    im = ax.imshow(corr, vmin=-np.nanmax(abs(corr)), vmax=np.nanmax(abs(corr)),
                   cmap=cmap_corr, interpolation='nearest')
    ax.set_title('{}'.format(titles4[3]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    cax = plt.colorbar(im)

    plt.tight_layout()
    plt.savefig(png)
    plt.close()
