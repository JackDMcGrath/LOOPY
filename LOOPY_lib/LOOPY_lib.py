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
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import LiCSBAS_tools_lib as tools_lib

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


with warnings.catch_warnings():  # To silence user warning
    warnings.simplefilter('ignore', UserWarning)
    mpl.use('Agg')

cmap_wrap = tools_lib.get_cmap('SCM.romaO')
cmap_corr = tools_lib.get_cmap('SCM.vik')


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
