# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 09:42:44 2022

@author: jdmcg
"""

#%% Import
import os
import numpy as np
import datetime as dt
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_inv_lib as inv_lib
import matplotlib.pyplot as plt
from matplotlib import dates as mdates


#%%
def plot_cand_network(ifgdates, bperp, rm_ifgdates, pngfile, ifg_cands=[], plot_bad=True):
    """
    Plot network of interferometric pairs.
    
    bperp can be dummy (-1~1).
    Suffix of pngfile can be png, ps, pdf, or svg.
    plot_bad
        True  : Plot bad ifgs by red lines
        False : Do not plot bad ifgs
    """

    imdates_all = tools_lib.ifgdates2imdates(ifgdates)
    n_im_all = len(imdates_all)
    imdates_dt_all = np.array(([dt.datetime.strptime(imd, '%Y%m%d') for imd in imdates_all])) ##datetime

    ifgdates = list(set(ifgdates)-set(rm_ifgdates)-set(ifg_cands))
    ifgdates.sort()
    imdates = tools_lib.ifgdates2imdates(ifgdates)
    imdates_dt = np.array(([dt.datetime.strptime(imd, '%Y%m%d') for imd in imdates])) ##datetime
    
    ### Identify gaps    
    G = inv_lib.make_sb_matrix(ifgdates)
    ixs_inc_gap = np.where(G.sum(axis=0)==0)[0]
    breakpoint()
    ### Plot fig
    figsize_x = np.round(((imdates_dt_all[-1]-imdates_dt_all[0]).days)/80)+2
    fig = plt.figure(figsize=(figsize_x, 6))
    ax = fig.add_axes([0.06, 0.12, 0.92,0.85])
    
    ### IFG blue lines
    for i, ifgd in enumerate(ifgdates):
        ix_m = imdates_all.index(ifgd[:8])
        ix_s = imdates_all.index(ifgd[-8:])
        label = 'IFG' if i==0 else '' #label only first
        plt.plot([imdates_dt_all[ix_m], imdates_dt_all[ix_s]], [bperp[ix_m],
                bperp[ix_s]], color='b', alpha=0.6, zorder=2, label=label)

    ### IFG bad red lines
    if plot_bad:
        for i, ifgd in enumerate(rm_ifgdates):
            ix_m = imdates_all.index(ifgd[:8])
            ix_s = imdates_all.index(ifgd[-8:])
            label = 'Removed IFG' if i==0 else '' #label only first
            plt.plot([imdates_dt_all[ix_m], imdates_dt_all[ix_s]], [bperp[ix_m],
                    bperp[ix_s]], color='r', alpha=0.6, zorder=6, label=label)
        
        for i, ifgd in enumerate(ifg_cands):
            ix_m = imdates_all.index(ifgd[:8])
            ix_s = imdates_all.index(ifgd[-8:])
            label = 'Bad Cands IFG' if i==0 else '' #label only first
            plt.plot([imdates_dt_all[ix_m], imdates_dt_all[ix_s]], [bperp[ix_m],
                    bperp[ix_s]], color='g', alpha=0.6, zorder=6, label=label)

    ### Image points and dates
    ax.scatter(imdates_dt_all, bperp, alpha=0.6, zorder=4)
    for i in range(n_im_all):
        if bperp[i] > np.median(bperp): va='bottom'
        else: va = 'top'
        ax.annotate(imdates_all[i][4:6]+'/'+imdates_all[i][6:],
                    (imdates_dt_all[i], bperp[i]), ha='center', va=va, zorder=8)

    ### gaps
    if len(ixs_inc_gap)!=0:
        gap_dates_dt = []
        for ix_gap in ixs_inc_gap:
            ddays_td = imdates_dt[ix_gap+1]-imdates_dt[ix_gap]
            gap_dates_dt.append(imdates_dt[ix_gap]+ddays_td/2)
        plt.vlines(gap_dates_dt, 0, 1, transform=ax.get_xaxis_transform(),
                   zorder=1, label='Gap', alpha=0.6, colors='k', linewidth=3)
        
    ### Locater        
    loc = ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    try:  # Only support from Matplotlib 3.1
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    except:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
        for label in ax.get_xticklabels():
            label.set_rotation(20)
            label.set_horizontalalignment('right')
    ax.grid(b=True, which='major')

    ### Add bold line every 1yr
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.grid(b=True, which='minor', linewidth=2)

    ax.set_xlim((imdates_dt_all[0]-dt.timedelta(days=10),
                 imdates_dt_all[-1]+dt.timedelta(days=10)))

    ### Labels and legend
    plt.xlabel('Time')
    if np.all(np.abs(np.array(bperp))<=1): ## dummy
        plt.ylabel('dummy')
    else:
        plt.ylabel('Bperp [m]')
    
    plt.legend()

    ### Save
    plt.savefig(pngfile)
    plt.close()

homedir = os.getcwd()
ifgdir = 'GEOCml10GACOS'
tsadir = os.path.join(homedir, 'TS_'+ifgdir)
ifgdir = os.path.join(homedir, ifgdir)
infodir = os.path.join(tsadir,'info')

ifgdates = tools_lib.get_ifgdates(ifgdir)
imdates = tools_lib.ifgdates2imdates(ifgdates)

n_ifg = len(ifgdates)
n_im = len(imdates)



bperp_file = os.path.join(ifgdir, 'baselines')
if os.path.exists(bperp_file):
    bperp = io_lib.read_bperp_file(bperp_file, imdates)
else: #dummy
    bperp = np.random.random(n_im).tolist()

bad_ifgfile = os.path.join(infodir, '12bad_ifg.txt')
bad_ifg_cands_file = os.path.join(infodir, '12bad_ifg_cand.txt')

bad_ifg_all = open(bad_ifgfile,'r').read().splitlines()
bad_ifg_cands_all = open(bad_ifg_cands_file,'r').read().splitlines()



netdir = os.path.join(tsadir, 'network')
pngfile = os.path.join(netdir, 'network12cands.png')
plot_cand_network(ifgdates, bperp, bad_ifg_all, bad_ifg_cands_all, pngfile)



