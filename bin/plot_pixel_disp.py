#!/usr/bin/env python3
""""
Script for plotting displacements directly from the cum.h5
"""


import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import argparse
import datetime as dt

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
    global args

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=CustomFormatter)
    parser.add_argument('-i', dest='h5_file', default='cum.h5', help='.h5 file containing results of LiCSBAS velocity inversion')
    parser.add_argument('-x', dest='x', default=None, type=int)
    parser.add_argument('-y', dest='y', default=None, type=int)


    args = parser.parse_args()

def plot_pixel():

    h5file = args.h5_file
    data = h5.File(h5file, 'r')
    dates = [dt.datetime.strptime(str(d), '%Y%m%d').date() for d in np.array(data['imdates'])]

    # Not referencing anymore - referencing already occurred in LiCSBAS13_sb_inv.py
    cum = np.array(data['cum'])

    disp = cum[:, args.y, args.x]

    plt.scatter(dates, disp)
    plt.title('({}/{})'.format(args.y, args.x))
    plt.savefig(os.path.join('./', '{}-{}.png'.format(args.y, args.x)))
    


def main():
    init_args()
    plot_pixel()

if __name__ == "__main__":
    main()
