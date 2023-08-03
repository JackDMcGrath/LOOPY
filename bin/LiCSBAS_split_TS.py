#!/usr/bin/env python3

#%% Import
import os
import time
import shutil
import numpy as np
from pathlib import Path
import argparse
import sys
import re
import glob
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_loop_lib as loop_lib
import LiCSBAS_plot_lib as plot_lib

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

def start():
    global start_time
    # intialise and print info on screen
    start_time = time.time()
    ver="1.0"; date=20230601; author="Jack McGrath"
    print("\n{} ver{} {} {}".format(os.path.basename(sys.argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)

def init_args():
    global args, mergeflag
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=CustomFormatter)
    parser.add_argument('-f', dest='frame_dir', default="./", help="directory of LiCSBAS output")
    parser.add_argument('-d', dest='unw_dir', default="GEOCml10", help="directory of the unw and multilooked ifgs")
    parser.add_argument('-c', dest='split_dir', default="GEOCml10", help="split directory basename of the unw and multilooked ifgs for merging")
    parser.add_argument('-s', dest='split_dates', default="split_dates.txt", help="text file containing the split dates used")
    parser.add_argument('--merge', dest='mergeflag', default=False, action='store_true', help="Merge directories")
    parser.add_argument('-m', dest='merge_suffix', default=False, help="Merge directories of this suffix")
    parser.add_argument('-k', dest='kaikoura', default=False, action='store_true', help='Split Merging at Kaikoura (needs a split date of 20161113)')
    parser.add_argument('-e', dest='coseismic', default=False, action='store_true', help='Merge Pre- and Post- Kaikoura datasets into 1 (should run -k first)')

    args = parser.parse_args()

def set_input_output():
    global unwdir, splitdates, splitbase, ifgdates, imdates, suffix, kaikoura

    # define input directories and file
    unwdir = os.path.abspath(os.path.join(args.frame_dir, args.unw_dir))
    splitbase = os.path.abspath(os.path.join(args.frame_dir, args.split_dir))
    splitfile = os.path.join(args.frame_dir, args.split_dates)
    if not os.path.exists(splitfile):
        raise Usage('{}: Split file does not exist! Exiting.....'.format(args.split_dates))

    splitdates = io_lib.read_ifg_list(splitfile)

    ifgdates = tools_lib.get_ifgdates(unwdir)
    imdates = tools_lib.ifgdates2imdates(ifgdates)

    if args.merge_suffix:
        suffix = args.merge_suffix
    else:
        suffix = ''

    if args.kaikoura and args.coseismic:
        raise Usage('Cannot use -k and -e flags at the same time! Run -k first then -e')

    kaikoura = 20161113

def make_split_dir():
    print('Splitting Time Series')
    unassigned = ifgdates.copy()
    pre_ix = 0
    post_ix = 0

    splitfile = os.path.join(args.frame_dir, 'splitdirs.txt')
    with open(splitfile, "w") as f:
        print('SplitID\tStart\tEnd', file=f)

    for i in np.arange(len(splitdates) - 1):
        split1 = int(splitdates[i])
        split2 = int(splitdates[i + 1])
        print('Split {}/{}: {}_{}'.format(i + 1, len(splitdates) - 1, split1, split2))

        split_ifg = []
        for ifg in ifgdates:
            im1, im2 = ifg.split('_')
            if int(im1) >= split1 and int(im2) <= split2:
                split_ifg.append(ifg)

        if len(split_ifg) == 0:
            print('No IFGs between {} -- {}. Skipping...'.format(split1, split2))

        else:
            if args.kaikoura:
                if split2 <= kaikoura:
                    pre_ix += 1
                    splitID = 'Pre' + str(pre_ix)

                elif split1 >= kaikoura:
                    post_ix += 1
                    splitID = 'Pos' + str(post_ix)
            else:
                pre_ix += 1
                splitID = str(pre_ix)

            splitdir = unwdir + 'Split' + splitID

            with open(splitfile, "a") as f:
                print('{} {} {}'.format(splitID, split1, split2), file=f)

            if os.path.exists(splitdir):
                print('{} Exists....'.format(splitdir))
            else:
                os.mkdir(splitdir)
                print('Creating {}'.format(splitdir))

            metafiles = glob.glob(os.path.join(unwdir, '*.*'))
            files = ['baselines', 'hgt']
            for file in files:
                if os.path.exists(os.path.join(unwdir, file)):
                    metafiles.append(os.path.join(unwdir, file))

            print('Soft Linking Metadata....')
            for file in metafiles:
                if not os.path.exists(os.path.join(splitdir, os.path.basename(file))):
                    os.symlink(file, os.path.join(splitdir, os.path.basename(file)))

            print('Soft Linking IFG Folders....')
            for ix, ifg in enumerate(split_ifg):
                if not os.path.exists(os.path.join(splitdir, ifg)):
                    if np.mod(ix + 1, 10) == 0:
                        print('{} / {}'.format(ix + 1, len(split_ifg)))
                    os.symlink(os.path.join(unwdir, ifg), os.path.join(splitdir, ifg))
                else:
                    print('{} / {}: {} Already Exists'.format(ix + 1, len(split_ifg), ifg))

            unassigned_tmp = unassigned.copy()
            unassigned = list(set(unassigned_tmp) - set(split_ifg))

    print('Unassigned Split-spanning IFGS:')
    for ifg in unassigned:
        print('\t {}'.format(ifg))
    print('\t{} IFGs unassigned'.format(len(unassigned)))

def merge_split_dir():
    print('Merging Timeseries')
    unassigned = ifgdates.copy()
    mergedir = unwdir + 'merge'

    if os.path.exists(mergedir):
        print('{} Exists!\n\t\tRemoving...'.format(os.path.basename(mergedir)))
        shutil.rmtree(mergedir)
    os.mkdir(mergedir)

    firstdir = True
    split_ix = 1

    for i in np.arange(len(splitdates) - 1):
        split1 = int(splitdates[i])
        split2 = int(splitdates[i + 1])

        splitdir = splitbase + 'Split' + str(split_ix) + suffix

        if os.path.exists(splitdir):
            split_ix += 1

            if firstdir:
                print('Copying metadata from {}'.format(os.path.basename(unwdir)))
                metafiles = glob.glob(os.path.join(unwdir, '*.*'))
                files = ['baselines', 'hgt']
                for file in files:
                    if os.path.exists(os.path.join(unwdir, file)):
                        metafiles.append(os.path.join(unwdir, file))

                for file in metafiles:
                    if not os.path.exists(os.path.join(mergedir, os.path.basename(file))):
                        print('\tCopying {}'.format(os.path.basename(file)))
                        shutil.copyfile(file, os.path.join(mergedir, os.path.basename(file)))
                    else:
                        print('\tAlready Exists {}'.format(os.path.basename(file)))

                firstdir = False

            print('Merging {}'.format(os.path.basename(splitdir)))
            ifgfiles = glob.glob(os.path.join(splitdir, '20*'))

            split_ifg = []
            for ix, ifg in enumerate(ifgfiles):
                pair = os.path.basename(ifg)
                split_ifg.append(pair)
                if np.mod(ix + 1, 10) == 0:
                    print('{} / {}'.format(ix + 1, len(ifgfiles)))
                ifgdir = os.path.join(splitdir, pair)
                ifgmerge = os.path.join(mergedir, pair)
                if not os.path.exists(ifgmerge):
                    os.mkdir(ifgmerge)
                shutil.copyfile(os.path.join(ifgdir, pair + '.unw'), os.path.join(ifgmerge, pair + '.unw'))
                shutil.copyfile(os.path.join(ifgdir, pair + '.unw.png'), os.path.join(ifgmerge, pair + '.unw.png'))
                os.symlink(os.path.join(ifgdir, pair + '.cc'), os.path.join(ifgmerge, pair + '.cc'))

            unassigned_tmp = unassigned.copy()
            unassigned = list(set(unassigned_tmp) - set(split_ifg))

        else:
            print('{} does not exist!'.format(splitdir))

    if len(unassigned) > 0:
        print('Copying {} unassigned IFGs'.format(len(unassigned)))
        for ix, ifg in enumerate(unassigned):
            if np.mod(ix + 1, 10) == 0:
                print('{} / {}'.format(ix + 1, len(unassigned)))

            ifgdir = os.path.join(unwdir, ifg)
            ifgmerge = os.path.join(mergedir, ifg)
            os.mkdir(ifgmerge)
            shutil.copyfile(os.path.join(ifgdir, ifg + '.unw'), os.path.join(ifgmerge, ifg + '.unw'))
            shutil.copyfile(os.path.join(ifgdir, ifg + '.unw.png'), os.path.join(ifgmerge, ifg + '.unw.png'))
            os.symlink(os.path.join(ifgdir, ifg + '.cc'), os.path.join(ifgmerge, ifg + '.cc'))

    uncorr_file = os.path.join(mergedir, 'uncorrected.txt')
    with open(uncorr_file, 'w') as f:
        for i in unassigned:
            print('{}'.format(i), file=f)

def merge_kaikoura_dir():
    print('Merging Timeseries into pre- and post- kaikoura')
    unassigned = ifgdates.copy()
    splitID = []
    start = []
    end = []

    f = open(os.path.join(args.frame_dir, 'splitdirs.txt'))
    line = f.readline()
    while line:
        if line[0] == "P":
            id, date1, date2 = [s for s in re.split('[: ]', line)]
            splitID.append(id)
            start.append(int(date1))
            end.append(int(date2[0:8]))
            line = f.readline()
        else:
            line = f.readline()

    for seis in ['Pre', 'Pos']:
        mergedir = unwdir + 'merge' + seis
        print('Preparing {}'.format(mergedir))

        if os.path.exists(mergedir):
            print('{} Exists!\n\t\tRemoving...'.format(os.path.basename(mergedir)))
            shutil.rmtree(mergedir)
        os.mkdir(mergedir)

        firstdir = True

        for id, ID in enumerate(splitID):
            if ID[:3] == seis:
                splitdir = splitbase + 'Split' + splitID[id] + suffix
                if os.path.exists(splitdir):
                    if firstdir:
                        mergeStart = start[id]
                        mergeEnd = start[id]
                        print('Copying metadata from {}'.format(os.path.basename(unwdir)))
                        metafiles = glob.glob(os.path.join(unwdir, '*.*'))
                        files = ['baselines', 'hgt']
                        for file in files:
                            if os.path.exists(os.path.join(unwdir, file)):
                                metafiles.append(os.path.join(unwdir, file))

                        for file in metafiles:
                            if not os.path.exists(os.path.join(mergedir, os.path.basename(file))):
                                print('\tCopying {}'.format(os.path.basename(file)))
                                shutil.copyfile(file, os.path.join(mergedir, os.path.basename(file)))
                            else:
                                print('\tAlready Exists {}'.format(os.path.basename(file)))

                        firstdir = False

                    print('Merging {}'.format(os.path.basename(splitdir)))
                    ifgfiles = glob.glob(os.path.join(splitdir, '20*'))

                    split_ifg = []
                    for ix, ifg in enumerate(ifgfiles):
                        pair = os.path.basename(ifg)
                        split_ifg.append(pair)
                        if np.mod(ix + 1, 10) == 0:
                            print('{} / {}'.format(ix + 1, len(ifgfiles)))
                        ifgdir = os.path.join(splitdir, pair)
                        ifgmerge = os.path.join(mergedir, pair)
                        if not os.path.exists(ifgmerge):
                            os.mkdir(ifgmerge)
                        shutil.copyfile(os.path.join(ifgdir, pair + '.unw'), os.path.join(ifgmerge, pair + '.unw'))
                        shutil.copyfile(os.path.join(ifgdir, pair + '.unw.png'), os.path.join(ifgmerge, pair + '.unw.png'))
                        os.symlink(os.path.join(ifgdir, pair + '.cc'), os.path.join(ifgmerge, pair + '.cc'))

                    mergeStart = min([mergeStart, start[id]])
                    mergeEnd = max([mergeEnd, end[id]])

                    unassigned_tmp = unassigned.copy()
                    unassigned = list(set(unassigned_tmp) - set(split_ifg))

                else:
                    print('{} does not exist!'.format(splitdir))

        assigned = []
        if len(unassigned) > 0:
            print('Identifying and copying unassigned spanner IFGs'.format(len(unassigned)))
            for ix, ifg in enumerate(unassigned):
                im1, im2 = [int(s) for s in re.split('[:_]', ifg)]
                if im1 >= mergeStart and im2 <= mergeEnd:
                    ifgdir = os.path.join(unwdir, ifg)
                    ifgmerge = os.path.join(mergedir, ifg)
                    os.mkdir(ifgmerge)
                    shutil.copyfile(os.path.join(ifgdir, ifg + '.unw'), os.path.join(ifgmerge, ifg + '.unw'))
                    shutil.copyfile(os.path.join(ifgdir, ifg + '.unw.png'), os.path.join(ifgmerge, ifg + '.unw.png'))
                    shutil.copyfile(os.path.join(ifgdir, ifg + '.cc'), os.path.join(ifgmerge, ifg + '.cc'))
                    assigned.append(ifg)
            unassigned_tmp = unassigned.copy()
            unassigned = list(set(unassigned_tmp) - set(assigned))



        uncorr_file = os.path.join(mergedir, 'uncorrected.txt')
        with open(uncorr_file, 'w') as f:
            for i in assigned:
                print('{}'.format(i), file=f)

def merge_coseismic_dir():
    global mergedir
    print('Merging Timeseries across kaikoura')
    unassigned = ifgdates.copy()
    mergedir = unwdir + 'mergeCos'

    if os.path.exists(mergedir):
        print('{} Exists!\n\t\tRemoving...'.format(os.path.basename(mergedir)))
        shutil.rmtree(mergedir)
    os.mkdir(mergedir)

    firstdir = True

    for seis in ['Pre', 'Pos']:
        splitdir = splitbase + 'merge' + seis

        if os.path.exists(splitdir):

            if firstdir:
                print('Copying metadata from {}'.format(os.path.basename(unwdir)))
                metafiles = glob.glob(os.path.join(unwdir, '*.*'))
                files = ['baselines', 'hgt']
                for file in files:
                    if os.path.exists(os.path.join(unwdir, file)):
                        metafiles.append(os.path.join(unwdir, file))

                for file in metafiles:
                    if not os.path.exists(os.path.join(mergedir, os.path.basename(file))):
                        print('\tCopying {}'.format(os.path.basename(file)))
                        shutil.copyfile(file, os.path.join(mergedir, os.path.basename(file)))
                    else:
                        print('\tAlready Exists {}'.format(os.path.basename(file)))

                firstdir = False

            print('Merging {}'.format(os.path.basename(splitdir)))
            ifgfiles = glob.glob(os.path.join(splitdir, '20*'))

            split_ifg = []
            for ix, ifg in enumerate(ifgfiles):
                pair = os.path.basename(ifg)

                split_ifg.append(pair)
                if np.mod(ix + 1, 50) == 0:
                    print('{} / {}'.format(ix + 1, len(ifgfiles)))
                ifgdir = os.path.join(splitdir, pair)
                ifgmerge = os.path.join(mergedir, pair)
                if not os.path.exists(ifgmerge):
                    os.mkdir(ifgmerge)
                shutil.copyfile(os.path.join(ifgdir, pair + '.unw'), os.path.join(ifgmerge, pair + '.unw'))
                shutil.copyfile(os.path.join(ifgdir, pair + '.unw.png'), os.path.join(ifgmerge, pair + '.unw.png'))
                os.symlink(os.path.join(ifgdir, pair + '.cc'), os.path.join(ifgmerge, pair + '.cc'))

            unassigned_tmp = unassigned.copy()
            unassigned = list(set(unassigned_tmp) - set(split_ifg))
            unassigned.sort()
        else:
            print('{} does not exist!'.format(splitdir))

    if len(unassigned) > 0:
        print('Checking and copying {} unassigned IFGs for coseismic'.format(len(unassigned)))
        n_coseismic = 0
        coseismic = []
        for ix, ifg in enumerate(unassigned):
            if int(ifg[:8]) < kaikoura and int(ifg[-8:]) > kaikoura:
                ifgdir = os.path.join(unwdir, ifg)
                ifgmerge = os.path.join(mergedir, ifg)
                os.mkdir(ifgmerge)
                shutil.copyfile(os.path.join(ifgdir, ifg + '.unw'), os.path.join(ifgmerge, ifg + '.unw'))
                shutil.copyfile(os.path.join(ifgdir, ifg + '.unw.png'), os.path.join(ifgmerge, ifg + '.unw.png'))
                os.symlink(os.path.join(ifgdir, ifg + '.cc'), os.path.join(ifgmerge, ifg + '.cc'))
                n_coseismic += 1
                coseismic.append(ifg)

    uncorr_file = os.path.join(mergedir, 'uncorrected.txt')
    with open(uncorr_file, 'w') as f:
        for i in coseismic:
            print('{}'.format(i), file=f)
    print('{} IFGs identified as coseismic (output to {}/{})'.format(n_coseismic, os.path.basename(mergedir), 'uncorrected.txt'))

def finish():
    #%% Finish
    elapsed_time = time.time() - start_time
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))
    print("\n{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)
    if args.mergeflag and args.coseismic:
        print('Outdir: {}'.format(mergedir), flush=True)

def main():
    global ifgdates

    # intialise
    start()
    init_args()

    # directory settings
    set_input_output()

    if not args.mergeflag:
        # Split data
        make_split_dir()
    else:
        if args.coseismic:
            # Merge pre and post kaikoura into 1 dataset
            merge_coseismic_dir()
        elif args.kaikoura:
            # Merge Splits into pre- and post- kaikoura datasets
            merge_kaikoura_dir()
        else:
            # Mere directories called Split 1--n
            merge_split_dir()

    # report finish
    finish()

if __name__ == '__main__':
    main()
