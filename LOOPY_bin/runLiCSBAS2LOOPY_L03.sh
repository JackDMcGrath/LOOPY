#!/bin/bash

# Script to automated run LiCSBAS and LOOPY for an entire frame, from download to
# final timeseries. Must be run in folder FRAME

# In this version, we can do lots of splits. We just don't care that one of them is Kaikoura

# Needed Files:
#   errorLocations.txt (optional)
#   mask or clip file (optional)

# UNWdir: GEOCml10 - multilooked data ONLY
# noLOOPYdir: GEOCml10[mask,clip]GACOS - Data never touched by LOOPY
# GEOCdir: GEOCml10[clip,mask]L01GACOS - Reference LOOPY data containing ALL the unw. Core of the LOOPY variable names
# splitdir: ${GEOCdir}Split[Pre, Pos][1-n]Splitting the data into manageable chunks
# L03dir: ${splitdir}L03 L03 correction of splitdir
# maskdir: ${L03dir}_intMask: Interger mask correction of L03dir
# nocorrSplit: Uncorrected Splitdir
# final_dir: last corrected data set

# mergebasedir ${GEOCdir}merge Base name for merges
# suffix: End of the correction names for splitdirs (e.g. L03_intMask)


###################
## Set User Toggles
###################

splitdates=splitdates.txt # date of major earthquake to split pre- and post- seismic networks across. Leave blank for no split

LiCSBAS_start="03" # LiCSBAS script to start processing from
coh_thresh="0.04" # Going to mask all pixels lower than this
error_locations=/nfs/a285/homes/eejdm/coastlines/gshhg/NZCoastRiver.txt
n_para=$1

if [ -z $n_para ]; then
  n_para=15
fi

n_looks=10

GEOCdir=GEOCml${n_looks}

####################
## Set Variables
####################

curdir=`pwd`
FRAME=`echo "${curdir##*/}" | awk '{print substr($0, 1, 17)}'`

az=`echo $FRAME | head -c 4 | tail -c 1`

echo 20141001 > $splitdates
echo 20161113 >> $splitdates
if [ az == 'A' ]; then
  echo 20190101 >> $splitdates
else
  echo 20200101 >> $splitdates
fi
echo 20230101 >> $splitdates

if [ -f splitdirs.txt ]; then
  rm -f splitdirs.txt
fi

####################
## Prep batch_LiCSBAS
####################

echo GEOCmldir $GEOCdir > params.txt
echo start_step $LiCSBAS_start >> params.txt
echo end_step 05 >> params.txt
echo order_op03_05 05 04 03 >> params.txt
echo n_para $n_para >> params.txt
echo nlook $n_looks >> params.txt

if [ -f $FRAME.clip ]; then
  clip_range=`cat $FRAME.clip`
  echo p05_clip_range_geo $clip_range >> params.txt
  echo do05op_clip y >> params.txt 
  GEOCdir=${GEOCdir}clip
fi

echo do04op_mask y >> params.txt
echo p04_mask_ifg_coh_thre $coh_thresh >> params.txt
GEOCdir=${GEOCdir}mask

if [ -f $FRAME.mask ]; then
  echo p04_mask_range_file $FRAME.mask >> params.txt
fi

if [ $LiCSBAS_start -le 03 ]; then
  GEOCdir=${GEOCdir}GACOS
fi

if [ -f batch_LiCSBAS.sh ]; then
  rm -f batch_LiCSBAS.sh
fi

copy_batch_LiCSBAS.sh 

edit_batch_LiCSBAS.sh batch_LiCSBAS.sh params.txt

echo ' '
echo '############################'
echo '#### Running LiCSBAS with LOOPY Correction for '${FRAME}', splitting at '${splitdate}
echo '############################'
echo ' '

echo ' '
echo '############################'
echo '#### Clipping, masking, and correcting data to '${GEOCdir}
echo '############################'
echo ' '

./batch_LiCSBAS.sh

echo ' '
echo '############################'
echo '#### Running LOOPY01 to identify hard errors in '${GEOCdir}
echo '############################'
echo ' '

L01dir=${GEOCdir}L01

LOOPY01_find_errors.py -d $GEOCdir -c $L01dir -e $error_locations --n_para ${n_para}

if [ ! -z $splitdates ]; then
  echo ' '
  echo '############################'
  echo '#### Splitting Timeseries around '${splitdates}
  echo '############################'
  echo ' '

  LiCSBAS_split_TS.py -d $L01dir -s $splitdates

  n_split=$((`wc -l splitdirs.txt | awk '{print $1}'` - 1))

  for i in $(tail -$n_split splitdirs.txt | awk '{print $1}'); do

    splitdir=${L01dir}Split${i}

    echo GEOCmldir $splitdir > params.txt
    echo start_step 11 >> params.txt
    echo end_step 12 >> params.txt
    echo p12_nullify n >> params.txt
    echo p12_null_both y >> params.txt
    echo p12_find_reference n >> params.txt 

    edit_batch_LiCSBAS.sh batch_LiCSBAS.sh params.txt

    echo ' '
    echo '#####################'
    echo '#### Running null_both for '$splitdir
    echo '#####################'
    echo ' '

    ./batch_LiCSBAS.sh

    echo ' '
    echo '#####################'
    echo '#### Running LOOPY03_iterative_inversion.py for '$splitdir
    echo '#####################'
    echo ' '

    corrdir=${splitdir}L03
    LOOPY03_iterative_inversion.py -d $splitdir -c ${corrdir} --n_para ${n_para}

    echo ' '
    echo '#####################'
    echo '#### Aggressively null corrected timeseries in '$corrdir
    echo '#####################'
    echo ' '

    echo GEOCmldir $corrdir > params.txt
    echo start_step 11 >> params.txt
    echo end_step 12 >> params.txt
    echo p12_null_both n >> params.txt
    echo p12_nullify y >> params.txt
    echo p12_treat_as_bad y >> params.txt

    edit_batch_LiCSBAS.sh batch_LiCSBAS.sh params.txt
    ./batch_LiCSBAS.sh

  done

  # ## Rename splits since we're only running 1 pre and 1 post seismic network
  # mv ${L01dir}SplitPre1L03 ${L01dir}mergePre
  # mv ${L01dir}SplitPos1L03 ${L01dir}mergePos


  echo ' '
  echo '#####################'
  echo '#### Merge split timeseries'
  echo '#####################'
  echo ' '

  LiCSBAS_split_TS.py -f ./ -d ${L01dir} -s $splidates -c ${L01dir} --merge

  GEOCdir=${L01dir}merge

else

    echo GEOCmldir $L01dir > params.txt
    echo start_step 11 >> params.txt
    echo end_step 12 >> params.txt
    echo p12_null_both y >> params.txt
    echo p12_find_reference n >> params.txt

    edit_batch_LiCSBAS.sh batch_LiCSBAS.sh params.txt

    echo ' '
    echo '#####################'
    echo '#### Running all nullification for '$L01dir
    echo '#####################'
    echo ' '

    ./batch_LiCSBAS.sh

    echo ' '
    echo '#####################'
    echo '#### Running LOOPY03_iterative_inversion.py for '$L01dir
    echo '#####################'
    echo ' '

    corrdir=${L01dir}L03
    LOOPY03_iterative_inversion.py -d $L01dir -c ${corrdir} --n_para ${n_para}

    echo ' '
    echo '#####################'
    echo '#### Aggressively null corrected timeseries in '$corrdir
    echo '#####################'
    echo ' '

    echo GEOCmldir $corrdir > params.txt
    echo start_step 11 >> params.txt
    echo end_step 12 >> params.txt
    echo p12_null_both n >> params.txt
    echo p12_nullify y >> params.txt
    echo p12_treat_as_bad y >> params.txt

    edit_batch_LiCSBAS.sh batch_LiCSBAS.sh params.txt
    ./batch_LiCSBAS.sh

    GEOCdir=$corrdir

    fi

echo ' '
echo '#####################'
echo '#### Running Full Time-series for ' $GEOCdir
echo '#####################'
echo ' '

echo GEOCmldir $GEOCdir > params.txt
echo start_step 11 >> params.txt
echo end_step 15 >> params.txt
echo p12_nullify n >> params.txt
echo p12_find_reference y >> params.txt

edit_batch_LiCSBAS.sh batch_LiCSBAS.sh params.txt

./batch_LiCSBAS.sh

echo ' '
echo '#####################'
echo '#### COMPLETED A TIMESERIES WITH L03 CORRECTIONS for '${FRAME}
echo '#####################'
echo ' '

echo ' '
echo '############################'
echo '#### LOOPY-LiCSBAS for '${FRAME}' complete!'
echo '############################'
echo ' '