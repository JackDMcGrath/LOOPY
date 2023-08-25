#!/bin/bash

# Script to automated run LiCSBAS and LOOPY for an entire frame, from LiCSBAS03 to final timeseries. 
# Must be run in folder FRAME
# Data preperation is done in the order clip, mask, GACOS
# This version will identify if the frame is a TOPS frame.
# If TOPS, assume large, uncorrectable cosesimic.
#     1) Only use the listed errors for L01
#     2) Split into Pren and Postn subnetworks for L03 correction
#     3) Merge into full Pre and Post networks for L04 correction
#     4) Merge into full network, with uncorrected Kaikoura

# If not TOPS
#     1) Allow searching for errors in L01
#     2) Split into subnetworks for L03 correction
#     4) Merge into full network for L04 correction

# Needed Files:
#   ${FRAME}.clip file (optional): Used by LiCSBAS05
#   ${FRAME}.mask file (optional): Used by LiCSBAS04
#   errorLocations.txt (optional): Used for LOOPY01
#   TOPS.txt:                      List of frames that have significant Kaikoura


## FRAME VARIABLE NAMES
# GEOCdir: GEOCml10[clip,mask]GACOS Original, Uncorrected Data
# L01dir: ${GEOCdir}L01             L01 corrected data (run on full TS)

# TOPS Frames:
#   splitdir: ${L01dir}Split${i}    subnetwork (i = Pre/Post1...n)
#   corrdir:  ${splitdir}L03        L03 corrected subnetwork
#   predir=${L01dir}mergePre        Full pre-seismic network
#   finalpredir=${predir}L04        L04 corrected preseismic
#   posdir=${L01dir}mergePos        Full post-seismic netwrok
#   finalposdir=${posdir}L04        L04 corrected post-seismic
#   finaldir=${L01dir}mergeCos      Full Pre, Cos, Post seismic, with uncorrected cosesimic

# Non-TOPS Frames:
#   splitdir: ${L01dir}Split${i}    subnetwork (i = 1..n)
#   corrdir:  ${splitdir}L03        L03 corrected subnetwork
#   mergedir=${L01dir}merge         Full network 
#   finaldir=${mergedir}L04         L04 corrected full network

# Non-Split Frames
#   corrdir=${L01dir}L03            L03 corrected network
#   finaldir=${corrdir}L04          L04 corrected network


###################
## Set User Toggles
###################

splitdates=splitdates.txt # date of major earthquake to split pre- and post- seismic networks across. Leave blank for no split
error_locations=/nfs/a285/homes/eejdm/coastlines/gshhg/NZCoastRiver.txt
TOPS_file=/nfs/a285/homes/eejdm/FINALS/TOPS.txt

LiCSBAS_start="03" # LiCSBAS script to start processing from
coh_thresh="0.04" # Going to mask all pixels lower than this

n_para=$1

if [ -z $n_para ]; then
  n_para=15
fi

n_looks=10
n_unw_thre=5 # n_unw_thre for final masking of velocity
L03_unw_thre=2.5 # n_unw_thre to carry out L03 correction on (best to set to lower than n_unw_thre for masking- as masking is occurring at default 5, no point wasting time on L03 for to low a value)
L04_unw_thre=2.5 # n_unw_thre to carry out L04 correction on (best to set to lower than n_unw_thre for masking)
maxTlen=7.5 # MaxTlen for the complete timeseries
downweight=10 # Downweightinf factor for uncorrected spanners

GEOCdir=GEOCml${n_looks}

logdir="runLog"
log="$logdir/$(date +%Y%m%d%H%M)$(basename $0 .sh).log"
mkdir -p $logdir

echo "Log file:   $log"
echo ""

####################
## Set Variables
####################

curdir=`pwd`
FRAME=`echo "${curdir##*/}" | awk '{print substr($0, 1, 17)}'`

if [ ! -z $(grep ${FRAME} ${TOPS_file}) ]; then
  kaikoura='y'
else
  kaikoura='n'
fi

echo 20141001 > $splitdates
echo 20161113 >> $splitdates
echo 20180101 >> $splitdates
echo 20200101 >> $splitdates
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

copy_batch_LiCSBAS.sh  2>&1 | tee -a $log

edit_batch_LiCSBAS.sh batch_LiCSBAS.sh params.txt 2>&1 | tee -a $log

echo ' '
echo '############################' 2>&1 | tee -a $log
echo '#### Running LiCSBAS with LOOPY Correction for '${FRAME}', splitting at '${splitdate} 2>&1 | tee -a $log
echo '############################' 2>&1 | tee -a $log
echo ' ' 2>&1 | tee -a $log

echo ' ' 2>&1 | tee -a $log
echo '############################' 2>&1 | tee -a $log
echo '#### Clipping, masking, and correcting data to '${GEOCdir} 2>&1 | tee -a $log
echo '############################' 2>&1 | tee -a $log
echo ' ' 2>&1 | tee -a $log

./batch_LiCSBAS.sh 2>&1 | tee -a $log

echo ' ' 2>&1 | tee -a $log
echo '############################' 2>&1 | tee -a $log
echo '#### Running LOOPY01 to identify hard errors in '${GEOCdir} 2>&1 | tee -a $log
echo '############################' 2>&1 | tee -a $log
echo ' ' 2>&1 | tee -a $log

L01dir=${GEOCdir}L01

if [ ${kaikoura} == 'y' ]; then
  LOOPY01_find_errors.py -d $GEOCdir -c $L01dir -e $error_locations --n_para ${n_para} --onlylisted 2>&1 | tee -a $log
else
  LOOPY01_find_errors.py -d $GEOCdir -c $L01dir -e $error_locations --n_para ${n_para} 2>&1 | tee -a $log
fi

if [ ! -z $splitdates ]; then
  echo ' ' 2>&1 | tee -a $log
  echo '############################' 2>&1 | tee -a $log

  if [ ${kaikoura} == 'y' ]; then
    echo "#### Splitting Timeseries around ${splitdates} into pre- and post-sesimic" 2>&1 | tee -a $log
    echo '############################' 2>&1 | tee -a $log
    echo ' ' 2>&1 | tee -a $log
    LiCSBAS_split_TS.py -d $L01dir -s $splitdates -k 2>&1 | tee -a $log
  else
    echo "#### Splitting Timeseries around ${splitdates}" 2>&1 | tee -a $log
    echo '############################' 2>&1 | tee -a $log
    echo ' ' 2>&1 | tee -a $log
    LiCSBAS_split_TS.py -d $L01dir -s $splitdates 2>&1 | tee -a $log
  fi

  n_split=$((`wc -l splitdirs.txt | awk '{print $1}'` - 1))

  for i in $(tail -$n_split splitdirs.txt | awk '{print $1}'); do

    splitdir=${L01dir}Split${i}

    echo GEOCmldir $splitdir > params.txt
    echo start_step 11 >> params.txt
    echo end_step 12 >> params.txt
    echo p12_nullify n >> params.txt
    echo p12_null_both y >> params.txt
    echo p12_find_reference n >> params.txt 

    edit_batch_LiCSBAS.sh batch_LiCSBAS.sh params.txt 2>&1 | tee -a $log

    echo ' ' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo '#### Running null_both for '$splitdir 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo ' '

    ./batch_LiCSBAS.sh 2>&1 | tee -a $log

    echo ' '
    echo '#####################' 2>&1 | tee -a $log
    echo '#### Running LOOPY03_iterative_inversion.py for '$splitdir 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo ' ' 2>&1 | tee -a $log

    corrdir=${splitdir}L03
    LOOPY03_iterative_inversion.py -d $splitdir -c ${corrdir} --n_para ${n_para} --n_unw_r_thre ${L03_unw_thre} 2>&1 | tee -a $log

    echo ' ' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo '#### Conservatively null corrected timeseries in '$corrdir 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo ' ' 2>&1 | tee -a $log

    echo GEOCmldir $corrdir > params.txt
    echo start_step 11 >> params.txt
    echo end_step 15 >> params.txt
    echo p12_null_both n >> params.txt
    echo p12_nullify y >> params.txt
    echo p12_treat_as_bad n >> params.txt

    edit_batch_LiCSBAS.sh batch_LiCSBAS.sh params.txt 2>&1 | tee -a $log
    ./batch_LiCSBAS.sh 2>&1 | tee -a $log

  done

  echo ' ' 2>&1 | tee -a $log
  echo '#####################' 2>&1 | tee -a $log
  echo '#### Run nullification on original timeseries, so that non-corrected ifgs are nulled' 2>&1 | tee -a $log
  echo '#####################' 2>&1 | tee -a $log
  echo ' ' 2>&1 | tee -a $log
  
  echo GEOCmldir $L01dir > params.txt
  echo start_step 11 >> params.txt
  echo end_step 12 >> params.txt
  echo p12_null_both y >> params.txt
  echo p12_nullify n >> params.txt

  edit_batch_LiCSBAS.sh batch_LiCSBAS.sh params.txt 2>&1 | tee -a $log
  ./batch_LiCSBAS.sh 2>&1 | tee -a $log
   
  echo ' ' 2>&1 | tee -a $log
  echo '#####################' 2>&1 | tee -a $log
  echo '#### Merge split timeseries' 2>&1 | tee -a $log
  echo '#####################' 2>&1 | tee -a $log
  echo ' ' 2>&1 | tee -a $log

  if [ ${kaikoura} == 'y' ]; then
    LiCSBAS_split_TS.py -f ./ -d ${L01dir} -s $splitdates -c ${L01dir} -m L03 --merge -k 2>&1 | tee -a $log
    
    echo ' ' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo '#### Residual Correct pre-seismic network' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo ' ' 2>&1 | tee -a $log
    
    predir=${L01dir}mergePre
    
    timespan=`grep mergePre mergedirs.txt | awk '{print $4}'`
    preTLen=$(python -c "print(round($timespan*0.8,1))") 

    echo GEOCmldir $predir > params.txt
    echo start_step 11 >> params.txt
    echo end_step 15 >> params.txt
    echo p12_null_both n >> params.txt
    echo p12_nullify n >> params.txt     # No need to nullify again - already done this
    echo p12_find_reference y >> params.txt
    echo p13_downweight_ifg ${downweight} >> params.txt
    echo p15_n_unw_r_thre ${L04_unw_thre} >> params.txt
    echo p15_maxTlen_thre ${preTLen} >> params.txt

    edit_batch_LiCSBAS.sh batch_LiCSBAS.sh params.txt 2>&1 | tee -a $log
    ./batch_LiCSBAS.sh 2>&1 | tee -a $log

    echo ' ' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo '#### Run Residual Correction' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo ' ' 2>&1 | tee -a $log
    
    finalpredir=${predir}L04

    LOOPY04_aggressive_residuals.py -d ${GEOCdir} -t TS_${predir} -o ${finalpredir} --nonan -n ${n_para} --filter 2>&1 | tee -a $log
    
    echo ' ' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo '#### Residual Correct post-seismic network' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo ' ' 2>&1 | tee -a $log
    
    posdir=${L01dir}mergePos
    timespan=`grep mergePos mergedirs.txt | awk '{print $4}'`
    posTLen=$(python -c "print(round($timespan*0.8,1))")

    echo ' ' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo '#### Replace spanners with conservatively nulled unw' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo ' ' 2>&1 | tee -a $log
    
    for ifg in $(cat ${posdir}/uncorrected.txt); do
	    cp -f ${L01dir}/$ifg/${ifg}_con.unw ${posdir}/$ifg/${ifg}.unw
    done

    echo GEOCmldir $posdir > params.txt
    echo start_step 11 >> params.txt
    echo end_step 15 >> params.txt
    echo p12_null_both n >> params.txt
    echo p12_nullify n >> params.txt     # No need to nullify again - already done this
    echo p13_downweight_ifg ${downweight} >> params.txt
    echo p12_find_reference y >> params.txt
    echo p15_n_unw_r_thre ${L04_unw_thre} >> params.txt
    echo p15_maxTlen_thre ${posTLen} >> params.txt

    edit_batch_LiCSBAS.sh batch_LiCSBAS.sh params.txt 2>&1 | tee -a $log
    ./batch_LiCSBAS.sh 2>&1 | tee -a $log

    echo ' ' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo '#### Run Residual Correction' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo ' ' 2>&1 | tee -a $log
    
    finalposdir=${posdir}L04

    LOOPY04_aggressive_residuals.py -d ${GEOCdir} -t TS_${posdir} -o ${finalposdir} --nonan -n ${n_para} --filter 2>&1 | tee -a $log

    echo ' ' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo '#### Full Post-Seismic Corrected Time Series' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo ' ' 2>&1 | tee -a $log
    
    echo GEOCmldir $finalposdir > params.txt
    echo start_step 11 >> params.txt
    echo end_step 15 >> params.txt
    echo p12_null_both n >> params.txt
    echo p12_nullify n >> params.txt     # No need to nullify again - already done this
    echo p12_find_reference y >> params.txt
    
    edit_batch_LiCSBAS.sh batch_LiCSBAS.sh params.txt 2>&1 | tee -a $log
    ./batch_LiCSBAS.sh 2>&1 | tee -a $log
   
    echo ' ' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo '#### Merge Coseismic network' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo ' ' 2>&1 | tee -a $log
    
    finaldir=${L01dir}mergeCos

    LiCSBAS_split_TS.py -f ./ -d ${L01dir} -s $splitdates -c ${L01dir} -m L04 --merge -e 2>&1 | tee -a $log

    # Now have a network of L01, L03 then L04 corrected Pre- and Post- seismic, with uncorrected coseismic

    echo ' ' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo '#### Replace spanners with conservatively nulled unw' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo ' ' 2>&1 | tee -a $log
    
    for ifg in $(cat ${finaldir}/uncorrected.txt); do
	    cp -f ${L01dir}/$ifg/${ifg}_con.unw ${finaldir}/$ifg/${ifg}.unw
    done

  else
    LiCSBAS_split_TS.py -f ./ -d ${L01dir} -s $splitdates -c ${L01dir} -m L03 --merge 2>&1 | tee -a $log

    echo ' ' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo '#### Add in nulled data for the uncorrected' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo ' ' 2>&1 | tee -a $log

    for ifg in $(cat ${L01dir}merge/uncorrected.txt); do
	    cp -f ${L01dir}/$ifg/${ifg}_con.unw ${L01dir}merge/$ifg/${ifg}.unw
    done
    
    mergedir=${L01dir}merge

    echo ' ' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo '#### Make Conservatively nulled timeseries' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo ' ' 2>&1 | tee -a $log

    echo GEOCmldir $mergedir > params.txt
    echo start_step 11 >> params.txt
    echo end_step 15 >> params.txt
    echo p12_null_both n >> params.txt
    echo p12_nullify y >> params.txt
    echo p12_treat_as_bad n >> params.txt
    echo p13_downweight_ifg ${downweight} >> params.txt
    echo p15_n_unw_r_thre ${L04_unw_thre} >> params.txt
    echo p15_maxTlen_thre ${maxTlen} >> params.txt

    edit_batch_LiCSBAS.sh batch_LiCSBAS.sh params.txt 2>&1 | tee -a $log
    ./batch_LiCSBAS.sh 2>&1 | tee -a $log

    echo ' ' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo '#### Run Residual Correction' 2>&1 | tee -a $log
    echo '#####################' 2>&1 | tee -a $log
    echo ' ' 2>&1 | tee -a $log
    
    finaldir=${GEOCdir}L04

    LOOPY04_aggressive_residuals.py -d ${GEOCdir} -t TS_${mergedir} -o ${finaldir} --nonan -n ${n_para} --filter 2>&1 | tee -a $log

  fi 
  
else

  echo GEOCmldir $L01dir > params.txt
  echo start_step 11 >> params.txt
  echo end_step 12 >> params.txt
  echo p12_null_both y >> params.txt
  echo p12_find_reference n >> params.txt

  edit_batch_LiCSBAS.sh batch_LiCSBAS.sh params.txt 2>&1 | tee -a $log

  echo ' ' 2>&1 | tee -a $log
  echo '#####################' 2>&1 | tee -a $log
  echo '#### Running all nullification for '$L01dir 2>&1 | tee -a $log
  echo '#####################' 2>&1 | tee -a $log
  echo ' ' 2>&1 | tee -a $log

  ./batch_LiCSBAS.sh 2>&1 | tee -a $log

  echo ' ' 2>&1 | tee -a $log
  echo '#####################' 2>&1 | tee -a $log
  echo '#### Running LOOPY03_iterative_inversion.py for '$L01dir 2>&1 | tee -a $log
  echo '#####################' 2>&1 | tee -a $log
  echo ' ' 2>&1 | tee -a $log

  corrdir=${L01dir}L03
  LOOPY03_iterative_inversion.py -d $L01dir -c ${corrdir} --n_para ${n_para} --n_unw_r_thre ${L03_unw_thre} 2>&1 | tee -a $log

  echo ' ' 2>&1 | tee -a $log
  echo '#####################' 2>&1 | tee -a $log
  echo '#### Aggressively null corrected timeseries in '$corrdir 2>&1 | tee -a $log
  echo '#####################' 2>&1 | tee -a $log
  echo ' ' 2>&1 | tee -a $log

  echo GEOCmldir $corrdir > params.txt
  echo start_step 11 >> params.txt
  echo p11_unw_thre 0 >> params.txt
  echo end_step 15 >> params.txt
  echo p12_null_both n >> params.txt
  echo p12_nullify y >> params.txt
  echo p12_treat_as_bad y >> params.txt
  echo p15_n_unw_r_thre ${L04_unw_thre} >> params.txt
  echo p15_maxTlen_thre ${maxTlen} >> params.txt

  edit_batch_LiCSBAS.sh batch_LiCSBAS.sh params.txt 2>&1 | tee -a $log
  ./batch_LiCSBAS.sh 2>&1 | tee -a $log

  echo ' ' 2>&1 | tee -a $log
  echo '#####################' 2>&1 | tee -a $log
  echo '#### Run Residual Correction' 2>&1 | tee -a $log
  echo '#####################' 2>&1 | tee -a $log
  echo ' ' 2>&1 | tee -a $log
    
  finaldir=${GEOCdir}L04

  LOOPY04_aggressive_residuals.py -d ${GEOCdir} -t TS_${corrdir} -o ${finaldir} --nonan -n ${n_para} --filter 2>&1 | tee -a $log

fi

echo ' ' 2>&1 | tee -a $log
echo '#####################' 2>&1 | tee -a $log
echo '#### Running Full Time-series for ' $finaldir 2>&1 | tee -a $log
echo '#####################' 2>&1 | tee -a $log
echo ' ' 2>&1 | tee -a $log

echo GEOCmldir $finaldir > params.txt
echo start_step 11 >> params.txt
echo end_step 15 >> params.txt
echo p11_unw_thre 0.05 >> params.txt
echo p12_nullify n >> params.txt
echo p12_find_reference y >> params.txt
echo p13_downweight_ifg ${downweight} >> params.txt
echo p15_n_unw_r_thre ${n_unw_thre} >> params.txt
echo p15_maxTlen_thre ${maxTlen} >> params.txt

edit_batch_LiCSBAS.sh batch_LiCSBAS.sh params.txt 2>&1 | tee -a $log

./batch_LiCSBAS.sh 2>&1 | tee -a $log

echo ' '  2>&1 | tee -a $log
echo '#####################' 2>&1 | tee -a $log
echo '#### COMPLETED A TIMESERIES WITH L03 CORRECTIONS for '${FRAME} 2>&1 | tee -a $log
echo '#####################' 2>&1 | tee -a $log
echo ' ' 2>&1 | tee -a $log

echo ' '  2>&1 | tee -a $log
echo '#####################' 2>&1 | tee -a $log
echo '#### Compare to original for '${FRAME} 2>&1 | tee -a $log
echo '#####################' 2>&1 | tee -a $log
echo ' ' 2>&1 | tee -a $log

LOOPY05_compare_corrections.py -f ./ -i ${GEOCdir} -c ${finaldir} -o comp_origVfinal -n ${n_para} --reset

echo ' ' 2>&1 | tee -a $log
echo '############################' 2>&1 | tee -a $log
echo '#### LOOPY-LiCSBAS for '${FRAME}' complete!' 2>&1 | tee -a $log
echo '############################' 2>&1 | tee -a $log
echo ' ' 2>&1 | tee -a $log
