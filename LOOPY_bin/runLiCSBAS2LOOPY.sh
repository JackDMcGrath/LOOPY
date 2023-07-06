#!/bin/bash

# Script to automated run LiCSBAS and LOOPY for an entire frame, from download
# final timeseries. Must be run in folder FRAME

# Needed Files:
#   splitfiles.txt
#   errorLocations.txt (optional)
#   mask or clip file (optional)

curdir=`pwd`
FRAME=`echo "${curdir##*/}" | awk '{print substr($0, 1, 17)}'`

kaikoura=20161113

if [ -f kaikoura.dates ]; then
  rm -f kaikoura.dates
  touch kaikoura.dates
fi

download=0
prepTS=0
splitTS=0
mergeTS=0
uncorrTS=0
compTS=0

mask_correct="y"
corr_correct="n"

startdate=""
enddate=""

datadir='/scratch/eejdm/FINALS/LOOPY/data/'${FRAME}

echo ' '
echo '############################'
echo '#### Running LiCSBAS with LOOPY Correction for '${FRAME}
echo '############################'
echo ' '

splitfile='../split_dates.txt'
errorLocations='../gmtErrors.txt'
maskclipdir='../maskclip'

para=45
nlook=10
removeOldSplits=0

n_lines=`wc -l $splitfile | awk '{print $1}'`
n_split=`echo $((${n_lines}-1))`

UNWdir='GEOCml'${nlook}
GEOCdir=${UNWdir}


echo ' '
echo '############################'
echo '#### Setting LiCSBAS Parameters'
echo '############################'
echo ' '

if [ -z $startdate ]; then
  startdate=20140101
fi

if [ -z $enddate ]; then
  enddate=20230601
fi

if [ -f batch_LiCSBAS.sh ]; then
  rm -r batch_LiCSBAS.sh
fi

copy_batch_LiCSBAS.sh

echo p01_frame ${FRAME} > batch_param.txt
if [ ${download} -eq 1 ]; then
  echo start_step 01 >> batch_param.txt
  echo p01_start_date ${startdate} >> ./batch_param.txt
  echo p01_end_date ${enddate} >> ./batch_param.txt
else

  echo ' '
  echo '############################'
  echo '#### Soft Linking GEOC, GACOS and GEOCml10'
  echo '############################'
  echo ' '

  echo start_step 02 >> batch_param.txt
  # Remove soft links to data
  if [ -d GEOC ] && [ -L GEOC ]; then
    rm -f GEOC
  else
    rm -rf GEOC
  fi
  if [ -d GEOCml10 ] && [ -L GEOCml10 ]; then
    rm -f GEOCml10
  else
    rm -rf GEOCml10
  fi
  if [ -d GACOS ] && [ -L GACOS ]; then
    rm -f GACOS
  else
    rm -rf GACOS
  fi

  mkdir GEOC
  ln -s ${datadir}/GEOC/* GEOC/
  mkdir GEOCml10
  ln -s ${datadir}/GEOCml10/* GEOCml10/
  mkdir GACOS
  ln -s ${datadir}/GACOS/* GACOS/

  cd GEOC
  for ifg in $(ls -d 20*); do
    im1=`echo ${ifg} | awk '{print substr($1,1,8)}'`
    im2=`echo ${ifg} | awk '{print substr($1,10,8)}'`
    if [ ${im1} -lt ${startdate} ] || [ ${im2} -gt ${enddate} ]; then
      rm -f ${ifg}
    else
      if [ ${im1} -lt ${kaikoura} ] && [ ${im2} -gt ${kaikoura} ]; then
        echo ${ifg} >> kaikoura.dates
      fi
    fi
  done
  cd ..

  echo '#### GEOC Linked'

  cd GEOCml10
  for ifg in $(ls -d 20*); do
    im1=`echo ${ifg} | awk '{print substr($1,1,8)}'`
    im2=`echo ${ifg} | awk '{print substr($1,10,8)}'`
    if [ ${im1} -lt ${startdate} ] || [ ${im2} -gt ${enddate} ]; then
      rm -f ${ifg}
    fi
  done
  cd ..

  echo '#### GEOCml10 Linked'

  cd GACOS
  for epoch in $(ls -d 20*); do
    im1=`echo ${epoch} | awk '{print substr($1,1,8)}'`
    if [ ${im1} -lt ${startdate} ] || [ ${im2} -gt ${enddate} ]; then
      rm -f ${im1}
    fi
  done
  cd ..

  echo '#### GACOS Linked'

fi

echo end_step 11 >> batch_param.txt
echo cometdev 1 >> batch_param.txt
echo nlook ${nlook} >> batch_param.txt
echo GEOCmldir ${UNWdir} >> batch_param.txt
echo n_para ${para} >> batch_param.txt
echo check_only n >> batch_param.txt
echo do03op_GACOS n >> batch_param.txt
echo p01_get_gacos y >> batch_param.txt
# echo p01_start_date 20180101 >> ./batch_param.txt
#echo p01_end_date 20180101 >> ./batch_param.txt
echo p12_multi_prime y >> ./batch_param.txt
echo p12_treat_as_bad n >> ./batch_param.txt
echo p13_null_noloop n >> ./batch_param.txt

if [ -f ${maskclipdir}/${FRAME}.mask ]; then
  if [ ! -f ${FRAME}.clip ]; then
    ln -s ${maskclipdir}/${FRAME}.mask .
  fi
  maskfile=${FRAME}.mask
  echo do04op_mask y >> batch_param.txt
  echo p04_mask_range_file $maskfile >> batch_param.txt
  GEOCdir=${GEOCdir}mask
fi

if [ -f ${maskclipdir}/${FRAME}.clip ]; then
  if [ ! -f ${FRAME}.clip ]; then
    ln -s ${maskclipdir}/${FRAME}.clip .
  fi
  clipfile=${FRAME}.clip
  echo do05op_clip y >> batch_param.txt
  echo p05_clip_range_geo `cat $clipfile` >> batch_param.txt
  GEOCdir=${GEOCdir}clip
fi

noLOOPYdir=${GEOCdir}

edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt


echo ' '
echo '############################'
echo '#### Download and Prepare '${GEOCdir}
echo '############################'
echo ' '

if [ ${prepTS} -eq 1 ]; then
  ./batch_LiCSBAS.sh # Download Data, multilook, run step 11 to find and remove bad IFGs
fi

if [ `wc -l TS_${GEOCdir}/info/11bad_ifg.txt | awk '{print $1}'` -gt 0 ]; then
  echo ' '
  echo '############################'
  echo '#### Removing IFGs from '${GEOCdir}' that failed Coherence and Coverage checks'
  echo '############################'
  echo ' '

  for ifg in `cat TS_${GEOCdir}/info/11bad_ifg.txt`; do
    echo ${ifg}
    rm -rf ${GEOCdir}/${ifg}
  done

  rm -rf TS_GEOCml10 # Tidy up
fi

# Now resetting Step 11 Thresholds so that we never disgard an IFG again (raises problems when nulling)
echo p11_unw_thre 0  > batch_param.txt
echo p11_coh_thre 0  >> batch_param.txt

edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

echo ' '
echo '############################'
echo '#### Apply GACOS to untouched data'
echo '############################'
echo ' '

echo GEOCmldir ${noLOOPYdir} > batch_param.txt
echo start_step 03 >> batch_param.txt
echo end_step 03 >> batch_param.txt
echo do03op_GACOS y >> batch_param.txt

edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

if [ ${prepTS} -eq 1 ]; then
  ./batch_LiCSBAS.sh # Create GACOS corrected UNW for comparative purposes
fi

noLOOPYdir=${noLOOPYdir}GACOS

echo ' '
echo '############################'
echo '#### Run LOOPY01_find_errors'
echo '############################'
echo ' '

corrdir=${GEOCdir}L01
if [ ${prepTS} -eq 1 ]; then
  if [ ${FRAME} == 073D_13256_001823 ]; then
    LOOPY01_find_errors.py -d ${GEOCdir} -e ${errorLocations} -c ${corrdir} --onlylisted --n_para ${para}
  else
    LOOPY01_find_errors.py -d ${GEOCdir} -e ${errorLocations} -c ${corrdir} --n_para ${para}
  fi
fi

GEOCdir=${corrdir}

echo ' '
echo '############################'
echo '#### Apply GACOS'
echo '############################'
echo ' '

LiCSBAS_reset_nulls.sh ${GEOCdir}  # Just in case

echo GEOCmldir ${GEOCdir} > batch_param.txt
echo start_step 03 >> batch_param.txt
echo end_step 03 >> batch_param.txt
echo do03op_GACOS y >> batch_param.txt

edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

if [ ${prepTS} -eq 1 ]; then
  ./batch_LiCSBAS.sh # Apply GACOS to full L01 corrected dataset
fi

GEOCdir=${GEOCdir}GACOS

if [ ${splitTS} -eq 1 ]; then

  echo ' '
  echo '############################'
  echo '#### Split Timeseries'
  echo '############################'
  echo ' '

  if [ ${removeOldSplits} -eq 1 ]; then
    if [ ! `find . -iwholename '*Split*' | wc -l` -eq 0 ]; then
      for i in $(seq 5 -1 0); do
        echo "    Removing [TS_]"${GEOCdir}"Split* in "${i}
        sleep 1
      done
      rm -rf *${GEOCdir}Split*
    fi
    if [ ! `find . -iwholename '*GEOC*merge*' | wc -l` -eq 0 ]; then
      for i in $(seq 5 -1 0); do
        echo "    Removing [TS_]"${GEOCdir}"merge* in "${i}
        sleep 1
      done
      rm -rf *${GEOCdir}merge*
    fi
  fi

  LiCSBAS_split_TS.py -d $GEOCdir -s $splitfile  # Split data into smaller, more manageable chuncks

  n_split=`ls -d ${GEOCdir}Split* | wc -l`

  for i in $(seq 1 $n_split); do

    splitdir=${GEOCdir}Split${i}

    echo ' '
    echo '#####################'
    echo '#### Running Uncorrected Timeseries of Split '${i}' for comparison'
    echo '#####################'
    echo ' '

    if [ -f batch_LiCSBAS.sh ]; then
      rm -r batch_LiCSBAS.sh
    fi

    copy_batch_LiCSBAS.sh

    echo start_step 11 > ./batch_param.txt
    echo end_step 15 >> ./batch_param.txt
    echo cometdev 0 >> ./batch_param.txt
    echo check_only n >> batch_param.txt
    echo p12_treat_as_bad n >> ./batch_param.txt
    echo p13_null_noloop n >> ./batch_param.txt
    echo GEOCmldir ${splitdir} >> ./batch_param.txt

    edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

    ./batch_LiCSBAS.sh  # Create uncorrected timeseries for comparison. No nulling

    mv TS_${splitdir} TS_${splitdir}_uncorrected
    ln -s ${splitdir} ${splitdir}_uncorrected  # For comparison if wanted
    nocorrSplit=${splitdir}_uncorrected
    echo ' '
    echo '#####################'
    echo '#### Correcting Split '${i}
    echo '#####################'
    echo ' '

    LOOPY03_correction_inversion.py -d ${splitdir} -t TS_${splitdir}_uncorrected -c ${splitdir}L03 --coast --n_para ${para}

    splitdir=${splitdir}L03
    final_dir=${splitdir}
    echo GEOCmldir ${splitdir} > ./batch_param.txt

    if [ $mask_correct = 'y' ]; then
      echo cometdev 1 >> ./batch_param.txt
      echo p12_loop_thre 100 >> ./batch_param.txt
      echo p12_multi_prime y >> ./batch_param.txt
      echo p12_treat_as_bad y >> ./batch_param.txt
      echo p13_null_noloop y >> ./batch_param.txt
      echo p15_coh_thre 0.05 >> ./batch_param.txt
      echo p15_n_unw_r_thre 0.1 >> ./batch_param.txt
      echo p15_vstd_thre 1000 >> ./batch_param.txt
      echo p15_maxTlen_thre 0 >> ./batch_param.txt
      echo p15_n_gap_thre 0 >> ./batch_param.txt
      echo p15_stc_thre 1000 >> ./batch_param.txt
      echo p15_n_ifg_noloop_thre 1000 >> ./batch_param.txt
      echo p15_n_loop_err_thre 1000 >> ./batch_param.txt
      echo p15_resid_rms_thre 1000 >> ./batch_param.txt

      edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt
      ./batch_LiCSBAS.sh  # Run Nullified TS to create a "Good" velocity field

      LiCSBAS_reset_nulls.sh ${splitdir}  # Reset Nullification before correction

      maskdir=${splitdir}_intMask
      LOOPY04_aggressiveResiduals.py -d ${splitdir} -t TS_${splitdir} -o ${maskdir} --apply_mask
      cp -r TS_${splitdir}/130resid/integer_correction ${maskdir}
      final_dir=${maskdir}
      echo GEOCmldir ${maskdir} > ./batch_param.txt
    fi

    if [ $corr_correct = 'y' ]; then
      edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

      ./batch_LiCSBAS.sh # Run Nullified TS on mask corrected data

      LiCSBAS_reset_nulls.sh ${maskdir} # Reset nullification before correction

      corrdir=${splitdir}_intCorr

      LOOPY04_aggressiveResiduals.py -d ${splitdir} -t TS_${maskdir} -o ${corrdir}
      cp -r TS_${maskdir}/130resid/integer_correction ${corrdir}

      final_dir=${corrdir}
      echo GEOCmldir ${corrdir} > ./batch_param.txt
    fi


    echo cometdev 0 >> ./batch_param.txt
    echo p12_treat_as_bad n >> ./batch_param.txt
    echo p13_null_noloop n >> ./batch_param.txt

    edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

    ./batch_LiCSBAS.sh # Run no null TS on corrected split dataset

  LOOPY05_compare_correction.py -f ./ -i ${nocorrSplit} -c ${final_dir} -o comp_Split${i} -n ${para} --reset

  done

fi

if [ ${mergeTS} -eq 1 ]; then

  mergedir=${GEOCdir}merge

  echo ' '
  echo '#####################'
  echo '#### Merging Split Timeseries into '${mergedir}
  echo '#####################'
  echo ' '

  if [ ${corr_correct} = 'y' ]; then
    suffix=L03_intCorr;
  elif [ ${mask_correct} = 'y' ]; then
    suffix=L03_intMask;
  else
    suffix=L03
  fi


  if [ ! `ls -d *${mergedir}* | wc -l` -eq 0 ]; then
    for i in $(seq 5 -1 0); do
      echo "    Removing [TS_]"${mergedir}"* in "${i}
      sleep 1
    done
    rm -rf *${mergedir}*
  fi

  LiCSBAS_split_TS.py -d $GEOCdir -s $splitfile -c $GEOCdir -m ${suffix}

  LiCSBAS_reset_nulls.sh ${mergedir}  # Pre-cautionary

  echo GEOCmldir ${mergedir} > ./batch_param.txt
  echo start_step 11 >> batch_param.txt
  echo end_step 15 >> batch_param.txt
  echo cometdev 1 >> ./batch_param.txt
  echo p12_treat_as_bad y >> ./batch_param.txt
  echo p13_null_noloop y >> ./batch_param.txt

  edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

  ./batch_LiCSBAS.sh # Generate Nulled Timeseries of whole dataset
  LiCSBAS_reset_nulls.sh ${mergedir}  # So that LOOPY03 will corrects as many UNWs in a pixel as possible

  echo ' '
  echo '#####################'
  echo '#### Correcting Full Timeseries'
  echo '#####################'
  echo ' '

  # Run Loopy03, but also inverse solve some random pixels that contain gaps but could be good, to improve interpolations
  # (Selection is based off the results of the time series, so its ok that we have reset the nulls)
  LOOPY03_correction_inversion.py -d ${mergedir} -t TS_${mergedir} -c ${mergedir}L03 --coast --randpix 500 --n_para 20

  mergedir=${mergedir}L03

  echo GEOCmldir ${mergedir} > ./batch_param.txt
  echo start_step 11 >> batch_param.txt
  echo end_step 15 >> batch_param.txt
  echo cometdev 1 >> ./batch_param.txt
  echo p12_treat_as_bad y >> ./batch_param.txt
  echo p13_null_noloop y >> ./batch_param.txt

  edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

  ./batch_LiCSBAS.sh # Generate nulled L03 Timeseries

  LiCSBAS_reset_nulls.sh ${mergedir}

  final_dir=${mergedir}

  if [ $mask_correct = 'y' ]; then
    maskdir=${mergedir}_intMask
    if [ $corr_correct = 'n' ]; then
      LiCSBAS_reset_nulls.sh ${noLOOPYdir} # (there will definitely not be nulled pixels in this data set, but check anyway)
      LOOPY04_aggressiveResiduals.py -d ${noLOOPYdir} -t TS_${mergedir} -o ${maskdir} -l ${mergedir}/uncorrected.txt --apply_mask
      final_dir=${maskdir}
    else
      LOOPY04_aggressiveResiduals.py -d ${GEOCdir} -t TS_${mergedir} -o ${maskdir} -l ${mergedir}/uncorrected.txt --apply_mask
    fi
    cp -r TS_${mergedir}/130resid/integer_correction ${maskdir}


    echo GEOCmldir ${maskdir} > ./batch_param.txt
    if [ $corr_correct = 'y' ]; then
      echo start_step 11 >> batch_param.txt
      echo end_step 15 >> batch_param.txt
      echo cometdev 1 >> ./batch_param.txt
      echo p12_treat_as_bad y >> ./batch_param.txt
      echo p13_null_noloop y >> ./batch_param.txt

    edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

    ./batch_LiCSBAS.sh # Run new time series where spanners have been corrected as well

    fi
    LiCSBAS_reset_nulls.sh ${maskdir}

  fi

  if [ $corr_correct = 'y' ]; then
    corrdir=${mergedir}_intCorr

    # Apply final correction to untouched data
    LiCSBAS_reset_nulls.sh ${noLOOPYdir} # (there will definitely not be nulled pixels in this data set, but check anyway)
    LOOPY04_aggressiveResiduals.py -d ${noLOOPYdir} -t TS_${maskdir} -o ${corrdir} -l ${mergedir}/uncorrected.txt
    cp -r TS_${maskdir}/130resid/integer_correction ${corrdir}
    final_dir=${corrdir}
  fi

  echo ' '
  echo '#####################'
  echo '#### Running LiCSBAS on corrected dataset ('${noLOOPYdir}' corrected to '${final_dir}') using default masking'
  echo '#####################'
  echo ' '

  if [ -f batch_LiCSBAS.sh ]; then
    rm -r batch_LiCSBAS.sh
  fi

  copy_batch_LiCSBAS.sh

  if [ ! ${mergeTS} -eq 1 ]; then
    final_dir=${GEOCdir}
  fi

  echo GEOCmldir ${final_dir} > ./batch_param.txt
  echo cometdev 1 >> ./batch_param.txt
  echo start_step 11 >> batch_param.txt
  echo end_step 15 >> batch_param.txt
  echo check_only n >> batch_param.txt
  echo p12_multi_prime y >> ./batch_param.txt
  echo p12_treat_as_bad n >> ./batch_param.txt
  echo p13_null_noloop n >> ./batch_param.txt
  echo p11_TSdir TS_${final_dir}_FINAL >> ./batch_param.txt
  echo p12_TSdir TS_${final_dir}_FINAL >> ./batch_param.txt
  echo p13_TSdir TS_${final_dir}_FINAL >> ./batch_param.txt
  echo p14_TSdir TS_${final_dir}_FINAL >> ./batch_param.txt
  echo p15_TSdir TS_${final_dir}_FINAL >> ./batch_param.txt
  ln -s ${final_dir} ${final_dir}_FINAL
  final_dir=${final_dir}_FINAL
  edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

  ./batch_LiCSBAS.sh # Final run of LiCSBAS, gently nulling the truly awful

  echo ' '
  echo '#####################'
  echo '#### FULLY CORRECTED TIME SERIES COMPLETE!!!!!!'
  echo '#### Look in TS_'${final_dir}
  echo '#####################'
  echo ' '

fi

if [ -f batch_LiCSBAS.sh ]; then
  rm -r batch_LiCSBAS.sh
fi

copy_batch_LiCSBAS.sh

if [ $uncorrTS} -eq 1 ]; then

if [ ! ${GEOCdir} = ${noLOOPYdir} ]; then
  echo ' '
  echo '#####################'
  echo '#### Create Timeseries from Uncorrected Dataset (inc. L01 correction): '${GEOCdir}
  echo '#####################'
  echo ' '

  echo GEOCmldir ${GEOCdir} > ./batch_param.txt
  echo n_para ${para} >> ./batch_param.txt
  echo check_only n >> ./batch_param.txt
  echo cometdev 1 >> ./batch_param.txt
  echo start_step 11 >> batch_param.txt
  echo end_step 15 >> batch_param.txt
  echo p12_treat_as_bad n >> ./batch_param.txt
  echo p13_null_noloop n >> ./batch_param.txt

  edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

  ./batch_LiCSBAS.sh
fi

  echo ' '
  echo '############################'
  echo '#### Create Timeseries from fully Uncorrected Dataset: '${noLOOPYdir}
  echo '############################'
  echo ' '

  echo GEOCmldir ${noLOOPYdir} > batch_param.txt
  echo start_step 11 >> batch_param.txt
  echo end_step 15 >> batch_param.txt
  echo do03op_GACOS y >> batch_param.txt
  echo cometdev 1 >> ./batch_param.txt
  echo p12_treat_as_bad n >> ./batch_param.txt
  echo p13_null_noloop n >> ./batch_param.txt

  edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

  ./batch_LiCSBAS.sh # Make a gently nulled timeseries from the uncorrected data

fi

if [ ${compTS} -eq 1 ]; then
  echo ' '
  echo '#####################'
  echo '#### Comparing Corrected and Uncorrected datasets'
  echo '#####################'
  echo ' '

  LOOPY05_compare_correction.py -f ./ -i ${noLOOPYdir} -c ${mergedir} -o comp_${noLOOPYdir}_L03 -n ${para} --reset
  if [ $mask_correct = 'y' ]; then
    LOOPY05_compare_correction.py -f ./ -i ${noLOOPYdir} -c ${final_dir} -o comp_${noLOOPYdir}_intMask -n ${para} --reset
  fi
  if [ $corr_correct = 'y' ]; then
    LOOPY05_compare_correction.py -f ./ -i ${noLOOPYdir} -c ${final_dir} -o comp_${noLOOPYdir}_intCorr -n ${para} --reset
  fi
fi

echo ' '
echo '############################'
echo '#### LOOPY-LiCSBAS for '${FRAME}' complete!'
echo '############################'
echo ' '
