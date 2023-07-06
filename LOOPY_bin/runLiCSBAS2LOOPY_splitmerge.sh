#!/bin/bash

# Script to automated run LiCSBAS and LOOPY for an entire frame, from download
# final timeseries. Must be run in folder FRAME

# Needed Files:
#   splitfiles.txt
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


curdir=`pwd`
FRAME=`echo "${curdir##*/}" | awk '{print substr($0, 1, 17)}'`

kaikoura=20161113 # date of major earthquake to split pre- and post- seismic networks across

if [ -f kaikoura.dates ]; then
  rm -f kaikoura.dates
  touch kaikoura.dates
fi

download=0
prepTS=1
splitTS=1
mergeTS=1
uncorrTS=1
compTS=1

mask_correct="y"
corr_correct="n"

startdate=""
enddate=""

echo ' '
echo '############################'
echo '#### Running LiCSBAS with LOOPY Correction for '${FRAME}
echo '############################'
echo ' '

procdir=/scratch/eejdm/FINALS/LOOPY
splitfile=${procdir}/split_dates.txt
errorLocations=${procdir}/gmtErrors.txt
maskclipdir=${procdir}/maskclip
datadir=${procdir}/data/${FRAME}

para=45
nlook=10
removeOldSplits=1

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
  echo '#### Soft Linking GEOC, GACOS and GEOCml10 between '${startdate}'--'${enddate}
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
    if [ ! `find . -iwholename 'splitdirs*txt' | wc -l` -eq 0 ]; then
      for txt in $(ls splitdirs*txt); do
        rm -f ${txt}
      done
    fi
  fi

  LiCSBAS_split_TS.py -d $GEOCdir -s $splitfile -k # Split data into smaller, more manageable chunks, labelled base dof Earthquake

  n_split=$((`wc -l splitdirs.txt | awk '{print $1}'` - 1))

  for i in $(tail -$n_split splitdirs.txt | awk '{print $1}'); do

    splitdir=${GEOCdir}Split${i}

    echo ' '
    echo '#####################'
    echo '#### Running Uncorrected Timeseries of '${i}' for comparison'
    echo '#####################'
    echo ' '

    if [ -f batch_LiCSBAS.sh ]; then
      rm -r batch_LiCSBAS.sh
    fi

    copy_batch_LiCSBAS.sh

    nocorrSplit=${splitdir}_uncorrected
    ln -s ${splitdir} ${nocorrSplit}  # For comparison if wanted

    echo start_step 11 > ./batch_param.txt
    echo end_step 15 >> ./batch_param.txt
    echo cometdev 0 >> ./batch_param.txt
    echo check_only n >> batch_param.txt
    echo p12_treat_as_bad n >> ./batch_param.txt
    echo p13_null_noloop n >> ./batch_param.txt
    echo GEOCmldir ${nocorrSplit} >> ./batch_param.txt

    edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

    ./batch_LiCSBAS.sh  # Create uncorrected timeseries for comparison. No nulling

    echo ' '
    echo '#####################'
    echo '#### Correcting Split '${i}
    echo '#####################'
    echo ' '

    LOOPY03_correction_inversion.py -d ${splitdir} -t TS_${nocorrSplit} -c ${splitdir}L03 --coast --dilation 1 --n_para ${para}

    L03dir=${splitdir}L03
    final_dir=${L03dir}

    echo GEOCmldir ${L03dir} > ./batch_param.txt
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

    LiCSBAS_reset_nulls.sh ${L03dir}  # Reset Nullification before correction

    maskdir=${L03dir}_intMask

    LOOPY04_aggressiveResiduals.py -d ${L03dir} -t TS_${L03dir} -o ${maskdir} --apply_mask

    cp -r TS_${L03dir}/130resid/integer_correction ${maskdir}
    final_dir=${maskdir}

    echo GEOCmldir ${maskdir} > ./batch_param.txt
    ./batch_LiCSBAS.sh  # Run Nullified TS to create a "Good" velocity field

    LOOPY05_compare_correction.py -f ./ -i ${nocorrSplit} -c ${final_dir} -o comp_Split${i} -n ${para} --reset

    LiCSBAS_reset_nulls.sh ${final_dir}  # Reset Nullification before Merging

  done

fi

# At this point, all of the split directories have been corrected.
# Now, we shall merge the pre- and post- seismic networks

if [ ${mergeTS} -eq 1 ]; then
  mergebasedir=${GEOCdir}merge
  mergepredir=${mergebasedir}Pre
  mergepostdir=${mergebasedir}Pos

  if [ ${corr_correct} = 'y' ]; then
    suffix=L03_intCorr;
  elif [ ${mask_correct} = 'y' ]; then
    suffix=L03_intMask;
  else
    suffix=L03
  fi

  echo ' '
  echo '#####################'
  echo '#### Merging Split'${suffix}' Timeseries into Pre- and Post- Seismic networks'
  echo '#####################'
  echo ' '

  if [ ! `ls -d *${mergebasedir}* | wc -l` -eq 0 ]; then
    for i in $(seq 5 -1 0); do
      echo "    Removing [TS_]"${mergebasedir}"* in "${i}
      sleep 1
    done
    rm -rf *${mergebasedir}*
  fi

  LiCSBAS_split_TS.py -f ./ -d ${GEOCdir} -s ../split_dates.txt -c ${GEOCdir} -m ${suffix} --merge -k


  echo ' '
  echo '#####################'
  echo '#### Ignoring Pre-Seismic Network - No change'
  echo '#####################'
  echo ' '

  echo ' '
  echo '#####################'
  echo '#### Creating Uncorrected Post-Seismic Timeseries from '${mergepostdir}
  echo '#####################'
  echo ' '

  n_gap_thre=$((`grep Pos splitdirs_full.txt | wc -l` - 1)) # Allow a number of gaps of 1 less than number of merged splits

  echo GEOCmldir ${mergepostdir} > ./batch_param.txt
  echo start_step 11 >> batch_param.txt
  echo end_step 15 >> batch_param.txt
  echo cometdev 1 >> ./batch_param.txt
  echo p12_treat_as_bad y >> ./batch_param.txt
  echo p13_null_noloop y >> ./batch_param.txt
  echo p15_n_gap_thre ${n_gap_thre} >> ./batch_param.txt  # Allows breaks at the merges

  edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

  ./batch_LiCSBAS.sh # Generate Nulled Timeseries of post-seismic dataset

  LiCSBAS_reset_nulls.sh ${mergepostdir}  # So that LOOPY03 will corrects as many UNWs in a pixel as possible

  echo ' '
  echo '#####################'
  echo '#### Correcting Post-Seismic Timeseries from '${mergepostdir}
  echo '#####################'
  echo ' '

  # Run Loopy03, but also inverse solve some random pixels that contain gaps but could be good, to improve interpolations
  # (Selection is based off the results of the time series, so its ok that we have reset the nulls)
  L03dir=${mergepostdir}L03
  LOOPY03_correction_inversion.py -d ${mergepostdir} -t TS_${mergepostdir} -c ${L03dir}_all --coast --randpix 500 --dilation 0 --merge ${n_gap_thre} --n_para ${para}

  # But we only ran LOOPY03 to try and correct the gaps, so recreate make a copy of mergepostdir, and copy in the spanners

  cp -r ${mergepostdir} ${L03dir}

  for ifg in $(cat ${mergepostdir}/uncorrected.txt); do
    if [ -d ${mergepostdir}/${ifg} ]; then
      if [ -d ${L03dir}/${ifg} ]; then
        rm -rf ${L03dir}/${ifg}
        cp -rf ${L03dir}_all/${ifg} ${L03dir}/${ifg}
      fi
    fi
  done

  echo GEOCmldir ${L03dir} > ./batch_param.txt
  echo start_step 11 >> batch_param.txt
  echo end_step 15 >> batch_param.txt
  echo cometdev 1 >> ./batch_param.txt
  echo p12_treat_as_bad y >> ./batch_param.txt
  echo p13_null_noloop y >> ./batch_param.txt
  echo p15_n_gap_thre 0 >> ./batch_param.txt  # Allows breaks at the merges

  edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

  ./batch_LiCSBAS.sh # Generate nulled L03 Timeseries

  LiCSBAS_reset_nulls.sh ${L03dir}

  # Only build in mask correction. No point doing corr if we're not using it
  maskdir=${L03dir}_intMask
  LOOPY04_aggressiveResiduals.py -d ${mergepostdir} -t TS_${L03dir} -o ${maskdir} -l ${L03dir}/uncorrected.txt --apply_masks
  cp -r TS_${mergepostdir}/130resid/integer_correction ${maskdir}

  # At this point, we now have a directory of previously corrected split IFGs, with spanner IFGs that have been corrected accordingly.
  # Need to consider at this point the benefits of nonan and nanning. This ^ step certainly needs to be done in the same way as the previous
  # wrt nan vs no nan
  allcorrdir=${mergepostdir}_allCorrected

  LOOPY05_compare_correction.py -f ./ -i ${noLOOPYdir} -c ${allcorrdir} -o comp_PostSeismic_L03 -n ${para} --reset


  echo ' '
  echo '#####################'
  echo '#### Running Corrected Full Post-Seismic Timeseries from '${maskdir}
  echo '#####################'
  echo ' '

  echo GEOCmldir ${maskdir} > ./batch_param.txt
  echo start_step 11 >> batch_param.txt
  echo end_step 15 >> batch_param.txt
  echo cometdev 1 >> ./batch_param.txt
  echo p12_treat_as_bad y >> ./batch_param.txt
  echo p13_null_noloop y >> ./batch_param.txt

  edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt
  ./batch_LiCSBAS.sh # Generate nulled L03 Timeseries

  LiCSBAS_reset_nulls.sh ${maskdir}

  # We now have a corrected pre-seismic and post-seismic network. Trick here now is to see if we can correct for Kaikoura
  # Gap Checks dont work, Kaikoura is a mess. Plan is to take the SplitPre1 and SplitPost1, merge them into a coseismic
  # timeseries, run L03 inversion on them (only using Post1 for speed) with 1000 pix, and mask correct.
  # Then we join mergePre and mergePost, and insert the corrected co-seismics
  # Note this works because we only corrected the spanning IFGs in the last step

  cp splitdirs.txt splitdirs_full.txt

  grep SplitID splitdirs_full.txt > splitdirs.txt
  grep Pre splitdirs_full.txt | tail -1 >> splitdirs.txt # In case you ever have multiple pre-seismics
  grep Pos1 splitdirs_full.txt >> splitdirs.txt # Currently we only work with 1 earthquake, so it will only ever be Pos1

  echo ' '
  echo '#####################'
  echo '#### Merging Split into Coseismic Timeseries'
  echo '#### (i.e. just Pre1 and Pos1)'
  echo '#####################'
  echo ' '

  # Rename pre-existing pos tiles
  for pos in $(ls -d *${GEOCdir}mergePos*); do
    echo "mv ${pos} ${pos}_fullPost"
    mv ${pos} ${pos}_fullPost
  done


  # Rename SplitPos1 as mergePos
  echo "ln -s ${GEOCdir}SplitPos1${suffix} ${GEOCdir}mergePos"
  echo "ln -s TS_${GEOCdir}SplitPos1${suffix} TS_${GEOCdir}mergePos"
  ln -s ${GEOCdir}SplitPos1${suffix} ${GEOCdir}mergePos
  ln -s TS_${GEOCdir}SplitPos1${suffix} TS_${GEOCdir}mergePos

  mergecodir=${mergebasedir}Cos
  mergecodir1=${mergecodir}Trim

  LiCSBAS_split_TS.py -f ./ -d ${GEOCdir} -s ../split_dates.txt -c ${GEOCdir} -m ${suffix} --merge -e

  mv ${mergecodir} ${mergecodir1}

  echo ' '
  echo '#####################'
  echo '#### Creating Uncorrected Co-Seismic Timeseries from '${mergecodir1}
  echo '#####################'
  echo ' '

  echo GEOCmldir ${mergecodir1} > ./batch_param.txt
  echo start_step 11 >> batch_param.txt
  echo end_step 15 >> batch_param.txt
  echo cometdev 1 >> ./batch_param.txt
  echo p12_treat_as_bad y >> ./batch_param.txt
  echo p13_null_noloop y >> ./batch_param.txt

  edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

  LiCSBAS_reset_nulls.sh ${mergecodir1}  # Pre-cautionary

  ./batch_LiCSBAS.sh # Generate Nulled Timeseries of co-seismic dataset

  LiCSBAS_reset_nulls.sh ${mergecodir1}  # So that LOOPY03 will corrects as many UNWs in a pixel as possible

  echo ' '
  echo '#####################'
  echo '#### Correcting Co-Seismic Timeseries from '${mergecodir1}
  echo '#####################'
  echo ' '

  # Run Loopy03, but also inverse solve some random pixels that contain gaps but could be good, to improve interpolations
  # (Selection is based off the results of the time series, so its ok that we have reset the nulls)
  # LOOPY03_correction_inversion.py -d ${mergecodir1} -t TS_${mergecodir1} -c ${mergecodir1}L03 --coast --randpix 1000 --dilation 0 --n_para ${para}
  # Invert everything, hope that there is enough either side of the EQ to constrain the data, but not so mush it takes forever. May need to pick a
  # % to correct for non-clipped data. Or nan first, and then invert the bad pixels only
  LOOPY03_correction_inversion.py -d ${mergecodir1} -t TS_${mergecodir1} -c ${mergecodir1}L03 --n_para ${para}
  # ^^ THIS DOES NOT MAKE TO COSEISMIC BETTER

  mergecodir1=${mergecodir1}L03

  echo GEOCmldir ${mergecodir1} > ./batch_param.txt
  echo start_step 11 >> batch_param.txt
  echo end_step 15 >> batch_param.txt
  echo cometdev 1 >> ./batch_param.txt
  echo p12_treat_as_bad y >> ./batch_param.txt
  echo p13_null_noloop y >> ./batch_param.txt

  edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

  ./batch_LiCSBAS.sh # Generate nulled L03 Timeseries

  LiCSBAS_reset_nulls.sh ${mergecodir1}

  # Only code in mask correction. No point doing corr if we're not using it
  maskcodir1=${mergecodir1}_intMask
  LOOPY04_aggressiveResiduals.py -d ${mergecodir1} -t TS_${mergecodir1} -o ${maskcodir1} -l ${mergecodir1}/uncorrected.txt --apply_mask
  cp -r TS_${mergecodir1}/130resid/integer_correction ${maskcodir1}
  echo "### Corrected Directory: ${maskcodir1}"

  LOOPY05_compare_correction.py -f ./ -i ${noLOOPYdir} -c ${maskcodir1} -o comp_coseismic -n ${para} --reset

  #Remove none - coseismic IFGs given the correction on those are not carried through
  if [ -d compcoseismic/IFGs ]; then
    for png in $(ls comp_coseismic/IFGs ); do
      ifg=${png::17}
      im1=${ifg::8}
      im2=${ifg: -8}
      if [ ${im1} -lt $kaikoura ] && [ ${im2} -gt ${kaikoura} ]; then
        echo "Preserve ${ifg}"
      else
        rm -f comp_coseismic/IFGs/${png}
      fi
    done
  fi

  echo ' '
  echo '#####################'
  echo '#### Restore Full Post-Seismic Time Series'
  echo '#####################'
  echo ' '

  # Remove symlinks
  if [ -L ${GEOCdir}mergePos ]; then
    rm -f ${GEOCdir}mergePos
  fi
  if [ -L TS_${GEOCdir}mergePos ]; then
    rm -f TS_${GEOCdir}mergePos
  fi

  # # Rename pre-existing pos tiles
  # for pos in $(ls -d *${GEOCdir}mergePos*_fullPost); do
  #   pos_old=${pos::-9}
  #   echo "Restoring ${pos_old} from ${pos}"
  #   mv ${pos} ${pos_old}
  # done

  # Rename pre-existing pos tiles
  for pos in $(ls -d *${GEOCdir}mergePos*_fullPost); do
    pos_old=${pos::-9}
    if [ ${pos_old} = ${maskdir} ]; then
      echo "Restoring ${pos_old} from ${mergepostdir}"
      mv ${pos} ${mergepostdir}
    fi
  done

  echo ' '
  echo '#####################'
  echo '#### Merging mergePre and mergePos into entire timeseries'
  echo '#####################'
  echo ' '

  LiCSBAS_split_TS.py -f ./ -d ${GEOCdir} -s ../split_dates.txt -c ${GEOCdir} -m ${suffix} --merge -e

  echo ' '
  echo '#####################'
  echo "#### Adding Corrected Co-seismic from ${maskcodir1} to ${mergecodir}"
  echo '#####################'
  echo ' '

  for ifg in $(cat ${mergecodir}/uncorrected.txt); do
    if [ -d ${maskcodir1}/${ifg} ]; then
      rm -rf ${mergecodir}/${ifg}
      cp -r ${maskcodir1}/${ifg} ${mergecodir}/${ifg}
    fi
  done

  LiCSBAS_reset_nulls.sh ${mergecodir}

  echo ' '
  echo '#####################'
  echo '#### Creating Nullified Complete Timeseries from '${mergecodir}
  echo '#####################'
  echo ' '

  echo GEOCmldir ${mergecodir} > ./batch_param.txt
  echo start_step 11 >> batch_param.txt
  echo end_step 15 >> batch_param.txt
  echo cometdev 1 >> ./batch_param.txt
  echo p12_treat_as_bad y >> ./batch_param.txt
  echo p13_null_noloop y >> ./batch_param.txt

  edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

  ./batch_LiCSBAS.sh # Generate Nulled Timeseries of full dataset

  echo ' '
  echo '#####################'
  echo "#### Final Correction applying untouched data ${noLOOYdir} to nullified timeseries of TS_${mergecodir}"
  echo '#####################'
  echo ' '

  LiCSBAS_reset_nulls.sh ${noLOOPYdir}
  final_dir=${noLOOPYdir}_fullCorrection
  LOOPY04_aggressiveResiduals.py -d ${noLOOPYdir} -t TS_${mergecodir} -o ${final_dir} --apply_mask --nonan # nonan means that we will have all pixels


  if [ ! ${mergeTS} -eq 1 ]; then
    final_dir=${GEOCdir}
  fi

  echo ' '
  echo '#####################'
  echo '#### Running LiCSBAS on corrected dataset ('${noLOOPYdir}' corrected to '${final_dir}') using default masking'
  echo '#####################'
  echo ' '

  if [ -f batch_LiCSBAS.sh ]; then
    rm -f batch_LiCSBAS.sh
  fi

  copy_batch_LiCSBAS.sh

  echo GEOCmldir ${final_dir} > ./batch_param.txt
  echo cometdev 1 >> ./batch_param.txt
  echo start_step 11 >> batch_param.txt
  echo end_step 15 >> batch_param.txt
  echo check_only n >> batch_param.txt
  echo p12_multi_prime y >> ./batch_param.txt
  echo p12_treat_as_bad n >> ./batch_param.txt
  echo p13_null_noloop n >> ./batch_param.txt
  echo p11_TSdir TS_${final_dir} >> ./batch_param.txt
  echo p12_TSdir TS_${final_dir} >> ./batch_param.txt
  echo p13_TSdir TS_${final_dir} >> ./batch_param.txt
  echo p14_TSdir TS_${final_dir} >> ./batch_param.txt
  echo p15_TSdir TS_${final_dir} >> ./batch_param.txt

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

if [ ${uncorrTS} -eq 1 ]; then

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
