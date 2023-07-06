#!/bin/bash


unwdir='GEOCml10LoopMaskGACOS'
splitfile='split_dates.txt'

n_lines=`wc -l $splitfile | awk '{print $1}'`
n_split=`echo $((${n_lines}-1))`

resetNullsOnly=0
removeOldSplits=1

# For each split
echo "Resetting Nullified Data"
for ifg in $(ls -d ${unwdir}/20*); do
  pair=${ifg: -17}
  origfile=${ifg}/${pair}_orig.unw
  unwfile=${ifg}/${pair}.unw
  if [ -f ${origfile} ]; then
    mv $origfile $unwfile
  fi
done

if [ ${resetNullsOnly} -eq 0 ]; then
  if [ ${removeOldSplits} -eq 1 ]; then
    echo "Removing [TS_]"${unwdir}"Split* in 5"
    sleep 1
    echo "Removing [TS_]"${unwdir}"Split* in 4"
    sleep 1
    echo "Removing [TS_]"${unwdir}"Split* in 3"
    sleep 1
    echo "Removing [TS_]"${unwdir}"Split* in 2"
    sleep 1
    echo "Removing [TS_]"${unwdir}"Split* in 1"
    sleep 1
    echo "Removing [TS_]"${unwdir}"Split* now"
    sleep 1
    rm -rf *${unw}Split*
  fi

  echo ' '
  echo '#################'
  echo 'SPLITTING TIMESERIES FOR CORRECTIONS'
  echo '#################'

  LiCSBAS_split_TS.py -d $unwdir -s $splitfile

  for i in $(seq 1 $n_split); do

    echo ' '
    echo '#################'
    echo 'CORRECTING SPLIT '${i}
    echo '#################'


    if [ -f ./batch_param.txt ]; then
      rm -f ./batch_param.txt
    fi

    touch ./batch_param.txt

    splitdir=${unwdir}Split${i}

    echo start_step 11 >> ./batch_param.txt
    echo end_step 15 >> ./batch_param.txt
    echo cometdev 1 >> ./batch_param.txt
    echo GEOCmldir ${splitdir} >> ./batch_param.txt
    echo n_para 40  >> ./batch_param.txt
    echo check_only n >> ./batch_param.txt
    echo p12_loop_thre 100 >> ./batch_param.txt
    echo p12_multi_prime y >> ./batch_param.txt
    echo p15_coh_thre 0.05 >> ./batch_param.txt
    echo p15_n_unw_r_thre 0.0 >> ./batch_param.txt
    echo p15_vstd_thre 100 >> ./batch_param.txt
    echo p15_maxTlen_thre 1 >> ./batch_param.txt
    echo p15_n_gap_thre 0 >> ./batch_param.txt
    echo p15_stc_thre 5 >> ./batch_param.txt
    echo p15_n_ifg_noloop_thre 5 >> ./batch_param.txt
    echo p15_n_loop_err_thre 1000 >> ./batch_param.txt
    echo p15_resid_rms_thre 10 >> ./batch_param.txt

    edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

    ./batch_LiCSBAS.sh

    echo "Resetting Nullified Data"
    for ifg in $(ls -d ${unwdir}/20*); do
      pair=${ifg: -17}
      origfile=${ifg}/${pair}_orig.unw
      unwfile=${ifg}/${pair}.unw
      if [ -f ${origfile} ]; then
        mv $origfile $unwfile
      fi
    done


    maskdir=${splitdir}_masked
    LOOPY04_aggressiveResiduals.py -d ${splitdir} -t TS_${splitdir} -o ${maskdir} --apply_mask

    cp -r TS_${splitdir}/130resid/integer_correction ${maskdir}

    echo GEOCmldir ${maskdir} > ./batch_param.txt
    edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

    ./batch_LiCSBAS.sh

    echo "Resetting Nullified Data"
    for ifg in $(ls -d ${unwdir}/20*); do
      pair=${ifg: -17}
      origfile=${ifg}/${pair}_orig.unw
      unwfile=${ifg}/${pair}.unw
      if [ -f ${origfile} ]; then
        mv $origfile $unwfile
      fi
    done

    corrdir=${splitdir}_corrected

    LOOPY04_aggressiveResiduals.py -d ${splitdir} -t TS_${maskdir} -o ${corrdir}
    cp -r TS_${maskdir}/130resid/integer_correction ${corrdir}

    echo GEOCmldir ${corrdir} > ./batch_param.txt
    echo cometdev 0 >> ./batch_param.txt

    edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

    ./batch_LiCSBAS.sh

  done


  echo ' '
  echo '#################'
  echo 'MERGING SPLIT TIMESERIES'
  echo '#################'


  mergedir=${unwdir}merge
  LiCSBAS_split_TS.py -d $unwdir -s $splitfile --merge_corrected

  echo ' '
  echo '#################'
  echo 'CORRECTING FULL TIMESERIES'
  echo '#################'

  echo GEOCmldir ${mergedir} > ./batch_param.txt
  echo cometdev 1 >> ./batch_param.txt
  edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

  ./batch_LiCSBAS.sh

  echo "Resetting Nullified Data"
  for ifg in $(ls -d ${mergedir}/20*); do
    pair=${ifg: -17}
    origfile=${ifg}/${pair}_orig.unw
    unwfile=${ifg}/${pair}.unw
    if [ -f ${origfile} ]; then
      mv $origfile $unwfile
    fi
  done

  maskdir=${mergedir}_masked
  LOOPY04_aggressiveResiduals.py -d ${unwdir} -t TS_${mergedir} -o ${maskdir} --apply_mask

  cp -r TS_${mergedir}/130resid/integer_correction ${maskdir}

  echo GEOCmldir ${maskdir} > ./batch_param.txt
  edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

  ./batch_LiCSBAS.sh

  echo "Resetting Nullified Data"
  for ifg in $(ls -d ${mergedir}/20*); do
    pair=${ifg: -17}
    origfile=${ifg}/${pair}_orig.unw
    unwfile=${ifg}/${pair}.unw
    if [ -f ${origfile} ]; then
      mv $origfile $unwfile
    fi
  done

  corrdir=${mergedir}_corrected

  LOOPY04_aggressiveResiduals.py -d ${unwdir} -t TS_${maskdir} -o ${corrdir}
  cp -r TS_${maskdir}/130resid/integer_correction ${corrdir}

  echo ' '
  echo '#################'
  echo 'GENERATING LOS TIMESERIES FROM CORRECTED TIMESERIES'
  echo '#################'

  echo GEOCmldir ${corrdir} > ./batch_param.txt
  echo cometdev 0 >> ./batch_param.txt

  edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

  ./batch_LiCSBAS.sh

fi
