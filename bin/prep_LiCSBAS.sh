#!/bin/bash

framelist='/scratch/eejdm/FINALS/DIV_frames.txt'
workdir='/scratch/eejdm/FINALS/LiCSBAS_standard'

for frame in $(cat ${framelist}); do

  mkdir -p ${workdir}/${frame}
  cd ${workdir}/${frame}
  if [ -f batch_LiCSBAS.sh ]; then
    rm -f batch_LiCSBAS.sh
  fi
  copy_batch_LiCSBAS.sh

  touch batch_param.txt

  echo p01_frame ${frame} >> batch_param.txt
  echo end_step 15 >> batch_param.txt
  echo cometdev 0 >> batch_param.txt
  echo nlook 10 >> batch_param.txt
  echo GEOCmldir GEOCml10 >> batch_param.txt
  echo n_para 45 >> batch_param.txt
  echo check_only n >> batch_param.txt
  echo do03op_GACOS y >> batch_param.txt
  echo p01_get_gacos y >> batch_param.txt

  edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

  ./batch_LiCSBAS.sh

  cd ${workdir}
done

for frame in $(cat ${framelist}); do

  cd ${workdir}/${frame}
  LOOPY01_find_errors.py -d GEOCml10 -e ../gmtErrors.txt --only_listed --n_para 40

  echo start_step 03 > batch_param.txt
  echo GEOCmldir GEOCml10LoopMask >> batch_param.txt

  edit_batch_LiCSBAS.sh ./batch_LiCSBAS.sh ./batch_param.txt

  ./batch_LiCSBAS.sh

  cp -r TS_GEOCml10LoopMaskGACOS TS_GEOCml10LoopMaskGACOS_noNull

  cd ${workdir}
done

for frame in $(cat ${framelist}); do

  cd ${workdir}/${frame}

  cp ${workdir}/split_dates.txt .

  run_Loopy.sh

done
