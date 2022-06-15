#!/bin/bash -eu

# LOOPY steps:
#  01: LOOPY01_find_errors.py
#  02: LOOPY02_corrections.py

#################
### Settings ####
#################
start_step="02"	# 01-02
end_step="02"	# 01-02

nlook="10"	# multilook factor, used in step02
ifgdir="GEOCml${nlook}GACOS"	# If start from 11 or later after doing 03-05, use e.g., GEOCml${nlook}GACOSmaskclip
n_para="" # Number of paralell processing in step 02-05,12,13,16. default: number of usable CPU
check_only="y" # y/n. If y, not run scripts and just show commands to be done

logdir="log"
log="$logdir/$(date +%Y%m%d%H%M)$(basename $0 .sh)_${start_step}_${end_step}.log"

### Frequently used options. If blank, use default. ###
p01_reset=""	# default: False
p01_n_para=""	# default: 4
p02_reset=""	# default: False
p02_n_para=""	# default: 4

### Less frequently used options. If blank, use default. ###
p01_ifgdir=""	# e.g. GEOCml10
p01_tsadir=""	# e.g. TS_GEOCml10
p02_ifgdir=""	# e.g. GEOCml10
p02_tsadir=""	# e.g. TS_GEOCml10

#############################
### Run (No need to edit) ###
#############################
echo ""
echo "Start step: $start_step"
echo "End step:   $end_step"
echo "Log file:   $log"
echo ""
mkdir -p $logdir

### Determine name of TSdir
TSdir="TS_$ifgdir"

if [ $start_step -le 01 -a $end_step -ge 01 ];then
  p01_op=""
  if [ ! -z $p01_ifgdir ];then p01_op="$p01_op -d $p01_ifgdir"; 
    else p01_op="$p01_op -d $ifgdir"; fi
  if [ ! -z $p01_tsadir ];then p01_op="$p01_op -t $p01_tsadir";
    else p02_op="$p02_op -t $TSdir"; fi
  if [ ! -z $p01_reset ];then p01_op="$p01_op --reset"; fi
  if [ ! -z $p01_n_para ];then p01_op="$p01_op --n_para $p01_n_para"; fi

  if [ $check_only == "y" ];then
    echo "LOOPY01_find_errors.py $p01_op"
  else
    LOOPY01_find_errors.py $p01_op 2>&1 | tee -a $log
    if [ ${PIPESTATUS[0]} -ne 0 ];then exit 1; fi
  fi
fi

if [ $start_step -le 02 -a $end_step -ge 02 ];then
  p02_op=""
  if [ ! -z $p02_ifgdir ];then p02_op="$p02_op -d $p02_ifgdir"; 
    else p02_op="$p02_op -d $ifgdir"; fi
  if [ ! -z $p02_tsadir ];then p02_op="$p02_op -t $p02_tsadir";
    else p02_op="$p02_op -t $TSdir"; fi 
  if [ ! -z $p02_reset ];then p02_op="$p02_op --reset"; fi
  if [ ! -z $p02_n_para ];then p02_op="$p02_op --n_para $p02_n_para"; fi

  if [ $check_only == "y" ];then
    echo "LOOPY02_corrections.py $p02_op"
  else
    LOOPY02_corrections.py $p02_op 2>&1 | tee -a $log
    if [ ${PIPESTATUS[0]} -ne 0 ];then exit 1; fi
  fi
fi

if [ $check_only == "y" ];then
  echo ""
  echo "Above commands will run when you change check_only to \"n\""
  echo ""
fi
