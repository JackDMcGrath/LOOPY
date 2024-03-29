#!/bin/bash

# Script to reset the nullified data to it's original state
# Recommend using LiCSBAS_reset_nulls.py --reset_all instead

ifgdir=$1

echo "Resetting Nullified Data in "${ifgdir}
for ifg in $(ls -d ${ifgdir}/20*); do
  pair=${ifg: -17}
  origfile=${ifg}/${pair}_orig.unw
  unwfile=${ifg}/${pair}.unw
  if [ -f ${origfile} ]; then
    mv -f $origfile $unwfile
    rm -rf ${ifg}/*orig*.unw
  fi
done
