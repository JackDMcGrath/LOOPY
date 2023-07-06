#!/bin/bash


ifgdir=$1

echo "Resetting Nullified Data in "${ifgdir}
for ifg in $(ls -d ${ifgdir}/20*); do
  pair=${ifg: -17}
  origfile=${ifg}/${pair}_orig.unw
  unwfile=${ifg}/${pair}.unw
  if [ -f ${origfile} ]; then
    mv -f $origfile $unwfile
  fi
done
