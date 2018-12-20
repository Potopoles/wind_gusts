#!/bin/bash

module load nco
module load cdo

rm *.nc*

ts=('0001' '0002' '0003' '0004' '0005' '0006' '0007' '0008' \ 
    '0009' '0010' '0011' '0012' '0013' '0014' '0015' '0016' \
    '0017' '0018' '0019' '0020' '0021' '0022' '0023' '0100' )
#ts=('0001' '0002' '0003' '0004' '0005' '0006' )

for t in ${ts[@]};
do
    echo $t
    fxconvert netcdf lfff${t}0000 -o lfff${t}0000.nc
done

var='VMAX_10M'
ncrcat -v $var lfff*.nc ${var}.nc
var='TKE'
ncrcat -v $var lfff*.nc ${var}.nc
var='U'
ncrcat -v $var lfff*.nc ${var}.nc
var='U_10M'
ncrcat -v $var lfff*.nc ${var}.nc
var='V_10M'
ncrcat -v $var lfff*.nc ${var}.nc
var='Z0'
ncrcat -v $var lfff*.nc ${var}.nc
var='T'
ncrcat -v $var lfff*.nc ${var}.nc
ar='TCM'
crcat -v $var lfff*.nc ${var}.nc



# calc ff_10m
cdo sqr U_10M.nc tmp_u.nc
cdo sqr V_10M.nc tmp_v.nc
cdo add tmp_u.nc tmp_v.nc tmp.nc
cdo sqrt tmp.nc tmp2.nc
cdo chname,U_10M,FF_10M tmp2.nc FF_10M.nc
rm tmp_u.nc
rm tmp_v.nc
rm tmp.nc
rm tmp2.nc

