#!/bin/bash

#base_dir='../pre_summer/'
#base_dir='/store/s83/heimc/EXP_TST/107/FCST16'
#base_dir='../107/FCST17'
#base_dir='../108/FCST17'
base_dir='../110'
#base_dir='../107'
exp_tag='105'

for date in `cat ll`; do
    echo '################################### '$date' ######################################'
    echo '#####################################################################################'
    echo $base_dir/${date}_${exp_tag}/lm_coarse/
    cp extract_gust.nl $base_dir/${date}_${exp_tag}/lm_coarse/
    cd $base_dir/${date}_${exp_tag}/lm_coarse/
    fieldextra extract_gust.nl
    mkdir ../grib
    mv c1ffsurf* ../grib
    cp extract_gust.nl ../grib
    cd -
    echo `pwd`
done
