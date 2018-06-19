#!/bin/bash

wd_path='../wd/'

#caseName='20180301_Burglind'
#modelRuns=("18010212" "18010300" "18010312")
#caseName='20180503_Bisensturm'
#modelRuns=("18050212" "18050300" "18050312" "18050400" "18050412")
#caseName='20180429_Foehnsturm'
#modelRuns=("18042812" "18042900" "18042912" "18043000" "18043012")
caseName='All'
modelRuns=("18010212" "18010300" "18010312" \
           "18050212" "18050300" "18050312" "18050400" "18050412" \
           "18042812" "18042900" "18042912" "18043000" "18043012")

for run in "${modelRuns[@]}"
do
	:
	#run="18010212"
    echo $run

    if [ ! -d "$caseName" ]; then
        mkdir "$caseName"
    fi

    tar -xf ${wd_path}${run}_101/debug/lm_tdir.tar

    extractFolder=${wd_path}${run}_101/debug/${run}_101
    if [ -d "$extractFolder" ]; then
        rm -r $extractFolder
    fi

    mv ${run}_101 ${wd_path}${run}_101/debug/

    outFolder=${caseName}/20${run}
    echo $outFolder
    if [ ! -d "$outFolder" ]; then
        mkdir "$outFolder"
    fi

    subfolder=$(find ${wd_path}${run}_101/debug/${run}_101/lm_lm_c_wd_* -maxdepth 0)
    cp $subfolder/fort* $outFolder/

    #exit 1
done


