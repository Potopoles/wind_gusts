#!/bin/bash

wd_path='../wd/'

# DEBUG
caseName='DEBUG_Sommer'
modelRuns=("18061000")

# REAL CASES
caseName='20170305_Zeus'
modelRuns=("17030412" "17030500" "17030512" "17030600" "17030612" "17030700" "17030712")
caseName='20170719_Konvektion'
modelRuns=("17071912")
caseName='20170724_Sommersturm'
modelRuns=("17072312" "17072400" "17072412" "17072500" "17072512")
caseName='20180103_Burglind'
modelRuns=("18010212" "18010300" "18010312")
caseName='20180116_Friederike'
modelRuns=("18011512" "18011600" "18011612" "18011700" "18011712" "18011800" "18011812")
#caseName='20180301_Foehntaeler'
#modelRuns=("18022812" "18030100" "18030112")
caseName='20180429_Foehnsturm'
modelRuns=("18042812" "18042900" "18042912" "18043000" "18043012")
caseName='20180503_Bisensturm'
modelRuns=("18050212" "18050300" "18050312" "18050400" "18050412")
caseName='20180530_Gewittertage'
modelRuns=("18052912" "18053000" "18053012" "18053100" "18053112")
#caseName='All'
#modelRuns=("17030412" "17030500" "17030512" "17030600" "17030612" "17030700" "17030712" \
#           "17071912" \
#           "17072312" "17072400" "17072412" "17072500" "17072512" \
#           "18010212" "18010300" "18010312" \
#           "18011512" "18011600" "18011612" "18011700" "18011712" "18011800" "18011812"\
#           "18022812" "18030100" "18030112"\
#           "18042812" "18042900" "18042912" "18043000" "18043012"\
#           "18050212" "18050300" "18050312" "18050400" "18050412"\
#           "18052912" "18053000" "18053012" "18053100" "18053112")








#caseName='20180119_JanuaryDays'
#modelRuns=("18011900" "18011912" "18012000" "18012012" "18012100" "18012112" "18012200"
#            "18012212" "18012300" "18012312" "18012400" "18012412" "18012512"
#            "18012600" "18012612" "18012700" "18012712" "18012800")


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
    echo $subfolder
    cp $subfolder/fort* $outFolder/

    #exit 1
done


