#!/bin/bash

case_names=('20170305_Zeus' '20170724_Sommersturm' '20180103_Burglind' \
            '20180116_Friederike' '20180301_Foehntaeler' \
            '20180429_Foehnsturm' '20180503_Bisensturm' \
            '20180530_Gewittertage' 'June_18' 'December_17')


for case_name in ${case_names[@]}
do
    echo $case_name
    echo 'make tarball'
    tar -rf tar_files/${case_name}.tar $case_name
    echo 'compress'
    gzip tar_files/${case_name}.tar
done



