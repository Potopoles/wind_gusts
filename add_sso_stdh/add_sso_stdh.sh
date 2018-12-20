#!/bin/bash

fx_template_date=2018060312
out_dir=/scratch/heimc/cache/105

for date in `cat list`; do
    echo '################################### '$date' ######################################'
    echo '#####################################################################################'
    yy=${date:2:2}
    sed "s/$fx_template_date/$date/" fxfilter.nl > tmp_fxfilter.nl
    fieldextra tmp_fxfilter.nl
    # COSMO-1
    cat /store/s83/owm/COSMO-1/ANA${yy}/laf${date} SSO_STDH > ${out_dir}/laf${date}
    # COMSO-E mean
    #cat /store/s83/owm/KENDA/ANA${yy}/mean/laf${date} laf${date}_SSO_STDH > ${out_dir}/laf${date}
    rm SSO_STDH
done
rm tmp_fxfilter.nl
