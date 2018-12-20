#!/bin/bash

model_inds=($(seq 0 1 5))
#model_inds=(4)
#printf "%s\n" "${model_inds[@]}"
#exit 1

for model_ind in ${model_inds[@]}
do
    echo $model_ind
    srun -n 1 -c 3 -J train${model_ind} python -u -W ignore org_training.py ${model_ind} \
                    > bOut/sbatch${model_ind}.out \
                    2> bOut/sbatch${model_ind}.err  &
done



