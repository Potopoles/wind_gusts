#!/bin/bash

# usage
#srun -p postproc ./run.sh train_01_readjust.py

source ~/load_miniconda.sh
python -u ${1} ${2}
