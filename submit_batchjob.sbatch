#!/bin/bash -l
#
#SBATCH --job-name="train_06"
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --account='s83'
#SBATCH --partition=postproc
#SBATCH --output=bOut/sbatch6.out
#SBATCH --error=bOut/sbatch6.err


echo -e "$SLURM_JOB_NAME started on $(date):\n $command $1 $2\n"

srun -n $SLURM_NTASKS python -u -W ignore "$SLURM_JOB_NAME"*

echo -e "$SLURM_JOB_NAME finished on $(date)\n"


