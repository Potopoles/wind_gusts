#!/bin/bash -l
#
#SBATCH --job-name="train_05"
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --account='s83'
#SBATCH --partition=dev
#SBATCH --output=bOut/sbatch0.out
#SBATCH --error=bOut/sbatch0.err


echo -e "$SLURM_JOB_NAME started on $(date):\n $command $1 $2\n"

srun -n $SLURM_NTASKS python -u -W ignore "$SLURM_JOB_NAME"*

echo -e "$SLURM_JOB_NAME finished on $(date)\n"


