#!/bin/bash -l
#
#SBATCH --job-name="110_17"
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --account='s83'
#SBATCH --partition=postproc
#SBATCH --output=sbatch0.out
#SBATCH --error=sbatch0.err


echo -e "$SLURM_JOB_NAME started on $(date):\n $command $1 $2\n"

srun -n $SLURM_NTASKS extract_gust.sh

echo -e "$SLURM_JOB_NAME finished on $(date)\n"


