#!/bin/bash -l
#
#SBATCH --job-name="add_sso"
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --account='s83'
#SBATCH --partition=postproc
#SBATCH --output=sbatch.out
#SBATCH --error=sbatch.err


echo -e "$SLURM_JOB_NAME started on $(date):\n $command $1 $2\n"

srun -n $SLURM_NTASKS add_sso_stdh.sh

echo -e "$SLURM_JOB_NAME finished on $(date)\n"


