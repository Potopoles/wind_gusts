#!/bin/bash -l
#
#SBATCH --job-name="03"
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --account='s83'
#SBATCH --partition=postproc
#SBATCH --output=bOut/sbatch1.out
#SBATCH --error=bOut/sbatch1.err


echo -e "$SLURM_JOB_NAME started on $(date):\n $command $1 $2\n"

srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK \
            python -u -W ignore "$SLURM_JOB_NAME"* $SLURM_CPUS_PER_TASK

echo -e "$SLURM_JOB_NAME finished on $(date)\n"


