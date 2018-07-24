#!/bin/bash -l
#
#SBATCH --job-name="phys_03"
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --account='s83'
#SBATCH --partition=normal
#SBATCH --output=bOut/sbatch0.out
#SBATCH --error=bOut/sbatch0.err


echo -e "$SLURM_JOB_NAME started on $(date):\n $command $1 $2\n"

srun -c $SLURM_CPUS_PER_TASK -n $SLURM_NTASKS python -u -W ignore "$SLURM_JOB_NAME"* $SLURM_CPUS_PER_TASK


echo -e "$SLURM_JOB_NAME finished on $(date)\n"


