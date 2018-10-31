#!/bin/bash -l
#
#SBATCH --job-name="tar"
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --account='s83'
#SBATCH --partition=pp-long
#SBATCH --output=tar.out
#SBATCH --error=tar.err


echo -e "$SLURM_JOB_NAME started on $(date):\n $command $1 $2\n"

srun -n $SLURM_NTASKS tar -rf tar_files/${1}.tar ${1} ; gzip tar_files/${1}.tar

echo -e "$SLURM_JOB_NAME finished on $(date)\n"


