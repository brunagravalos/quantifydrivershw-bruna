#!/bin/bash

#SBATCH --qos=gp_debug
#SBATCH -A bsc32
#SBATCH -n 1
#SBATCH -t 2:00:00
#SBATCH --exclusive
#SBATCH --chdir=.
#SBATCH --cpus-per-task=1
#SBATCH --output=process_%j.out
#SBATCH --error=process_%j.err

SCRIPT_PATH="/gpfs/scratch/bsc32/bsc214253/quantifydrivershw/src/quantifydrivers/train_and_shap/train_and_SHAP_modularized3.py"
CONTAINER_PATH="/gpfs/scratch/bsc32/bsc167965/environments/pangeo_pytorch-202502"
BIND_PATH="/gpfs/scratch/bsc32/bsc167965/"

export PROJ_LIB="/opt/conda/share/proj"

# Choose a seed. Using a fixed number '42' here, but you might want to use $SLURM_JOB_ID

module load singularity

#singularity exec --nv -B $BIND_PATH \
#  $CONTAINER_PATH \
#  python $SCRIPT_PATH 1234

#singularity exec /gpfs/scratch/bsc32/bsc167965/environments/pangeo_pytorch-202502 \
#  python -m pip install colorlog

singularity exec --nv -B $BIND_PATH \
  $CONTAINER_PATH \
  python $SCRIPT_PATH 1234
