#!/bin/bash

#SBATCH --qos=gp_debug
#SBATCH -A bsc32
#SBATCH -n 1
#SBATCH -t 2:00:00
#SBATCH --exclusive
#SBATCH --chdir=.
#SBATCH --output=process_%j.out
#SBATCH --error=process_%j.err


SCRIPT_PATH="/gpfs/scratch/bsc32/bsc214253/quantifydrivershw/src/quantifydrivers/hypm_tunning/aux.py"

uv run python "$SCRIPT_PATH"
