#!/bin/bash
#SBATCH --qos=gp_debug
#SBATCH --job-name=zipping
#SBATCH --output=process_%j.out
#SBATCH --error=process_%j.err
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --qos=gp_debug
#SBATCH -A bsc32


cd /gpfs/scratch/bsc32/bsc214253/

zip -r0 climate_data_new.zarr.zip climate_data_new.zarr