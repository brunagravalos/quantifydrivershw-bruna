#!/bin/bash

#SBATCH --qos=gp_debug
#SBATCH -A bsc32
#SBATCH -n 1
#SBATCH -t 2:00:00
#SBATCH --exclusive
#SBATCH --chdir=.
#SBATCH --output=process_%j.out
#SBATCH --error=process_%j.err


SCRIPT_PATH="/gpfs/scratch/bsc32/bsc214253/quantifydrivershw/src/quantifydrivers/utils/utils.py"

#SCRIPT_PATH="/gpfs/scratch/bsc32/bsc214253/quantifydrivershw/src/quantifydrivers/train_and_shap/training_evaluation_SHAP_pipeline.py"
#CONTAINER_PATH="/gpfs/scratch/bsc32/bsc167965/environments/pangeo_pytorch-202502"
#BIND_PATH="/gpfs/scratch/bsc32/bsc167965/"

#export PROJ_LIB="/opt/conda/share/proj"

# Choose a seed. Using a fixed number '42' here, but you might want to use $SLURM_JOB_ID

#module load singularity

#singularity exec --nv -B $BIND_PATH \
#  $CONTAINER_PATH \
#  python $SCRIPT_PATH 1234

#singularity exec /gpfs/scratch/bsc32/bsc167965/environments/pangeo_pytorch-202502 \
#  python -m pip install colorlog

#singularity exec --nv -B $BIND_PATH \
#  $CONTAINER_PATH \
#  python $SCRIPT_PATH 1234


uv run python "$SCRIPT_PATH"
# Run the script inside the uv-managed environment
#uv run python "$SCRIPT_PATH" seed=$CURRENT_SEED site=cordoba
# 66316748,2930678936,2546691362,231159514,3904498325,946438445,1095601156,791870896,1432871125,755510091
# 1493800520,3487919346,1938714511,3965736568,1930440936,1187877992,3387705611,3520819031,3701866991,3822060012