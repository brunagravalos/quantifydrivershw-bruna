#!/bin/bash

#SBATCH --qos=acc_bsces
#SBATCH -A bsc32
#SBATCH -n 1
#SBATCH -t 2:00:00
#SBATCH --chdir=.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --array=1-20
#SBATCH --output=process_%A_%a.out
#SBATCH --error=process_%A_%a.err

#list of seeds
SEED_LIST=(66316748 2930678936 2546691362 231159514 3904498325 946438445 1095601156 791870896 1432871125 755510091 1493800520 3487919346 1938714511 3965736568 1930440936 1187877992 3387705611 3520819031 3701866991 3822060012)

CURRENT_SEED=${SEED_LIST[$SLURM_ARRAY_TASK_ID - 1]}

SCRIPT_PATH="/gpfs/scratch/bsc32/bsc214253/quantifydrivershw/src/quantifydrivers/train_and_shap/training_evaluation_SHAP_pipeline.py"


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

# Run the script inside the uv-managed environment
uv run python "$SCRIPT_PATH" seed=$CURRENT_SEED site=marrakech