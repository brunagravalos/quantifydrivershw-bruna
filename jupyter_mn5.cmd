#!/bin/bash
#SBATCH --cpus-per-task 1
#SBATCH --time 01:00:00
##SBATCH --exclusive
#SBATCH --job-name jupyter-notebook-mn5
#SBATCH --output jupyter-notebook-mn5-%J.out
#SBATCH --error  jupyter-notebook-mn5-%J.err
#SBATCH --account bsc32
#SBATCH --qos gp_bsces

# Get tunneling info:
XDG_RUNTIME_DIR=""
PORT=$(shuf -i8000-9999 -n1)
NODE=$(hostname -s)
USER=$(whoami)

# Print tunneling instructions in the jupyter-notebook-log:
echo -e "

MacOS or linux terminal command to create your ssh tunnel
ssh -N -L ${PORT}:${NODE}:${PORT} ${USER}@glogin4.bsc.es

Use a Browser on your local machine to go to:
http://localhost:${PORT}  (prefix w/ https:// if using password)
"

# Load modules or conda environments here:
module load JupyterNotebook/7.1.3-GCCcore-13.2.0-Python-3.11.5

# Launch Jupyter with the command below:
jupyter-lab --no-browser --port=${PORT} --ip=${NODE}