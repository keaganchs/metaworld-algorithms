#!/bin/bash

#SBATCH --job-name=mtsac_mt10
#SBATCH --output=log/out_and_err_%j.txt
#SBATCH --error=log/out_and_err_%j.txt
#SBATCH --partition=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=2000
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:1

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate malg

export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

python mtsac_mt10.py \
    --track True \
    --wandb_project "dime" \
    --wandb_entity "keagan" \
