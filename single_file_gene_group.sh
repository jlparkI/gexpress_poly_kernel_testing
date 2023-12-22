#!/bin/bash

#SBATCH --job-name ggvr
#SBATCH --output ggvr
#SBATCH -w gpu-1
#SBATCH -p gpu

source /stg3/data3/Jonathan/.bashrc
source /stg3/data3/Jonathan/.bash_profile

module load cuda
source ../venv_poly/bin/activate

export CUDA_VISIBLE_DEVICES=3


python run_key_experiments.py --sfgene_exp \
	/stg3/data1/sam/enhancer_prediction/validation/BSS01391_count_matrix_pro.npy \
	/stg3/data3/Jonathan/jonathan2/poly_kernel_motifs/ydata/BSS01391.y.npy
