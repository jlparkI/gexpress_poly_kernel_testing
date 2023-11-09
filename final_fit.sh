#!/bin/bash

#SBATCH --job-name f50
#SBATCH --output f50
#SBATCH -w gpu-2
#SBATCH -p gpu

source /stg3/data3/Jonathan/.bashrc
source /stg3/data3/Jonathan/.bash_profile

module load cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/stg3/data3/Jonathan/cuda_reqs/
conda activate xgpr

export CUDA_VISIBLE_DEVICES=2

mkdir /scratch/temp_dstore

python run_key_experiments.py --prom_path /stg3/data1/sam/enhancer_prediction/fimo_scan/motif_count_matrices_pro_3_overlapped \
	--ypath /stg3/data3/Jonathan/poly_kernel_motifs/ydata \
	--en_path /stg3/data3/Jonathan/poly_kernel_motifs/motif_count_matrices_enh_3\
	--nonred_fpath /stg3/data3/Jonathan/poly_kernel_motifs/EpiMapID_Name_nonDup.txt \
	--storage /scratch/temp_dstore \
	--exp_type final

rm -rf /scratch/temp_dstore
