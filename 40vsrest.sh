#!/bin/bash

#SBATCH --job-name 40vr
#SBATCH --output 40vr
#SBATCH -w gpu-2
#SBATCH -p gpu

source /stg3/data3/Jonathan/.bashrc
source /stg3/data3/Jonathan/.bash_profile

module load cuda
source ../venv_poly/bin/activate

export CUDA_VISIBLE_DEVICES=3


mkdir /scratch/dstore_2
python run_key_experiments.py --prom_path /stg3/data1/sam/enhancer_prediction/fimo_scan/motif_count_matrices_pro_3_overlapped \
	--ypath /stg3/data3/Jonathan/jonathan2/poly_kernel_motifs/ydata \
	--nonred_fpath /stg3/data3/Jonathan/jonathan2/poly_kernel_motifs/EpiMapID_Name_nonDup.txt \
	--storage /scratch/dstore_2 \
	--exp_type 40vsrest

rm -rf /scratch/dstore_2

mkdir /scratch/dstore_2
python run_key_experiments.py --prom_path /stg3/data1/sam/enhancer_prediction/fimo_scan/motif_count_matrices_pro_3 \
	--ypath /stg3/data3/Jonathan/jonathan2/poly_kernel_motifs/ydata \
	--nonred_fpath /stg3/data3/Jonathan/jonathan2/poly_kernel_motifs/EpiMapID_Name_nonDup.txt \
	--storage /scratch/dstore_2 \
	--exp_type 40vsrest

rm -rf /scratch/dstore_2
