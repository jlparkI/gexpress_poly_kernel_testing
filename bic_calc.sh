#!/bin/bash

#SBATCH --job-name aic_bic
#SBATCH --output aic_bic
#SBATCH -w gpu-1
#SBATCH -p gpu

source /stg3/data3/Jonathan/.bashrc
source /stg3/data3/Jonathan/.bash_profile

module load cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/stg3/data3/Jonathan/cuda_reqs/

source /stg3/data3/Jonathan/jonathan2/poly_kernel_motifs/venv_poly/bin/activate

export CUDA_VISIBLE_DEVICES=1


mkdir /scratch/dstore_2


python run_key_experiments.py --prom_path /stg3/data1/sam/enhancer_prediction/fimo_scan/motif_count_matrices_pro_3_overlapped \
	--ypath /stg3/data1/sam/enhancer_prediction/training_y \
	--en_path /stg3/data3/Jonathan/jonathan2/poly_kernel_motifs/motif_count_matrices_enh_3\
	--nonred_fpath /stg3/data3/Jonathan/jonathan2/poly_kernel_motifs/EpiMapID_Name_nonDup.txt \
	--storage /scratch/dstore_2 \
	--exp_type bic \
        --add_interactions

rm -rf /scratch/dstore_2
