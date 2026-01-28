#!/bin/bash
#SBATCH --job-name=dudu_cyclic_linear_high_dimension
#SBATCH --account=gts-yxie77-paid
#SBATCH -p gpu-v100
#SBATCH -N 1
#SBATCH --gres=gpu:V100:1
#SBATCH --mem-per-gpu=40G
#SBATCH -t 3:00:00
#SBATCH -q inferno
#SBATCH -o /storage/home/hcoda1/3/dwu381/p-yxie77-0/Flow_Matrix/sbatch_output/output_%j.txt
#SBATCH -e /storage/home/hcoda1/3/dwu381/p-yxie77-0/Flow_Matrix/sbatch_output/output_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dwu381@gatech.edu

cd $SLURM_SUBMIT_DIR

module load anaconda3/2022.05
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /storage/coda1/p-yxie77/0/dwu381/dongze
cd /storage/home/hcoda1/3/dwu381/p-yxie77-0/DoFlow

python /storage/home/hcoda1/3/dwu381/p-yxie77-0/DoFlow/main_simulation.py