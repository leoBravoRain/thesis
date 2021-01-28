#!/bin/bash
#SBATCH -J plasticc_features 
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --tasks-per-node=10
#SBATCH --partition=intel


date
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate astro

# export MKL_NUM_THREAD=1
ipython compute_features_plasticc.py
