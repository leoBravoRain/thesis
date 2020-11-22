#!/bin/bash
#SBATCH -J Ejemplo
#SBATCH --nodes=1
#SBATCH --mem=10gb
#SBATCH --tasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --partition=intel

ID=$SLURM_JOB_ID

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate astro

date
gpu_ids=`/usr/local/bin/tarjeta_libre 1 ,`
if [ ! -z ${gpu_ids} ]
then
  CUDA_VISIBLE_DEVICES=${gpu_ids} python -u train_classifier.py
fi
