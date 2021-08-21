#!/bin/bash

#SBATCH --job-name xai
#SBATCH --output output_%j.txt
#SBATCH --error errors_%j.txt

#SBATCH --mail-type ALL
#SBATCH --mail-user giulia.pezzutti@studenti.unipd.it

# #SBATCH --gres=gpu:1

#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p allgroups
#SBATCH -t 10-00:00:00
#SBATCH --mem 20G

# setup my personal/shared project folder
MY_DIR=/home/pezzuttigi/XAI-EEG/source/

# make my folder the current directory
cd $MY_DIR

# setup python3 environment for machine learning computations 
source /nfsd/opt/anaconda3/anaconda3.sh
conda activate /nfsd/signet/hda_env_gpu

# run code
srun python $MY_DIR/main.py

# deactivate environment at the end of the job
conda deactivate