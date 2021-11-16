#!/bin/bash

#SBATCH -J xai
#SBATCH -o %j_output.txt
#SBATCH -e %j_errors.txt

#SBATCH --mail-type ALL
#SBATCH --mail-user giulia.pezzutti@studenti.unipd.it

#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p allgroups
#SBATCH -t 10-00:00:00
#SBATCH --mem 80G

#SBATCH --gres=gpu:rtx

# setup my personal/shared project folder
DIRECTORY=/home/pezzuttigi/XAI-EEG/source/
DATA=/home/pezzuttigi/XAI-EEG/dataset/EEG/
OUTPUT=/home/pezzuttigi/XAI-EEG/output/

# make my folder current directory
cd $DIRECTORY

srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY --bind $OUTPUT\
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/main.py 