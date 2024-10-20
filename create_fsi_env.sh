#!/bin/bash
#$ -l h_rt=1:0:0
#$ -l h_vmem=3.0G
#$ -pe smp 12
#$ -l gpu=1
#$ -cwd
#$ -j y

cd ~

module load anaconda3
module load gcc
module load cuda/12.2

conda create -y -n fsi python=3.10
conda activate fsi

pip install --upgrade pip
pip install -r ./fsi.txt

python -m torch.utils.collect_env
