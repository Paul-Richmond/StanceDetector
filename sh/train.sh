#!/bin/bash
#$ -l h_rt=1:0:0
#$ -l h_vmem=3.0G
#$ -pe smp 12
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -cwd
#$ -j y
#$ -m b
cd ~

module load anaconda3
module load gcc
module load cuda/12.2

mamba activate fsi

# change where the huggingface dataset and model are cached to
# see https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
export HF_HOME=/data/scratch/$USER/.cache/huggingface/hub

cd /data/scratch/$USER/DeepImpactLLM/src

python3 finetune.py training.training_args_cfg.run_name=Llama-2-7b-chat \
