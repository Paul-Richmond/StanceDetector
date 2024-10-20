#!/bin/bash
#$ -l h_rt=1:0:0
#$ -l h_vmem=3.0G
#$ -pe smp 12
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -cwd
#$ -j y
#$ -m b
#$ -N pipe_llama-2-7b_ft
cd ~

module load anaconda3
module load gcc
module load cuda/12.2

mamba activate fsi

# change where the huggingface dataset and model are cached to
# see https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
export HF_HOME=/data/scratch/$USER/.cache/huggingface/hub

cd /data/scratch/$USER/DeepImpactLLM/src

python3 finetune.py training.training_args_cfg.run_name=llama-2-7b_ft \
model.model_name=meta-llama/Llama-2-7b-hf \
dataset.col_to_select=text_0

python3 infer.py ~dataset.splits.train=train \
inference.base_model_name=meta-llama/Llama-2-7b-hf \
inference.lora_adapter_name=FundSciImpact/llama-2-7b_ft \
inference.output_dname=llama-2-7b_ft \
inference.output_fname=text_0
