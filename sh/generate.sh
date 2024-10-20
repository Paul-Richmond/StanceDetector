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

conda activate fsi

# change where the huggingface dataset and model are cached to
# see https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
export HF_HOME=/data/scratch/$USER/.cache/huggingface/hub

cd /data/scratch/$USER/StanceDetector/src

# Example of generating using a foundational LLM
python3 infer.py ~dataset.splits.train=train \
~inference.lora_adapter_name \
inference.generation_cfg.max_tokens=50 \
inference.output_dname=Llama-2-7b-chat-hf \
inference.output_fname=messages_0

# Example of generating using a finetuned LLM
#python3 infer.py ~dataset.splits.train=train \
#inference.lora_adapter_name=FundSciImpact/llama-2-7b_ft \
#inference.output_dname=llama-2-7b_ft \
#inference.output_fname=messages_0
