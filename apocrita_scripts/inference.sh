#!/bin/bash
#$ -l h_rt=1:0:0
#$ -l h_vmem=3.0G
#$ -pe smp 8
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -cwd
#$ -j y
#$ -m b

cd ~

module load miniforge
module load gcc
module load cuda/12.2

conda activate fsi

# change where the huggingface dataset and model are cached to
# see https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
export HF_HOME=/data/scratch/$USER/.cache/huggingface/hub

cd /data/scratch/$USER/StanceDetector/src

# Example of inference using a foundational LLM
# We remove the train dataset split with ~dataset.splits.train=train to only use test split
# We remove the lora_adapter_name with ~inference.lora_adapter_name i.e. use base model only
# We override the maximum number of tokens to generate with inference.generation_cfg.max_tokens=50
# We specify the output directory with inference.output_dname=Llama-2-7b-chat-hf
# We specify the output filename with inference.output_fname=messages_0
python3 infer.py ~dataset.splits.train=train \
~inference.lora_adapter_name \
inference.generation_cfg.max_tokens=50 \
inference.output_dname=Llama-2-7b-chat-hf \
inference.output_fname=messages_0

# Example of inference using a finetuned LLM
# We remove the train dataset split to only use test
# We specify the lora_adapter_name with inference.lora_adapter_name=FundSciImpact/llama-2-7b_ft i.e. we use a fine-tuned model
# We specify the output directory with inference.output_dname=llama-2-7b_ft
# We specify the output filename with inference.output_fname=messages_0
#python3 infer.py ~dataset.splits.train=train \
#inference.lora_adapter_name=FundSciImpact/llama-2-7b_ft \
#inference.output_dname=llama-2-7b_ft \
#inference.output_fname=messages_0
