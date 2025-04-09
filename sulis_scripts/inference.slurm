#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=8:00:00
#SBATCH --account=su008-apw822

#set -x  #<-- uncomment for verbose output

module purge
module load GCC/12.2.0 CUDA/12.2.0 Python/3.10.8

source ~/fsi/bin/activate

# Example of generating using a foundational LLM
srun python3 ~/StanceDetector/src/infer.py ~dataset.splits.train=train \
~inference.lora_adapter_name \
inference.generation_cfg.max_tokens=150 \
inference.output_dname=Llama-2-7b-chat-hf \
inference.output_fname=messages_0

# Example of generating using a finetuned LLM
#python3 infer.py ~dataset.splits.train=train \
#inference.lora_adapter_name=FundSciImpact/llama-2-7b_ft \
#inference.output_dname=llama-2-7b_ft \
#inference.output_fname=messages_0
