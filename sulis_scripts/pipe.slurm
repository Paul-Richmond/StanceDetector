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

srun python3 ~/StanceDetector/src/finetune.py training.training_args_cfg.run_name=llama-2-7b_ft \
model.model_name=meta-llama/Llama-2-7b-hf \
dataset.col_to_select=text_0

srun python3 ~/StanceDetector/src/infer.py ~dataset.splits.train=train \
dataset.col_to_select=text_0 \
inference.base_model_name=meta-llama/Llama-2-7b-hf \
inference.lora_adapter_name=FundSciImpact/llama-2-7b_ft \
inference.output_dname=llama-2-7b_ft \
inference.output_fname=text_0
