"""
finetune.py - Language Model Fine-Tuning Script

This script orchestrates the fine-tuning process for language models using the Hugging Face ecosystem.
It handles loading datasets, setting up model configurations, initializing optimizers, and managing
the training process while logging experiments with Weights & Biases. The script supports both
chat-template and regular language models, applying appropriate tokenization strategies based on
model type. Configuration is managed through Hydra, allowing flexible parameter adjustment without
code changes. After training, the model is automatically uploaded to the Hugging Face Hub.
"""

import os

import huggingface_hub
import hydra
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig

# Local utilities
from utils.instantiators import (load_fastlanguagemodel,
                                 load_optimizer,
                                 instantiate_training)
from utils.io import load_dataset_splits
from utils.processing import apply_chat_template, add_eos_token

# Load environment variables and authenticate with Hugging Face and W&B
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
wandb_token = os.getenv("WANDB_API_KEY")
huggingface_hub.login(token=hf_token)
wandb.login(key=wandb_token)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Main function to run language model fine-tuning process.

    Args:
        cfg (DictConfig): Configuration object with parameters for the training process
    """
    # Load dataset splits determined by dataset configuration file
    ds = load_dataset_splits(cfg.dataset)
    # Get column name that will be tokenized
    col_to_tokenize = cfg.dataset.col_to_select

    # Initialize trainer class and training arguments
    trainer_cls, training_args = instantiate_training(cfg.training)

    # Initialize model and tokenizer
    model, tokenizer = load_fastlanguagemodel(cfg.model.model_name, cfg.model)

    # Initialize optimizer for training
    optimizer = load_optimizer(cfg.optimizer, model)

    # Handle different tokenization strategies based on whether model has a chat template
    if tokenizer.chat_template is not None:
        # For chat models, use the 'messages' field
        training_args.dataset_text_field = 'messages'
        for dataset_ in ds:
            # For training set, don't add generation prompt; for other sets, add it
            bool_flag = False if dataset_ == 'train' else True
            # Apply chat template formatting to dataset
            ds[dataset_] = ds[dataset_].map(apply_chat_template,
                                            batched=True,
                                            fn_kwargs={"tokenizer": tokenizer,
                                                       "col_to_tokenize": col_to_tokenize,
                                                       "add_generation_prompt": bool_flag},
                                            desc="Applying apply_template")
    else:
        # For non-chat models, use the 'text' field
        training_args.dataset_text_field = 'text'
        for dataset_ in ds:
            # For training set, use different processing than evaluation sets
            bool_flag = False if dataset_ == 'train' else True
            # Add EOS token to train dataset texts
            ds[dataset_] = ds[dataset_].map(add_eos_token,
                                            batched=True,
                                            fn_kwargs={"tokenizer": tokenizer,
                                                       "col_to_tokenize": col_to_tokenize,
                                                       "eval_set": bool_flag},
                                            desc="Adding eos_token")

    # Initialize the trainer with all components
    trainer = trainer_cls(model=model,
                          args=training_args,
                          train_dataset=ds.get('train'),
                          eval_dataset=ds.get('test'),
                          data_collator=None,
                          tokenizer=tokenizer,
                          optimizers=(optimizer, None),
                          callbacks=None,
                          compute_metrics=None,
                          preprocess_logits_for_metrics=None,
                          formatting_func=None,
                          )

    # Start training process
    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)

    # Push trained model to Hugging Face Hub

    trainer.push_to_hub()
    # Clean up connections to services
    wandb.finish()
    huggingface_hub.logout()


if __name__ == "__main__":
    main()
