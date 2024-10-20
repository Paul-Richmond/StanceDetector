import os

from dotenv import load_dotenv
import huggingface_hub
import hydra
from omegaconf import DictConfig
import wandb

from utils.instantiators import (load_fastlanguagemodel,
                                 load_optimizer,
                                 instantiate_training, )
from utils.io import load_dataset_splits
from utils.processing import apply_chat_template, add_eos_token

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
wandb_token = os.getenv("WANDB_API_KEY")

huggingface_hub.login(token=hf_token)
wandb.login(key=wandb_token)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    ds = load_dataset_splits(cfg.dataset)
    col_to_tokenize = cfg.dataset.col_to_select
    trainer_cls, training_args = instantiate_training(cfg.training)
    # get the model and tokenizer via unsloth
    model, tokenizer = load_fastlanguagemodel(cfg.model.model_name, cfg.model)
    # get the optimizer
    optimizer = load_optimizer(cfg.optimizer, model)

    if tokenizer.chat_template is not None:
        training_args.dataset_text_field = 'messages'
        for dataset_ in ds:
            bool_flag = False if dataset_ == 'train' else True
            ds[dataset_] = ds[dataset_].map(apply_chat_template,
                                            batched=True,
                                            fn_kwargs={"tokenizer": tokenizer,
                                                       "col_to_tokenize": col_to_tokenize,
                                                       "add_generation_prompt": bool_flag},
                                            desc="Applying apply_template")
    else:
        training_args.dataset_text_field = 'text'
        for dataset_ in ds:
            bool_flag = False if dataset_ == 'train' else True
            ds[dataset_] = ds[dataset_].map(add_eos_token,
                                            batched=True,
                                            fn_kwargs={"tokenizer": tokenizer,
                                                       "col_to_tokenize": col_to_tokenize,
                                                       "eval_set": bool_flag},
                                            desc="Adding eos_token")

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

    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)
    trainer.push_to_hub()
    wandb.finish()
    huggingface_hub.logout()


if __name__ == "__main__":
    main()
