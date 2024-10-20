import os
from dotenv import load_dotenv
import huggingface_hub
from datasets import load_dataset
from omegaconf import OmegaConf

from peft import TaskType, PromptTuningInit
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer
from unsloth import is_bfloat16_supported

from utils.instantiators import load_automodelforcausallm
from utils.processing import add_eos_token, format_data

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
huggingface_hub.login(token=hf_token)

DATA_REPO = 'FundSciImpact/semeval-2016'
MODEL_REPO = 'meta-llama/Llama-2-7b-hf'
TOKENIZER_REPO = 'meta-llama/Llama-2-7b-hf'

PROMPT = """You are a helpful, respectful, and honest assistant for stance detection for a given target. 
Always answer from the possible options given below as helpfully as possible. 
Stance detection is the process of determining whether the author of a tweet is in support of or against a given target. 
The target may not always be explicitly mentioned in the text, and the tweet’s stance can be conveyed implicitly through subtext, regional and cultural references, or other implicit meanings.
The possible stances are:
* support: The tweet has a positive or supportive attitude towards the target, either explicitly or implicitly.
* against: The tweet opposes or criticizes the target, either explicitly or implicitly.
* none: The tweet is neutral or doesn’t have a stance towards the target.
"""

MODEL_DICT = {'model_cfg': {'name': MODEL_REPO,
                            'device_map': 'auto',
                            'trust_remote_code': True,
                            'attn_implementation': 'flash_attention_2' if is_bfloat16_supported() else 'sdpa'},
              'bnb_cfg': {'load_in_4bit': True,
                          'bnb_4bit_compute_dtype': 'bfloat16' if is_bfloat16_supported() else 'float16',
                          'bnb_4bit_quant_storage': None,
                          'bnb_4bit_quant_type': 'nf4',
                          'bnb_4bit_use_double_quant': True},
              'prompt_cfg': {'task_type': TaskType.CAUSAL_LM,
                             'prompt_tuning_init': PromptTuningInit.TEXT,
                             'prompt_tuning_init_text': PROMPT,
                             'num_virtual_tokens': 30,
                             'tokenizer_name_or_path': MODEL_REPO},
              }

TRAINING_ARGS_DICT = {'num_train_epochs': 3,
                      'eval_strategy': 'steps',
                      'eval_steps': 0.111,
                      'eval_on_start': True,
                      'save_strategy': 'steps',
                      'save_steps': 0.111,
                      'log_level': 'info',
                      'logging_steps': 1,
                      'max_grad_norm': 1.0,
                      'auto_find_batch_size': True,
                      'fp16': not is_bfloat16_supported(),
                      'bf16': is_bfloat16_supported(),
                      'per_device_train_batch_size': 64,
                      'per_device_eval_batch_size': 64,
                      'output_dir': 'FundSciImpact/Llama-2-7b-hf-prompt_tuned_v6',
                      'report_to': None,
                      'push_to_hub': True,
                      'hub_model_id': 'FundSciImpact/Llama-2-7b-hf-prompt_tuned_v6',
                      'hub_private_repo': True,
                      'optim': 'adamw_8bit',
                      'learning_rate': 3e-04,
                      'lr_scheduler_type': 'cosine_with_min_lr',
                      'warmup_ratio': 0.1,
                      'lr_scheduler_kwargs': {'min_lr_rate': 0.1},
                      'dataset_text_field': 'text',
                      'packing': False,
                      'max_seq_length': 1024
                      }


# def load_automodelforcausallm(cfg: DictConfig) -> Union[PreTrainedModel, PeftModel]:
#     """
#     Load and configure a PreTrainedModel or PeftModel based on the provided configuration.
#
#     Args:
#         cfg (DictConfig): Configuration containing model, quantization, and LoRA settings.
#
#     Returns:
#         Union[PreTrainedModel, PeftModel]: The configured causal language model.
#     """
#     model_cfg: Dict[str, Any] = OmegaConf.to_container(cfg.model_cfg, resolve=True)
#     model_name: str = model_cfg.pop('name')
#
#     if 'bnb_cfg' in cfg.keys():
#         quant_config: BitsAndBytesConfig = BitsAndBytesConfig(**OmegaConf.to_container(cfg.bnb_cfg, resolve=True))
#         model: Union[PreTrainedModel, PeftModel] \
#             = AutoModelForCausalLM.from_pretrained(model_name, **model_cfg, quantization_config=quant_config)
#         model = prepare_model_for_kbit_training(model)
#     else:
#         model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name, **model_cfg)
#
#     if 'lora_cfg' in cfg.keys():
#         lora_config: LoraConfig = LoraConfig(**OmegaConf.to_container(cfg.lora_cfg, resolve=True))
#         model = get_peft_model(model, lora_config)
#     elif 'prompt_cfg' in cfg.keys():
#         prompt_config = PromptTuningConfig(**OmegaConf.to_container(cfg.prompt_cfg, resolve=True))
#         model = get_peft_model(model, prompt_config)
#
#     return model


def main():
    ds = load_dataset(DATA_REPO)
    model_cfg = OmegaConf.create(MODEL_DICT)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    tokenizer.pad_token = tokenizer.eos_token

    for dataset_ in ds:
        bool_flag = False if dataset_ == 'train' else True
        ds[dataset_] = ds[dataset_].map(format_data,
                                        batched=True,
                                        fn_kwargs={"eval_set": bool_flag},
                                        desc=f"Applying format_data"
                                        )
        ds[dataset_] = ds[dataset_].map(add_eos_token,
                                        batched=True,
                                        fn_kwargs={"tokenizer": tokenizer,
                                                   "col_to_tokenize": 'text',
                                                   "eval_set": bool_flag},
                                        desc="Adding eos_token")

    model = load_automodelforcausallm(model_cfg)
    model.print_trainable_parameters()
    training_args = SFTConfig(**TRAINING_ARGS_DICT)
    trainer = SFTTrainer(model=model,
                         args=training_args,
                         train_dataset=ds.get('train'),
                         eval_dataset=ds.get('test'),
                         data_collator=None,
                         tokenizer=tokenizer,
                         callbacks=None,
                         compute_metrics=None,
                         preprocess_logits_for_metrics=None,
                         formatting_func=None,
                         )
    trainer.train()
    trainer.push_to_hub()
    huggingface_hub.logout()


if __name__ == "__main__":
    main()
