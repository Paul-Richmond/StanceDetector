from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from peft import (LoraConfig,
                  PeftModel,
                  get_peft_model,
                  prepare_model_for_kbit_training,
                  PromptTuningConfig)
from unsloth import FastLanguageModel, is_bfloat16_supported, get_chat_template
from torch.optim import Optimizer
from transformers import (AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          PreTrainedModel)
from typing import Any, Dict, Union


def load_optimizer(optimizer_cfg: DictConfig, model: PreTrainedModel) -> Optimizer:
    """
    Instantiate and return an optimizer based on the provided configuration.

    Args:
        optimizer_cfg (DictConfig): Configuration for the optimizer.
        model (PreTrainedModel): The model whose parameters will be optimized.

    Returns:
        Optimizer: The instantiated optimizer.
    """
    optimizer = instantiate(optimizer_cfg, params=model.parameters(), **optimizer_cfg)
    return optimizer


def load_fastlanguagemodel(model_name: str, cfg: DictConfig) -> tuple:
    """
    Instantiate and return an Unsloth loaded model and tokenizer
    based on the provided configuration.

    Args:
        model_name (str): The repository name of the model and tokenizer.
        cfg (DictConfig): Configuration for the model and tokenzier.

    Returns:
        tuple: A tuple containing the instantiated model and tokenizer.
    """
    # need to do this to get mixture of positional and keyword arguments
    model_cfg = OmegaConf.to_container(cfg.model_cfg, resolve=True)

    model, tokenizer = FastLanguageModel.from_pretrained(model_name, **model_cfg)

    if 'lora_cfg' in cfg.keys():
        # we need to use OmegaConf.to_container to avoid json serialization errors when saving model
        model = FastLanguageModel.get_peft_model(model, **OmegaConf.to_container(cfg.lora_cfg, resolve=True))

    if tokenizer.chat_template is not None:
        tokenizer = get_chat_template(tokenizer, chat_template=cfg.chat_template)

    return model, tokenizer


def load_automodelforcausallm(cfg: DictConfig) -> Union[PreTrainedModel, PeftModel]:
    """
    Load and configure a PreTrainedModel or PeftModel based on the provided configuration.

    Args:
        cfg (DictConfig): Configuration containing model, quantization, and LoRA settings.

    Returns:
        Union[PreTrainedModel, PeftModel]: The configured causal language model.
    """
    model_cfg: Dict[str, Any] = OmegaConf.to_container(cfg.model_cfg, resolve=True)
    model_name: str = model_cfg.pop('name')

    if 'bnb_cfg' in cfg.keys():
        quant_config: BitsAndBytesConfig = BitsAndBytesConfig(**OmegaConf.to_container(cfg.bnb_cfg, resolve=True))
        model: Union[PreTrainedModel, PeftModel] \
            = AutoModelForCausalLM.from_pretrained(model_name, **model_cfg, quantization_config=quant_config)
        model = prepare_model_for_kbit_training(model)
    else:
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name, **model_cfg)

    if 'lora_cfg' in cfg.keys():
        lora_config: LoraConfig = LoraConfig(**OmegaConf.to_container(cfg.lora_cfg, resolve=True))
        model = get_peft_model(model, lora_config)
    elif 'prompt_cfg' in cfg.keys():
        prompt_config = PromptTuningConfig(**OmegaConf.to_container(cfg.prompt_cfg, resolve=True))
        model = get_peft_model(model, prompt_config)

    return model


def instantiate_training(cfg: DictConfig) -> tuple:
    """
    Instantiate trainer and training arguments based on the provided configuration.

    Args:
        cfg (DictConfig): Configuration containing trainer and training arguments.

    Returns:
        tuple: A tuple containing the instantiated trainer and training arguments.
    """
    partial_trainer = instantiate(cfg.trainer_cfg)
    tr_args = instantiate(cfg.training_args_cfg,
                          fp16=not is_bfloat16_supported(),
                          bf16=is_bfloat16_supported(),
                          **OmegaConf.to_container(cfg.training_args_cfg, resolve=True))
    return partial_trainer, tr_args
