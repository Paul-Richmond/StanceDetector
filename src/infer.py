"""
infer.py - Script for model inference and evaluation on classification tasks

This script handles loading models (with optional LoRA adapters), running inference,
and evaluating the results using classification metrics. Configuration is managed through Hydra, allowing flexible
parameter adjustment without code changes.
"""

import logging
import os

import huggingface_hub
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import classification_report, f1_score
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Local utilities
from utils.io import load_dataset_splits, save_dict_to_json
from utils.processing import post_process_results

# Load environment variables and authenticate with Hugging Face
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
huggingface_hub.login(token=hf_token)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="../configs", config_name="default_infer")
def main(cfg: DictConfig) -> None:
    """
    Main function for model inference and evaluation.

    Args:
        cfg: Hydra configuration object containing all parameters
    """
    # Extract configuration parameters for dataset and inference
    dataset_col = cfg.dataset.col_to_select  # Column to use as input for the model
    parse_dict = cfg.inference.parse_dict  # Dictionary for parsing model outputs
    support_labels = cfg.inference.f1_support_labels  # Labels to consider for F1 score
    labels = cfg.inference.dataset_labels_column  # Column containing ground truth labels
    stance_target_column = cfg.inference.stance_target_column  # Column for stance targets if applicable

    # Load dataset - expects a DatasetDict with single split
    ds = load_dataset_splits(cfg.dataset)
    ds_split_names = list(ds)
    ds = ds[list(ds)[0]]  # Extract the single Dataset from DatasetDict

    # Configure generation parameters
    if 'generation_cfg' in cfg.inference.keys():
        # Convert OmegaConf to regular dict and initialize SamplingParams
        generation_cfg = OmegaConf.to_container(cfg.inference.generation_cfg, resolve=True)
        sampling_params = SamplingParams(**generation_cfg)
    else:
        # Use default sampling parameters if none specified
        sampling_params = SamplingParams()

    # Handle LoRA adapter configuration if provided
    lora_adapter_name = cfg.inference.get('lora_adapter_name', None)
    model_name = lora_adapter_name if lora_adapter_name is not None else cfg.inference.base_model_name

    if lora_adapter_name is not None:
        # Set up LoRA for fine-tuned inference
        enable_lora = True
        lora_request = LoRARequest("sql_adapter", 1, lora_path=lora_adapter_name)
        logger.info(f"Using LoRA adapter: {lora_adapter_name}")
    else:
        # Use the base model without LoRA
        enable_lora = False
        lora_request = None

    # Initialize the Language Model
    if 'model_cfg' in cfg.inference.keys():
        # Create LLM with custom configuration
        llm = LLM(model=cfg.inference.base_model_name, enable_lora=enable_lora, **cfg.inference.model_cfg)
    else:
        # Create LLM with default configuration
        llm = LLM(model=cfg.inference.base_model_name, enable_lora=enable_lora)

    # Generate completions using the model
    completions = []
    if llm.get_tokenizer().chat_template is not None:
        # Use chat method if model has a chat template
        outputs = llm.chat(ds[dataset_col], sampling_params=sampling_params, lora_request=lora_request)
    else:
        # Use regular generate method for non-chat models
        outputs = llm.generate(ds[dataset_col], sampling_params=sampling_params, lora_request=lora_request)

    # Extract generated text from outputs
    for output in outputs:
        completions.append(output.outputs[0].text)

    # Evaluate model predictions against ground truth
    y_true = ds[labels]
    y_pred = post_process_results(completions, parse_dict)  # Process raw completions into clean predictions

    # Calculate and log macro F1 score
    f_score = f1_score(y_true, y_pred, labels=support_labels, average='macro')
    logger.info(f'Macro F1_av on labels {support_labels}: {f_score:.4f}')

    # Generate and log classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred, zero_division=0.0)
    logger.info(f'Classification report: \n {class_report}')
    class_report_dict = classification_report(y_true=y_true, y_pred=y_pred, zero_division=0.0, output_dict=True)

    # Prepare results dictionary with all relevant information
    output_dict = {
        'model': model_name,
        'dataset': {'name': cfg.dataset.name, 'splits': ds_split_names[0]},
        'sampling_params': sampling_params.__repr__(),
        'prompts': ds[dataset_col],
        'generated_text': completions,
        'y_pred': y_pred,
        'y_true': y_true,
        'macro_f1': f_score,
        'classification_report': class_report_dict,
        'stance_target': ds[stance_target_column]
    }

    # Save results to disk in JSON format
    save_dict_to_json(
        data=output_dict,
        directory=cfg.inference.output_dir,
        filename=cfg.inference.output_fname
    )


if __name__ == "__main__":
    main()
