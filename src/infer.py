import logging
import os

from dotenv import load_dotenv
import huggingface_hub
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import classification_report, f1_score
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from utils.io import load_dataset_splits
from utils.io import save_dict_to_json
from utils.processing import post_process_results

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
huggingface_hub.login(token=hf_token)

# Create a logger and set level to INFO.
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="../configs", config_name="default_infer")
def main(cfg: DictConfig) -> None:
    # Get variables from cfg
    dataset_col = cfg.dataset.col_to_select
    parse_dict = cfg.inference.parse_dict
    support_labels = cfg.inference.f1_support_labels
    labels = cfg.inference.dataset_labels_column
    stance_target_column = cfg.inference.stance_target_column
    #
    ds = load_dataset_splits(cfg.dataset)  # expect DatasetDict object with a single key
    ds_split_names = list(ds)
    ds = ds[list(ds)[0]]  # extract the single Dataset object from the DatasetDict object

    if 'generation_cfg' in cfg.inference.keys():
        generation_cfg = OmegaConf.to_container(cfg.inference.generation_cfg, resolve=True)
        sampling_params = SamplingParams(**generation_cfg)
    else:
        sampling_params = SamplingParams()

    lora_adapter_name = cfg.inference.get('lora_adapter_name', None)
    model_name = lora_adapter_name if lora_adapter_name is not None else cfg.inference.base_model_name

    if lora_adapter_name is not None:
        enable_lora = True
        lora_request = LoRARequest("sql_adapter", 1, lora_path=lora_adapter_name)
        logger.info(f"Using LoRA adapter: {lora_adapter_name}")
    else:
        enable_lora = False
        lora_request = None

    # Create an LLM.
    if 'model_cfg' in cfg.inference.keys():
        llm = LLM(model=cfg.inference.base_model_name, enable_lora=enable_lora, **cfg.inference.model_cfg)
    else:
        llm = LLM(model=cfg.inference.base_model_name, enable_lora=enable_lora)

    # Use llm to generate completions
    completions = []
    if llm.get_tokenizer().chat_template is not None:
        outputs = llm.chat(ds[dataset_col], sampling_params=sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(ds[dataset_col], sampling_params=sampling_params, lora_request=lora_request)
    for output in outputs:
        completions.append(output.outputs[0].text)

    # Score predictions
    y_true = ds[labels]
    y_pred = post_process_results(completions, parse_dict)
    f_score = f1_score(y_true, y_pred, labels=support_labels, average='macro')
    logger.info(f'Macro F1_av on labels {support_labels}: {f_score:.4f}')
    class_report = classification_report(y_true=y_true, y_pred=y_pred, zero_division=0.0)
    logger.info(f'Classification report: \n {class_report}')
    class_report_dict = classification_report(y_true=y_true, y_pred=y_pred, zero_division=0.0)

    # Prepare output_dict and save it to disk
    output_dict = {'model': model_name,
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

    save_dict_to_json(data=output_dict,
                      directory=cfg.inference.output_dir,
                      filename=cfg.inference.output_fname)


if __name__ == "__main__":
    main()
