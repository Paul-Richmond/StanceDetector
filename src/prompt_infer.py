import os
import gc
from dotenv import load_dotenv
import huggingface_hub
from datasets import load_dataset

import torch
from unsloth import is_bfloat16_supported

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.io import batch_dataset
from utils.processing import format_data

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
huggingface_hub.login(token=hf_token)


DATA_REPO = 'FundSciImpact/semeval-2016'
MODEL_REPO = 'meta-llama/Llama-2-7b-hf'
ADAPTER_REPO = 'FundSciImpact/Llama-2-7b-hf-prompt_tuned'
TOKENIZER_REPO = 'meta-llama/Llama-2-7b-hf'


class Completer:

    def __init__(self, model, tokenizer, dataset, batch_size=None, generation_config=None, col_to_tokenize=None):
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.original_pad_side = tokenizer.padding_side

        self.dataset = dataset
        self.batch_size = 16 if batch_size is None else batch_size

        if generation_config is None:
            self.generation_config = {
                # "temperature": 0.7,
                # "top_p": 0.1,
                # "repetition_penalty": 1.18,
                # "top_k": 40,
                # "do_sample": True,
                "max_new_tokens": 16,
                "pad_token_id": self.tokenizer.pad_token_id,
                "max_time": 60
            }
        else:
            self.generation_config = generation_config

        if self.generation_config.get("pad_token_id", None) is None:
            self.generation_config.update({"pad_token_id": self.tokenizer.pad_token_id})

        self.label = 'abstract' if col_to_tokenize is None else col_to_tokenize

    def predict(self, batch):
        texts = batch[self.label]
        self.tokenizer.padding_side = 'left'  # inference needs left padding
        # clear some memory
        torch.cuda.empty_cache()
        gc.collect()
        with torch.no_grad():
            model_inputs = self.tokenizer(texts,
                                          padding='longest',
                                          pad_to_multiple_of=8,
                                          add_special_tokens=False,
                                          return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**model_inputs, **self.generation_config).to("cpu")
        batch["predictions"] = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.tokenizer.padding_side = self.original_pad_side
        return batch

    def get_predictions(self):
        """
        Processes the dataset to generate and parse predictions.

        The dataset is first split into prompts and ground truths, then predictions are generated
        for each prompt, and finally, the predictions are parsed.

        Returns:
            datasets.Dataset: The processed dataset with predictions included.
        """
        self.dataset = self.dataset.map(self.predict,
                                        batched=True,
                                        batch_size=self.batch_size,
                                        desc='Generating model output')
        return self.dataset


def main():
    ds = load_dataset(DATA_REPO, split='test')
    ds = ds.map(format_data,
                batched=True,
                fn_kwargs={"eval_set": False},
                desc=f"Applying format_data"
                )
    major_version, minor_version = torch.cuda.get_device_capability()
    if major_version == 9:
        # H100 or A100 so go big
        batch_size = 128
    else:
        # Smaller GPU memory so go smaller
        batch_size = 48

    batches = batch_dataset(ds, batch_size=batch_size)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    generation_config = {
        # "temperature": 0.7,
        # "top_p": 0.1,
        # "repetition_penalty": 1.18,
        # "top_k": 40,
        # "do_sample": True,
        "max_new_tokens": 16,
        "pad_token_id": tokenizer.pad_token_id,
        "max_time": 60
    }

    model = AutoModelForCausalLM.from_pretrained(MODEL_REPO,
                                                 torch_dtype='bfloat16' if is_bfloat16_supported() else 'float16',
                                                 device_map='auto',
                                                 attn_implementation='flash_attention_2' if is_bfloat16_supported() else 'sdpa')
    model = PeftModel.from_pretrained(model, ADAPTER_REPO)
    model.to(device='cuda')
    model.eval()

    predictions = []
    for batch in batches:
        texts = batch['text']
        with torch.no_grad():
            model_inputs = tokenizer(texts,
                                     padding='longest',
                                     pad_to_multiple_of=8,
                                     add_special_tokens=False,
                                     return_tensors="pt").to(model.device)
            outputs = model.generate(**model_inputs, **generation_config).to("cpu")
        batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_predictions)

    ds.add_column("predictions", predictions)

    ds.push_to_hub('FundSciImpact/prompt_tuned_predictions')

    huggingface_hub.logout()


if __name__ == "__main__":
    main()
