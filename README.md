# StanceDetector

---

A pipeline for detecting stance in social media comments.
It uses the 🤗 Hugging Face libraries as well as 🦥 Unsloth,
vLLM and, hopefully soon, 🦜️🔗 LangChain.

## 1. Getting started

---

### 1.1 Setup necessary accounts & hardware

You need a Hugging Face account to allow access to gated LLMs and private models
and datasets. For training, you will also need a Weights & Biases (W&B) account. 

Since both training and inference are compute intensive, 
you should have access to an Nvidia GPU with a reasonable amount of RAM.
We have used QMUL's Apocrita HPC cluster (https://docs.hpc.qmul.ac.uk) and
the Sulis cluster (https://sulis-hpc.github.io) both of which have Ampere family
GPUs with at least 40Gb of RAM.

As of 21st October 2024, Unsloth only runs on a single GPU.

### 1.2 Clone this repository

Clone or download this repository and transfer it to cluster storage via FileZilla. 
For example, on Apocrita to clone this repository use
```shell
cd /data/scratch/$USER/
git clone git@github.com:Paul-Richmond/StanceDetector.git
```


### 1.3 Create `.env` file

Once you have your Hugging Face and W&B accounts and API keys, 
you need to copy the file `.env_example` and rename it to `.env` 
(this is included in `.gitignore` and doesn't expose your private keys). 
Then paste your private Hugging Face and W&B API keys into the renamed file.

### 1.4 Create environment

#### 1.4.1 On Apocrita
For use on Apocrita I have included a shell script to set up a new conda environment
called `fsi` and install the packages contained within `fsi.txt`.

To run the script log on to Apocrita and execute
```shell
cd /data/scratch/$USER/StanceDetector

squb create_fsi_env.sh
```
#### 1.4.2 On Sulis

Sulis does not use conda, so we have to create a python virtual environment instead.


## 2. Usage

---

The pipeline is designed to be flexible and allow for different 
models, tokenizers, datasets, optimizers and training hyperparameters
as well as custom trainers and callbacks.

A lot of this flexibility is enabled by using the [hydra package](https://hydra.cc)
whose key feature is
> the ability to dynamically create a hierarchical configuration
> by composition and override it through config files and the command line.

The hydra configuration files are stored within the `configs` directory as follows
```
├── configs
│   ├── dataset
│   │   ├── *.yaml
│   ├── inference
│   │   ├── *.yaml
│   ├── model
│   │   ├── *.yaml
│   ├── optimizer
│   │   ├── *.yaml
│   ├── training 
│   │   ├── *.yaml
│   ├── default.yaml   
│   ├── default_infer.yaml 
```

### 2.1 Training

TBF

### 2.2 Inference

The script `src/infer.py` 
1. downloads a preformatted dataset from Hugging Face
2. uses vLLM to generate responses to the prompt/chat inputs contained within the dataset
3. parses the generated responses and computes the macro average F1 score and classification report
4. saves a JSON file wit the results


The script is controlled through the yaml files contained within the configs directory.

#### 2.2.1 `configs/dataset`

Each yaml file in this directory should contain

`name`: A Hugging Face repository for a preformatted dataset.

`splits`: A dictionary containing the dataset splits to be loaded.
**NB** Currently `infer.py` only handles a single dataset split.

`col_to_select`: The name of the column within the dataset to use as inputs.

#### 2.2.2 `configs/inference`

Each yaml file in this directory should contain

`base_model_name`: A Hugging Face repository for a foundational LLM.

`lora_adapter_name`: An optional Hugging Face repository containing LoRA adapters. 
Set the value to `null` if no LoRA adapters are to be used.

`model_cfg`: An optional dictionary containing keyword parameters passed to vLLM's `LLM` class.

For example using 
```yaml
model_cfg:
  quantization: bitsandbytes
  load_format: bitsandbytes
```
would load a quantized model using the `bitsandbytes` package.

`generation_cfg`: An optional dictionary containing keyword parameters passed to vLLM's
`SamplingParameters` class. It controls how vLLM generates responses.

`parse_dict`: A dictionary of four dictionaries which controls how the parsing of
the vLLM generated repsonses happens. The inner dictionaries have keys disagree, agree,
neutral and failed. The first 