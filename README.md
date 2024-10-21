# StanceDetector

---

A pipeline for detecting stance in social media comments.
It uses the 🤗 Hugging Face libraries as well as 🦥 Unsloth,
vLLM and 🦜️🔗 LangChain.

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

The hydra configuration files are stored within the `configs` directory.

### 2.1 Training

TBF

### 2.2 Inference

TBF