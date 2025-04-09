# StanceDetector

---

A pipeline for detecting stance in social media comments. It uses the 🤗 Hugging Face 
libraries as well as 🦥 Unsloth, vLLM and 🦜️🔗 LangChain.

---

## Table of Contents

- [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup Instructions](#setup-instructions)
    - [Optional - Installing Ollama](#optional---installing-ollama)
- [Model fine-tuning and inference structure](#model-fine-tuning-and-inference-structure)
    - [Core Architecture](#core-architecture)
    - [Hydra Configuration System](#hydra-configuration-system)
        - [Configuration Structure](#configuration-structure)
        - [How Hydra Works in This Codebase](#how-hydra-works-in-this-codebase)
    - [How to use Hydra overrides](#how-to-use-hydra-overrides)
    - [Training Flow](#training-flow)
    - [Inference Flow](#inference-flow)
    - [Key Training Components](#key-training-components)
        - [Model Loading](#model-loading)
        - [LoRA Configuration](#lora-configuration)
        - [Training Configuration](#training-configuration)
    - [Key Inference Components](#key-inference-components)
        - [Model Loading for Inference](#model-loading-for-inference)
        - [LoRA Adapter Integration](#lora-adapter-integration)
        - [Inference Configuration](#inference-configuration)
        - [Output Processing](#output-processing)
        - [Evaluation Metrics](#evaluation-metrics)
- [Running Training or Inference](#running-training-or-inference)
  - [Running Training](#running-training)
  - [Running Inference](#running-inference)
- [Agentic prompting techniques](#agentic-prompting-techniques)
  - [Chain of Stance (CoS)](#chain-of-stance-cos)
  - [Collaborative rOle-infused LLM-based Agents (COLA)](#collaborative-role--infused-llm-based-agents-cola)

---

## Installation

### Prerequisites

- [Anaconda](https://docs.conda.io/projects/conda/en/stable/) installed
- Python 3.10
- Mac or Linux environment recommended


### Setup Instructions

---

1. Setup necessary accounts & hardware:

    You need a Hugging Face account to allow access to gated LLMs and private models
    and datasets. For training, you will also need a Weights & Biases (W&B) account. 
    
    Since both training and inference are compute intensive, 
    you should have access to an Nvidia GPU with a reasonable amount of RAM.
    We have used QMUL's Apocrita HPC cluster (https://docs.hpc.qmul.ac.uk) and
    the Sulis cluster (https://sulis-hpc.github.io) both of which have Ampere family
    GPUs with at least 40Gb of RAM.
    
    As of 21st October 2024, Unsloth only runs on a single GPU.

2. Clone this repository:

    For example, on Apocrita to clone this repository use
    ```shell
    cd /data/scratch/$USER/
    git clone git@github.com:Paul-Richmond/StanceDetector.git
    ```

3. Create environment:

   - On Apocrita:

     For use on Apocrita I have included a shell script to set up a new conda environment
     called `fsi` and install the packages contained within `fsi.txt`.
    
     To run the script, log on to Apocrita and execute
     ```shell
     cd /data/scratch/$USER/StanceDetector
    
     qsub create_fsi_env.sh
     ```
   
   - On Sulis:

     For use on Sulis I have included a script to set up a new virtual environment
     called `fsi` and install the packages contained within `fsi.txt`.
    
     To run the script, log on to Sulis and execute
     ```shell
     cd /StanceDetector
    
     sbatch create_fsi_env.slurm
     ```
     
4. Add API keys using a `.env` file
 
    - Create a new `.env` file in the project root:
        ```bash
        touch .env
        ```
    - Add your API keys

      Open the `.env` file in your preferred editor and add your keys using the format:

        ```
        HUGGINGFACE_API_KEY=your_hf_api_key_here
        WANDB_API_KEY=your_wb_api_key_here
        ```

      > 💡 Do **not** wrap the values in quotes unless required for escaping special characters.

### Optional - Installing Ollama

If you want to use Ollama and a local LLM, you can install it as follows:

1. Visit the [Ollama website](https://ollama.com) and download the installer for your operating system.

2. Follow the installation guide specific to your platform.

3. Once installed, verify the installation by running the following command in your terminal:
   ```bash
   ollama --version
   ```
   This should output the installed version of Ollama. If it fails, you may need to start the Ollama application.

4. Download a model e.g. Llama3.2
   ```bash
   ollama pull llama3.2
   ```
   > Installing Ollama on Apocrita is tricky but doable. The Linux installation script requires use of `sudo` 
   > which is not allowed on Apocrita. Instead, download and move the binary file to `./bin` then insert 
   > `.bin/ollama serve &` to any shell script just before activating the conda environment.


## Model fine-tuning and inference structure

---

This codebase implements a fine-tuning and inference system for open-weight large language models and uses Hydra for 
configuration management, allowing for modular and flexible experiment setup.

The script for fine-tuning is `src/finetune.py` and for inference it is `src/infer.py`.

### Core Architecture

The system follows a modular architecture where:

1. Configuration is managed through Hydra's hierarchical YAML files
2. Model training is orchestrated via the `finetune.py` script
3. Inference is performed through the `infer.py` script
4. Utility functions handle data loading, model instantiation, and optimization

### Hydra Configuration System

Hydra is central to this codebase's design, providing a powerful way to configure and 
parameterize training and inference runs:

#### Configuration Structure

The configuration files are organized in a hierarchical directory structure:
```
configs/
├── default.yaml             # Main entry point for training
├── default_infer.yaml       # Main entry point for inference
├── dataset/
│   └── semeval.yaml         # SemEval dataset configuration
├── inference/
│   └── infer.yaml           # Inference parameters
├── model/
│   └── model.yaml           # Model architecture settings
├── optimizer/
│   └── adamw.yaml           # AdamW optimizer configuration
└── training/
    └── training_args.yaml   # Training parameters
```
- `configs/default.yaml`: The entry point configuration for training that composes other config modules
- `configs/default_infer.yaml`: The entry point configuration for inference
- Specialized configuration modules organized by function:
  - `dataset/`: Dataset configurations like SemEval
  - `model/`: Model architecture settings (using Llama-2-7b-chat)
  - `optimizer/`: Optimizer configurations (AdamW settings)
  - `training/`: Training parameters (epochs, evaluation strategy, etc.)
  - `inference/`: Inference settings for model evaluation

#### How Hydra Works in This Codebase

1. **Configuration Composition**: The `@hydra.main` decorator in both `finetune.py` and `infer.py` loads configurations from multiple YAML files, merging them into a unified `DictConfig` object.

2. **Default Configuration**: `configs/default.yaml` specifies which configuration modules to include by default through its `defaults` list:
   ```yaml
   defaults:
     - optimizer: adamw
     - dataset: semeval
     - model: model
     - training: training_args
     - _self_
   ```

   Similarly, `configs/default_infer.yaml` includes inference-specific configurations:
   ```yaml
   defaults:
     - dataset: semeval
     - inference: infer
     - _self_
   ```

3. **Configuration Resolution**: Hydra resolves references between config files (using the `${...}` syntax) and instantiates objects from these configurations.

4. **Run Directory Management**: Hydra automatically creates output directories for each run with timestamps:
   ```yaml
   hydra:
     run:
       dir: hydra_output/${now:%Y-%m-%d}/${now:%H-%M-%S}
   ```
   
5. **Instantiating objects**: Hydra provides hydra.utils.instantiate() for instantiating objects. The config passed to this function must have a key called `_target_`, with the value of a fully qualified class name.

### How to use Hydra overrides

#### Basic Syntax

The basic syntax for overriding a configuration value is:
```bash
python script.py config_group.config_name.param=value
```
Hydra provides several special operators:

- `~` - Delete a configuration value (e.g.: `~dataset.splits.train=train`)
- `+` - Append to a list
- `++` - Append unique values to a list

#### Common Override Patterns

Override Simple Values
```bash
python3 finetune.py model.model_name=gpt2
```

Override Nested Configurations
```bash
python3 finetune.py training.training_args_cfg.num_train_epochs=5
```

Override Lists/Dictionaries
```bash
python3 finetune.py model.lora_cfg.target_modules='[q_proj,v_proj]'
```

Change Entire Configuration Files
```bash
python3 finetune.py optimizer=adam
```

Multiple Overrides
```bash
python3 finetune.py model.model_name=gpt2 \
training.training_args_cfg.num_train_epochs=5
```


### Training Flow

The training process in `finetune.py` follows these steps:

1. **Environment Setup**: Load API keys for Hugging Face and Weights & Biases
2. **Dataset Loading**: Fetch the dataset from Hugging Face and prepare 
3. **Model Instantiation**: Load the pretrained model using Unsloth's `FastLanguageModel`
4. **LoRA Configuration**: Apply LoRA (Low-Rank Adaptation) parameters for efficient fine-tuning
5. **Processing Pipeline**: Apply chat templates or add EOS tokens to the dataset
6. **Training Execution**: Initialize the trainer with all components and run training
7. **Model Publication**: Push the trained model to Hugging Face Hub

### Inference Flow

The inference process in `infer.py` follows these steps:

1. **Environment Setup**: Load API keys for Hugging Face
2. **Configuration Loading**: Use Hydra to load inference-specific configurations
3. **Dataset Loading**: Load the dataset splits for evaluation
4. **Model Preparation**: Initialize the language model using vLLM for efficient inference
5. **LoRA Adapter Integration**: If a fine-tuned LoRA adapter is specified, it's loaded and applied
6. **Text Generation**: Generate completions for the dataset inputs using the model
7. **Post-Processing**: Convert generated text into structured predictions using parsing rules
8. **Evaluation**: Calculate metrics (F1 score, classification report) for the predictions
9. **Results Storage**: Save all results, predictions, and metrics to a JSON file

### Key Training Components

#### Model Loading

Models are loaded through `FastLanguageModel` from the Unsloth library for training:

```python
model, tokenizer = FastLanguageModel.from_pretrained(model_name, **model_cfg)
```

#### LoRA Configuration

The system applies LoRA for parameter-efficient fine-tuning, targeting specific layers:

```yaml
lora_cfg:
  r: 8
  target_modules:
    - up_proj
    - down_proj
    - gate_proj
    - k_proj
    - q_proj
    - v_proj
    - o_proj
  # other LoRA parameters...
```

#### Training Configuration

Training uses TRL's `SFTTrainer` class, configured with parameters like:

```yaml
num_train_epochs: 3
eval_strategy: steps
eval_steps: 0.111  # Evaluate 9 times throughout training
per_device_train_batch_size: 16
```

### Key Inference Components

#### Model Loading for Inference

The inference script uses vLLM's `LLM` class to load models efficiently:

```python
llm = LLM(model=cfg.inference.base_model_name, enable_lora=enable_lora)
```


#### LoRA Adapter Integration

For fine-tuned models, LoRA adapters are loaded and applied:

```python
if lora_adapter_name is not None:
    enable_lora = True
    lora_request = LoRARequest("sql_adapter", 1, lora_path=lora_adapter_name)
    logger.info(f"Using LoRA adapter: {lora_adapter_name}")
```

#### Inference Configuration

The inference parameters are defined in `configs/inference/infer.yaml`:

```yaml
base_model_name: meta-llama/Llama-2-7b-chat-hf
lora_adapter_name: FundSciImpact/llama-2-7b-chat_ft  # lora_adapter_name is optional

generation_cfg:
  max_tokens: 16
  min_tokens: 1
  temperature: 0.0
```

#### Output Processing

The system includes custom parsing logic to extract stance predictions from generated text:

```yaml
parse_dict:
  disagree:
    indicators:
      - against
    label: against
  agree:
    indicators:
      - support
    label: support
  neutral:
    indicators:
      - none
    label: none
```
This allows customization of the labels. For example, somes papers use labels `["FAVOR", "AGAINST", "NONE"]` instead of
`["support", "against", "none"]`.

#### Evaluation Metrics

The inference pipeline calculates standard classification metrics:

```python
f_score = f1_score(y_true, y_pred, labels=support_labels, average='macro')
class_report = classification_report(y_true=y_true, y_pred=y_pred, zero_division=0.0)
```

### Benefits of the Hydra Approach

This configuration system offers several advantages:

1. **Modularity**: Easily swap components (e.g., datasets, models, optimizers) without code changes
2. **Reproducibility**: Configurations are versioned and explicit
3. **Separation of Concerns**: Clean separation between code and configuration
4. **Flexibility**: Override configurations via command line

The system makes it straightforward to experiment with different models, hyperparameters, and training strategies by simply changing configuration files or providing command-line overrides. This same flexibility extends to the inference process, allowing for easy evaluation of multiple fine-tuned models with various generation parameters.

## Running Training or Inference

### Running Training

To train a model on Apocrita, use the `apocrita_scripts/train.sh` script. You **must** provide a `run_name` variable to
ensure the training run is uniquely identifiable. This can be done using a Hydra override as shown below:

```bash
python3 finetune.py training.training_args_cfg.run_name=<YOUR_RUN_NAME>
```

Replace `<YOUR_RUN_NAME>` with a descriptive name for the training run.

### Running Inference

To perform inference using a trained model, use the `apocrita_scripts/inference.sh` script. Modify the Hydra overrides
within the script as needed, ensuring they match your model and dataset configuration.


## Agentic prompting techniques

### Chain of Stance (CoS)

Chain of Stance (CoS) is a step-by-step approach for structured stance detection using large language models. This method breaks down the complex task of stance detection into a sequential process, enabling more reliable and interpretable results.
The methodology is described in the research paper "Chain of Stance: Stance Detection with Large Language Models" (arXiv:2408.04649).

#### Implementation Details

The script `src/cos.py`, implements a conversational structure where each step of reasoning is explicitly modeled. 
The implementation leverages:
- Structured messaging system for maintaining the chain of reasoning
- Direct integration with VLLM for text analysis
- Careful prompt engineering to guide the model through the stance detection process

An alternative implementation, located in `src/utils/methods.py`, leverages LangGraph to structure the CoS approach.
However, due to the lack of batch processing support in LangChain's VLLM integration, its practicality is limited for
stance classification tasks involving large datasets with thousands of comments.

### Collaborative rOle-infused LLM-based Agents (COLA)

COLA is a multi-agent framework that leverages role-specific LLM agents to collaboratively analyze stance in statements. This approach divides the analytical process among three specialized agents, each bringing a unique perspective to the stance detection task:
#### Agent Roles and Responsibilities
1. **Linguist Agent**: Focuses on the linguistic features of the statement, analyzing semantic structure, word choice, rhetorical devices, and contextual implications to identify stance markers in the text.
2. **Domain Expert Agent**: Evaluates the statement from a subject matter perspective, analyzing factual claims, domain-specific terminology, and technical accuracy to identify topic-specific stance indicators.
3. **Social Media User Agent**: Considers the nuances of social media expressions in the statement such as hashtags and internet slang.

#### Workflow Structure
COLA employs a directed acyclic graph (DAG) workflow where:
1. Each agent first independently analyzes the statement from their specialized perspective
2. Their analyses are then synthesized into argument classifications:
    - Pro arguments (supporting the target)
    - Anti arguments (opposing the target)
    - Neutral arguments (providing balanced perspective)

3. A judge component evaluates all inputs to determine the final stance classification

#### Implementation Details
The COLA framework is implemented using LangChain's DAG structure, allowing for:
- Parallel agent execution
- Structured information flow between components
- Explicit argument categorization
- Consistent prompting through specialized templates for each agent role

This collaborative approach improves stance detection by:
- Breaking down complex analytical tasks into specialized components
- Capturing multiple perspectives on the same content
- Providing explicit reasoning chains for stance judgments
- Reducing individual agent bias through collaborative evaluation
