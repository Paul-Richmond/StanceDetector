"""
cos.py - Chain of Stance

This script is an implementation of the paper `Chain of Stance: Stance Detection with Large Language Models`, arXiv:2408.04649.
It uses the vLLM package to interact with large language models for stance detection tasks.
The script follows a multi-step process to analyze statements, detect their stances towards specific targets, 
and evaluate the contextual polarity using guided decoding and sampling parameters.
The main functionality includes loading a dataset, processing statements through defined steps of analysis 
with the help of a language model, and calculating evaluation metrics like F1 score and classification report.
"""

import logging
import os
from timeit import default_timer as timer

import huggingface_hub
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.metrics import classification_report, f1_score
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from utils.io import save_dict_to_json
from utils.processing import post_process_results

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
huggingface_hub.login(token=hf_token)

# Create a logger and set level to INFO.
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def get_vllm_responses(outputs):
    """
    Extracts the text responses from the outputs of the language model.

    Args:
        outputs (list): A list of model output objects containing text responses.

    Returns:
        list: A list of strings, where each string is the model's response.
    """
    return [output.outputs[0].text for output in outputs]


def step_one(step_one_template: str, statements: list) -> list:
    """
    Constructs the input messages for the first step of stance detection.

    Args:
        step_one_template (str): A template defining the task for step one.
        statements (list): A list of textual statements to analyze.

    Returns:
        list: A nested list of dictionaries, where each dictionary represents
              a user message with the content formatted by the template.
    """
    return [[{'role': 'user',
              'content': step_one_template.format(statement=statement)}]
            for statement in statements]


def other_steps(step_template: str, length: int) -> list:
    """
    Generates a list of repeated templates for subsequent processing steps.

    Args:
        step_template (str): The template for the current step in processing.
        length (int): The number of times to repeat the template.

    Returns:
        list: A list containing the specified template repeated 'length' times.
    """
    return [step_template for _ in range(length)]


def step_six(step_six_template: str, targets: list) -> list:
    """
    Constructs the input messages for step six, which determines stance polarity.

    Args:
        step_six_template (str): A template string for step six.
        targets (list): A list of targets for stance determination.

    Returns:
        list: A list of strings, each formatted with the respective target.
    """
    return [step_six_template.format(target=target) for target in targets]


def add_to_messages(messages, inputs, role='user'):
    """
    Appends new inputs to the existing message chains.

    Args:
        messages (list): A list of message chains, where each chain is a list of dictionaries.
        inputs (list): A list of new messages to be appended as responses.
        role (str, optional): The role of the message sender (e.g., 'user', 'assistant'). Defaults to 'user'.
    
    Returns:
        list: The updated list of message chains with appended inputs.
    """
    # Check len(messages) == len(inputs)
    for idx, input in enumerate(inputs):
        # Might need logic to convert response to string
        messages[idx].append({'role': role, 'content': input})
    return messages


def main():
    """
    The main function orchestrates the stance detection process:
    - Loads the test dataset containing statements, targets, and ground truth stances.
    - Processes data through multiple steps, interacting with the LLM.
    - Evaluates the macro F1 score and classification metrics.
    - Saves the final output, including generated messages and metrics, to a JSON file.
    """
    # Load the dataset and extract relevant fields.
    ds = load_dataset('FundSciImpact/semeval-2016', split='test')
    statements = ds['Tweet']
    targets = ds['Target']
    y_true = ds['Stance']
    length = ds.num_rows

    parse_dict = {'disagree': {'indicators': ['against'], 'label': 'against'},
                  'agree': {'indicators': ['support'], 'label': 'support'},
                  'neutral': {'indicators': ['none', 'neutral'], 'label': 'none'},
                  'failed': {'label': 'none'}}

    guided_decoding_params = GuidedDecodingParams(choice=['support', 'against', 'none'])

    sampling_params = SamplingParams(max_tokens=300,
                                     top_k=50,
                                     top_p=0.95,
                                     temperature=0.8)

    sampling_params_final = SamplingParams(max_tokens=10,
                                           top_k=50,
                                           top_p=0.95,
                                           temperature=0.2,
                                           guided_decoding=guided_decoding_params)

    llm = LLM(model='meta-llama/Llama-2-7b-chat-hf',
              trust_remote_code=True,  # mandatory for hf models
              dtype='half',
              max_model_len=4096,
              )

    step_one_template = '''Given text "{statement}", understand the contextual information of the text, which includes the topic of the text, the identity of the author, the target audience, and the relevant socio-cultural background.'''

    step_two_template = '''What are the core viewpoints and main intentions to be expressed in the text?'''

    step_three_template = '''Analyze the language expression and emotional inclination of the text. Identify the emotional words and rhetorical devices used within the text, and analyze the tone adopted by the author (e.g., affirmative, negative, interrogative, exclamatory, etc.). Based on this analysis, describe the author's emotional attitude.'''

    step_four_template = '''Compare similarities and contrasts between text and various possible stances (support, against, none), based on the above-mentioned information.'''

    step_five_template = '''Conduct logical inference based on the context and other relevant information to confirm the consistency and rationality of the stance.'''

    step_six_template = '''Based on all previous information, determine the stance polarity towards {target}.'''

    # As using guided_decoding_params we shouldn't need to summarize the stance
    # step_final_template = '''Summarize the stance polarity.'''

    start = timer()
    messages = step_one(step_one_template, statements)
    outputs = llm.chat(messages, sampling_params, use_tqdm=True)
    responses = get_vllm_responses(outputs)
    messages = add_to_messages(messages, responses, role='assistant')

    messages = add_to_messages(messages, other_steps(step_two_template, length), role='user')
    outputs = llm.chat(messages, sampling_params)
    responses = get_vllm_responses(outputs)
    messages = add_to_messages(messages, responses, role='assistant')

    messages = add_to_messages(messages, other_steps(step_three_template, length), role='user')
    outputs = llm.chat(messages, sampling_params)
    responses = get_vllm_responses(outputs)
    messages = add_to_messages(messages, responses, role='assistant')

    messages = add_to_messages(messages, other_steps(step_four_template, length), role='user')
    outputs = llm.chat(messages, sampling_params)
    responses = get_vllm_responses(outputs)
    messages = add_to_messages(messages, responses, role='assistant')

    messages = add_to_messages(messages, other_steps(step_five_template, length), role='user')
    outputs = llm.chat(messages, sampling_params)
    responses = get_vllm_responses(outputs)
    messages = add_to_messages(messages, responses, role='assistant')

    messages = add_to_messages(messages, step_six(step_six_template, targets), role='user')
    outputs = llm.chat(messages, sampling_params_final)
    responses = get_vllm_responses(outputs)
    messages = add_to_messages(messages, responses, role='assistant')

    # As using guided_decoding_params we shouldn't need to summarize the stance with step_final_template
    # messages = add_to_messages(messages, other_steps(step_final_template, length), role='user')
    # outputs = llm.chat(messages, sampling_params_final)
    # responses = get_vllm_responses(outputs)
    # messages = add_to_messages(messages, responses, role='assistant')

    end = timer()
    minutes_elasped = (end - start) / 60
    logger.info(f"Completed generation for {length} entries in {minutes_elasped:.1f} minutes")

    y_pred = post_process_results(responses, parse_dict)
    f_score = f1_score(y_true, y_pred, labels=['against', 'support'], average='macro')
    logger.info(f"Macro F1_av on labels 'against', 'support': {f_score:.4f}")
    class_report = classification_report(y_true=y_true, y_pred=y_pred, zero_division=0.0)
    logger.info(f'Classification report: \n {class_report}')
    class_report_dict = classification_report(y_true=y_true, y_pred=y_pred, zero_division=0.0, output_dict=True)

    output_dict = {'all_output': messages,
                   'y_pred': y_pred,
                   'y_true': y_true,
                   'macro_f1': f_score,
                   'classification_report': class_report_dict,
                   }

    save_dict_to_json(data=output_dict,
                      directory='outputs',
                      filename=f'cos_test')


if __name__ == '__main__':
    main()
