from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from sklearn.metrics import classification_report, f1_score
from timeit import default_timer as timer

import logging
import os
from dotenv import load_dotenv
import huggingface_hub

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
    return [output.outputs[0].text for output in outputs]


def step_one(step_one_template: str, statements: list) -> list:
    return [[{'role': 'user',
              'content': step_one_template.format(statement=statement)}]
            for statement in statements]


def other_steps(step_template: str, length: int) -> list:
    return [step_template for _ in range(length)]


def step_six(step_six_template: str, targets: list) -> list:
    return [step_six_template.format(target=target) for target in targets]


def add_to_messages(messages, inputs, role='user'):
    # Check len(messages) == len(inputs)
    for idx, input in enumerate(inputs):
        # Might need logic to convert response to string
        messages[idx].append({'role': role, 'content': input})
    return messages


def main():
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

    step_final_template = '''Summarize the stance polarity.'''

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
    class_report_dict = classification_report(y_true=y_true, y_pred=y_pred, zero_division=0.0)

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
