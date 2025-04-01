from datasets import load_dataset, Dataset
from timeit import default_timer as timer
# from langchain_huggingface import HuggingFacePipeline
from langchain_community.llms import VLLM
# from langchain_ollama.chat_models import ChatOllama
from sklearn.metrics import classification_report, f1_score
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import logging
import os
from dotenv import load_dotenv
import huggingface_hub
import torch

from utils.io import save_dict_to_json
from utils.processing import post_process_results
from utils.methods import Cola

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
huggingface_hub.login(token=hf_token)

# Create a logger and set level to INFO.
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

model_cfg = {"attn_implementation": "sdpa",  # "flash_attention_2" doesn't work with torch.compile
             "device_map": "auto",
             "torch_dtype": torch.float16
             }

sys_linguist = '''You are a linguist.'''

sys_expert = '''You are a {role}.'''

sys_user = '''You are a heavy social media user and are very familiar with the way of expression on the Internet.'''

hu_linguist = '''Accurately and concisely explain the linguistic elements in the statement and how these elements affect meaning, including grammatical structure, tense and inflection, virtual speech, rhetorical devices, lexical choices and so on. Do nothing else.
{statement}'''

hu_expert = '''Accurately and concisely explain the key elements contained in the statement, such as characters, events, parties, religions, etc. Also explain their relationship with {target} (if exist). Do nothing else.
{statement}'''

hu_user = '''Analyze the following statement, focusing on the content, hashtags, Internet slang and colloquialisms, emotional tone, implied meaning, and so on. Do nothing else.
{statement}'''

opinion_template = '''Statement: {statement}.
Linguistic analysis: {linguist_analysis}.
The analysis of {role}: {expert_analysis}.
The analysis of a heavy social media user: {user_analysis}.
You think the attitude behind the following statement is {stance} {target}. Identify the top three pieces of evidence from the analyses that best support your opinion and argue for your opinion.
'''

judge_template = '''Determine whether the sentence is in support of or against {target}, or is neutral.
Statement: {statement}
Judge this in relation to the following arguments:
Arguments that the attitude is support: {pro_arguments}
Arguments that the attitude is against: {anti_arguments}
Arguments that the attitude is neutral: {neutral_arguments}
Choose from: 
A: Support
B: Against
C: Neutral
Constraint: Answer with only the option above that is most accurate and nothing else.
'''

target_role_map = {
    "Atheism": "theologian",
    "Climate Change is a Real Concern": "environmental scientist",
    "Feminist Movement": "sociologist",
    "Hillary Clinton": "political scientist",
    "Legalization of Abortion": "sociologist",
    "Donald Trump": "political scientist"
}

parse_dict = {'disagree': {'indicators': ['against'], 'label': 'against'},
              'agree': {'indicators': ['support'], 'label': 'support'},
              'neutral': {'indicators': ['none', 'neutral'], 'label': 'none'},
              'failed': {'label': 'none'}}


def batch_dataset(dataset, batch_size):
    all_data = []
    for examples in dataset.iter(batch_size):
        statements = examples['Tweet']
        targets = examples['Target']
        batch = [{'statement': statement,
                  'target': target,
                  'role': target_role_map[target]} for statement, target in zip(statements, targets)]
        all_data.append(batch)

    return all_data


def main():
    ds = load_dataset('FundSciImpact/semeval-2016', split='train')
    keep_cols = {'Tweet', 'Target', 'Stance'}
    remove_cols = list(set(ds.column_names) - keep_cols)
    ds = ds.remove_columns(remove_cols)
    batches = batch_dataset(ds, 100)

    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", **model_cfg)
    #
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    # tokenizer.padding_side = "left"  # left pad for inference
    # tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    #
    # pipe = pipeline("text-generation",
    #                 model=model,
    #                 tokenizer=tokenizer,
    #                 max_new_tokens=1000,
    #                 top_k=50,
    #                 top_p=0.8,
    #                 temperature=0.8)
    #
    # agent_llm = analysis_llm = judge_llm = HuggingFacePipeline(pipeline=pipe)
    # agent_llm = analysis_llm = ChatOllama(
    #     model="llama3.2",
    #     num_predict=300,
    #     top_k=50,
    #     top_p=0.95,
    #     temperature=0.8,
    # )
    #
    # judge_llm = ChatOllama(
    #     model="llama3.2",
    #     num_predict=10,
    #     top_k=10,
    #     top_p=0.8,
    #     temperature=0.1,
    # )
    agent_llm = analysis_llm = judge_llm = VLLM(model='meta-llama/Llama-3.2-3B-Instruct',
                                                trust_remote_code=True,  # mandatory for hf models

                                                max_new_tokens=300,
                                                top_k=50,
                                                top_p=0.95,
                                                temperature=0.8,
                                                dtype='half',
                                                max_model_len='2048',
                                                )

    cola = Cola(
        sys_linguist=sys_linguist,
        sys_expert=sys_expert,
        sys_user=sys_user,
        hu_linguist=hu_linguist,
        hu_expert=hu_expert,
        hu_user=hu_user,
        opinion_template=opinion_template,
        judge_template=judge_template,
        agent_llm=agent_llm,
        analysis_llm=analysis_llm,
        judge_llm=judge_llm
    )
    graph = cola.initialize_graph()

    all_output = []

    for idx, batch in enumerate(batches):
        logger.info(f"Generating batch {idx}")
        start = timer()
        if len(batch) == 1:
            all_output.append(graph.invoke(batch[0]))
        else:
            all_output.append(graph.batch(batch))
        end = timer()
        minutes_elasped = (end - start) / 60
        logger.info(f"Batch {idx} completed in {minutes_elasped:.1f} minutes")
        save_dict_to_json(data=all_output,
                          directory='outputs',
                          filename=f'coda_test')

    judgements = [output['judgement'] for output in all_output]
    y_pred = post_process_results(judgements, parse_dict)
    y_true = ds['Stance']
    f_score = f1_score(y_true, y_pred, labels=['against', 'support'], average='macro')
    logger.info(f"Macro F1_av on labels 'against', 'support': {f_score:.4f}")
    class_report = classification_report(y_true=y_true, y_pred=y_pred, zero_division=0.0)
    logger.info(f'Classification report: \n {class_report}')
    class_report_dict = classification_report(y_true=y_true, y_pred=y_pred, zero_division=0.0)

    output_dict = {'all_output': all_output,
                   'y_pred': y_pred,
                   'y_true': y_true,
                   'macro_f1': f_score,
                   'classification_report': class_report_dict,
                   }

    save_dict_to_json(data=output_dict,
                      directory='outputs',
                      filename=f'coda_test')


if __name__ == '__main__':
    main()
