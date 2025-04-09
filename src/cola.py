"""
cola.py - This script implements the methodology described in the paper 
"Chain of Stance: Stance Detection with Large Language Models" (arXiv:2408.04649)
and applies it to the SemEval-2016 dataset.

This is an initial working version.
The `langgraph` code has been refactored and moved to src/methods.py in subsequent iterations.
"""

import logging
import os
import json
import torch
from typing import Annotated, TypedDict

from dotenv import load_dotenv
import huggingface_hub

from langchain_community.llms import VLLM
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from datasets import load_dataset

from utils.io import save_dict_to_json

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
huggingface_hub.login(token=hf_token)

# Create a logger and set level to INFO.
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

llm = VLLM(model='meta-llama/Llama-3.2-3B-Instruct',
           trust_remote_code=True,  # mandatory for hf models
           max_new_tokens=100,
           top_k=10,
           top_p=0.95,
           temperature=0.0,
           dtype='half',
           # vllm_kwargs=model_cfg
           )
# llm = ChatOllama(model="llama3.2",
#                  num_predict=100,
#                  top_k=10,
#                  top_p=0.95,
#                  temperature=0.0,
#                  )

# assign experts for target
# taken from https://github.com/tsinghua-fib-lab/COLA
target_role_map = {
    "Atheism": "theologian",
    "Climate Change is a Real Concern": "environmental scientist",
    "Feminist Movement": "sociologist",
    "Hillary Clinton": "political scientist",
    "Legalization of Abortion": "sociologist",
    "Donald Trump": "political scientist"
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

sys_message_linguist = SystemMessagePromptTemplate.from_template(sys_linguist)
sys_message_expert = SystemMessagePromptTemplate.from_template(sys_expert)
sys_message_user = SystemMessagePromptTemplate.from_template(sys_user)
human_message_linguist = HumanMessagePromptTemplate.from_template(hu_linguist)
human_message_expert = HumanMessagePromptTemplate.from_template(hu_expert)
human_message_user = HumanMessagePromptTemplate.from_template(hu_user)
message_opinion = HumanMessagePromptTemplate.from_template(opinion_template)
message_judge = HumanMessagePromptTemplate.from_template(judge_template)


class State(TypedDict):
    # to track conversation history
    messages: Annotated[list, add_messages]
    # input
    target: str
    statement: str
    # output
    linguist_analysis: str
    expert_analysis: str
    user_analysis: str
    pro_arguments: str
    anti_arguments: str
    neutral_arguments: str
    judgement: str


class Input(TypedDict):
    target: str
    statement: str


class Output(TypedDict):
    linguist_analysis: str
    expert_analysis: str
    user_analysis: str
    pro_arguments: str
    anti_arguments: str
    neutral_arguments: str
    judgement: str


def linguist_node(state: State) -> State:
    sys_message = sys_message_linguist.format()
    human_message = human_message_linguist.format(statement=state['statement'])
    messages = [sys_message, *state['messages'], human_message]
    res = llm.invoke(messages)
    return {'linguist_analysis': res.content, 'messages': [human_message, res]}


def expert_node(state: State) -> State:
    role = target_role_map[state['target']]
    sys_message = sys_message_expert.format(role=role)
    human_message = human_message_expert.format(statement=state['statement'],
                                                target=state['target'])
    messages = [sys_message, *state['messages'], human_message]
    res = llm.invoke(messages)
    return {'expert_analysis': res.content, 'messages': [human_message, res]}


def user_node(state: State) -> State:
    sys_message = sys_message_user.format()
    human_message = human_message_user.format(statement=state['statement'])
    messages = [sys_message, *state['messages'], human_message]
    res = llm.invoke(messages)
    return {'user_analysis': res.content, 'messages': [human_message, res]}


# def agent(state: State, sys_msg_template: SystemMessagePromptTemplate, output_key: str) -> State:
#     sys_message = sys_msg_template.format(target=state['target'],
#                                           role=state['role'])
#     user_message = HumanMessage(state['statement'])
#     messages = [sys_message, *state['messages'], user_message]
#     res = llm.invoke(messages)
#     return {output_key: res.content, 'messages': [user_message, res]}

def analysis(state: State) -> State:
    role = target_role_map[state['target']]
    arguments = {}
    stance_to_argument = {'support': 'pro_arguments', 'against': 'anti_arguments', 'none': 'neutral_arguments'}
    for stance, argument in stance_to_argument.items():
        user_message = message_opinion.format(stance=stance,
                                              target=state['target'],
                                              statement=state['statement'],
                                              role=role,
                                              linguist_analysis=state['linguist_analysis'],
                                              expert_analysis=state['expert_analysis'],
                                              user_analysis=state['user_analysis'])
        messages = [user_message, *state['messages']]
        res = llm.invoke(messages)
        arguments.update({argument: res.content})

    arguments.update({'messages': list(arguments.values())})
    return arguments


def judge(state: State) -> State:
    user_message = message_judge.format(target=state['target'],
                                        statement=state['statement'],
                                        pro_arguments=state['pro_arguments'],
                                        anti_arguments=state['anti_arguments'],
                                        neutral_arguments=state['neutral_arguments'])
    messages = [user_message, *state['messages']]
    res = llm.invoke(messages)
    return {'judgement': res.content, 'messages': res}


builder = StateGraph(State, input=Input, output=Output)
builder.add_node('Linguist', linguist_node)
builder.add_node('Expert', expert_node)
builder.add_node('User', user_node)
builder.add_node('Analysis', analysis)
builder.add_node('Judgement', judge)
builder.add_edge(START, 'Linguist')
builder.add_edge('Linguist', 'Expert')
builder.add_edge('Expert', 'User')
builder.add_edge('User', 'Analysis')
builder.add_edge('Analysis', 'Judgement')
builder.add_edge('Judgement', END)
graph = builder.compile()


def batch_dataset(dataset, batch_size):
    all_data = []
    for examples in dataset.iter(batch_size):
        statements = examples['Tweet']
        targets = examples['Target']
        batch = [{'statement': statement, 
                  'target': target} for statement, target in zip(statements, targets)]
        all_data.append(batch)

    return all_data


def get_gpu_memory_usage():
    """
    Get the current GPU memory usage for all available GPUs.
    Returns a list of dictionaries containing memory stats for each GPU.
    """
    if not torch.cuda.is_available():
        return None

    gpu_stats = []
    for i in range(torch.cuda.device_count()):
        gpu_stats.append({
            'device_id': i,
            'device_name': torch.cuda.get_device_name(i),
            'total_memory': torch.cuda.get_device_properties(i).total_memory / (1024 ** 2),  # Convert to MB
            'allocated_memory': torch.cuda.memory_allocated(i) / (1024 ** 2),
            'cached_memory': torch.cuda.memory_reserved(i) / (1024 ** 2)
        })
    return gpu_stats


def main():
    ds = load_dataset('FundSciImpact/semeval-2016', split='train')
    keep_cols = {'Tweet', 'Target', 'Stance'}
    remove_cols = list(set(ds.column_names) - keep_cols)
    ds = ds.remove_columns(remove_cols)
    batches = batch_dataset(ds, 4)

    all_output = []
    all_gpu_stats = []

    for idx, batch in enumerate(batches):
        all_output.extend(graph.batch(batch))

        gpu_stats = get_gpu_memory_usage()
        if gpu_stats:
            logger.info(f"\nBatch {idx} GPU Usage:")
            for gpu in gpu_stats:
                logger.info(f"\nGPU {gpu['device_id']} - {gpu['device_name']}")
                logger.info(f"Total Memory: {gpu['total_memory']:.2f} MB")
                logger.info(f"Allocated Memory: {gpu['allocated_memory']:.2f} MB")
                logger.info(f"Cached Memory: {gpu['cached_memory']:.2f} MB")
                logger.info(f"Utilization: {(gpu['allocated_memory'] / gpu['total_memory'] * 100):.2f}%")

        output_dict = {'batch': idx,
                       'gpu_stats': all_gpu_stats.append(gpu_stats),
                       'output': all_output}

        if idx % 12 == 0:
            save_dict_to_json(data=output_dict,
                              directory='outputs',
                              filename=f'vllm_batch_{idx}')


if __name__ == '__main__':
    main()
