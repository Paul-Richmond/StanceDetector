import logging
import os

from dotenv import load_dotenv
import huggingface_hub
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import classification_report, f1_score

from langchain_community.llms import VLLM
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ChatMessage, messages_to_dict
from langchain.prompts import (PromptTemplate,
                               ChatPromptTemplate,
                               HumanMessagePromptTemplate,
                               MessagesPlaceholder)
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.chat_history import (BaseChatMessageHistory,
                                         InMemoryChatMessageHistory)

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

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def create_initial_messages(dataset):
    targets = dataset['Target']
    tweets = dataset['Tweet']
    human_template = "What is the stance towards the target {target} in the following tweet: {tweet}."
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    return [{"messages": [human_message_prompt.format(target=target, tweet=tweet)]} for target, tweet in
            zip(targets, tweets)]


def create_followup_messages(dataset):
    size = dataset.num_rows
    return [{"messages": [HumanMessage(content="Explain how you arrived at your answer")]} for _ in range(size)]


def create_batch_configs(size):
    return [{"configurable": {"session_id": str(i)}} for i in range(size)]


@hydra.main(version_base=None, config_path="../configs", config_name="default_infer")
def main(cfg: DictConfig) -> None:
    # Get variables from cfg
    # dataset_col = cfg.dataset.col_to_select
    # parse_dict = cfg.inference.parse_dict
    # support_labels = cfg.inference.f1_support_labels
    # labels = cfg.inference.dataset_labels_column
    # stance_target_column = cfg.inference.stance_target_column
    #
    ds = load_dataset_splits(cfg.dataset)  # expect DatasetDict object with a single key
    ds_split_names = list(ds)
    ds = ds[list(ds)[0]]  # extract the single Dataset object from the DatasetDict object

    sys_msg = SystemMessage("""You are a helpful, respectful, and honest assistant for stance detection for a given target. Stance detection is the process of determining whether the author of a tweet is in support of or against a given target. The target may not always be explicitly mentioned in the text, and the tweet’s stance can be conveyed implicitly through subtext, regional and cultural references, or other implicit meanings. 
    The possible stances are:
      support: The tweet has a positive or supportive attitude towards the target, either explicitly or implicitly.
      against: The tweet opposes or criticizes the target, either explicitly or implicitly.
      none: The tweet is neutral or doesn’t have a stance towards the target.""")

    prompt = ChatPromptTemplate.from_messages([sys_msg, MessagesPlaceholder(variable_name="messages"), ])
    #
    # if 'generation_cfg' in cfg.inference.keys():
    #     generation_cfg = OmegaConf.to_container(cfg.inference.generation_cfg, resolve=True)
    #
    # lora_adapter_name = cfg.inference.get('lora_adapter_name', None)
    # model_name = lora_adapter_name if lora_adapter_name is not None else cfg.inference.base_model_name
    # if 'model_cfg' in cfg.inference.keys():
    model_name = cfg.inference.base_model_name
    llm = VLLM(model=model_name,
               trust_remote_code=True,  # mandatory for hf models
               max_new_tokens=128,
               top_k=10,
               top_p=0.95,
               temperature=0.8,
               dtype='half',
               # vllm_kwargs=model_cfg
               )
    chain = prompt | llm

    init_messages = create_initial_messages(ds)
    follow_messages = create_followup_messages(ds)
    batch_configs = create_batch_configs(ds.num_rows)

    chain_with_message_history = RunnableWithMessageHistory(chain,
                                                            get_session_history,
                                                            input_messages_key="messages")
    history_list = []
    for init_message, batch_config in zip(init_messages, batch_configs):
        response = chain_with_message_history.invoke(init_message, batch_config)

    for follow_message, batch_config in zip(follow_messages, batch_configs):
        response = chain_with_message_history.invoke(follow_messages, batch_config)
        session_id = batch_config['configurable']['session_id']
        session_history = chain_with_message_history.get_session_history(session_id)
        history_list.append(messages_to_dict(session_history.messages))

    # Prepare output_dict and save it to disk
    output_dict = {'model': model_name,
                   'dataset': {'name': cfg.dataset.name, 'splits': ds_split_names[0]},
                   'chat_histories': history_list,
                   }
    try:
        save_dict_to_json(data=output_dict,
                          directory=cfg.inference.output_dir,
                          filename=cfg.inference.output_fname)
    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    main()
