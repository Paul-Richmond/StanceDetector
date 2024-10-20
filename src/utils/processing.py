from tqdm import tqdm


def apply_chat_template(examples, tokenizer, col_to_tokenize, add_generation_prompt):
    messages = examples[col_to_tokenize]
    text = [tokenizer.apply_chat_template(message,
                                          tokenize=False,
                                          add_generation_prompt=add_generation_prompt)
            for message in messages]
    return {"messages": text}


def add_eos_token(examples, tokenizer, col_to_tokenize, eval_set=False):
    prompts = examples[col_to_tokenize]
    EOS_TOKEN = tokenizer.eos_token
    texts = []
    if not eval_set:
        # it's a training dataset, so we add eos_token
        texts.extend([prompt + EOS_TOKEN for prompt in prompts])
    else:
        # it's a test/validation dataset, so we expect to generate after the prompt and don't add eos_token
        texts.extend([prompt for prompt in prompts])
    return {"text": texts, }


def post_process_results(results, parse_dict):
    """
    This function post-processes the results from a large language model to label text.

    Args:
        results (list): A list of strings, each string is a result from language model output.
        parse_dict (dict): A dictionary containing indicators and labels

    Returns:
        list: A list of strings, each string is a classification label.
    """

    # A list to store the prediction labels
    y_pred = []

    # Words or phrases that indicate each stance category
    disagree_indicators = parse_dict['disagree']['indicators']
    agree_indicators = parse_dict['agree']['indicators']
    neutral_indicators = parse_dict['neutral']['indicators']

    # Dictionary to map 'disagree', 'agree', 'neutral' and 'failed' to a classification label
    label_dict = {k: parse_dict[k]['label'] for k in parse_dict.keys()}

    # Iterate over LLM generated text in the results
    for result in tqdm(results, desc='Parsing predictions'):
        # Normalize the result to lower case and remove leading/trailing white spaces
        normalized_result = str(result).strip().lower()

        if any(indicator in normalized_result for indicator in disagree_indicators):
            # If the result is also found in agree_indicators or neutral_indicators, label it as 'failed'
            if (any(indicator in normalized_result for indicator in agree_indicators)
                    or any(indicator in normalized_result for indicator in neutral_indicators)):
                y_pred.append(label_dict['failed'])
            else:
                y_pred.append(label_dict['disagree'])
        elif any(indicator in normalized_result for indicator in neutral_indicators):
            # If the result is also found in disagree_indicators or agree_indicators, label it as 'failed'
            if (any(indicator in normalized_result for indicator in disagree_indicators)
                    or any(indicator in normalized_result for indicator in agree_indicators)):
                y_pred.append(label_dict['failed'])
            else:
                y_pred.append(label_dict['neutral'])
        elif any(indicator in normalized_result for indicator in agree_indicators):
            # If the result is also found in disagree_indicators or neutral_indicators, label it as 'neutral'
            if (any(indicator in normalized_result for indicator in disagree_indicators)
                    or any(indicator in normalized_result for indicator in neutral_indicators)):
                y_pred.append(label_dict['failed'])
            else:
                y_pred.append(label_dict['agree'])
        else:
            # If no specific stance label is found, label it as 'failed'
            y_pred.append(label_dict['failed'])

    return y_pred


def format_data(examples, eval_set=False):
    tweets = examples["Tweet"]
    targets = examples["Target"]
    labels = examples["Stance"]

    texts = []
    for tweet, target, label in zip(tweets, targets, labels):
        user_content = f"Tweet: {tweet} \n Stance towards the target {target}:"
        if not eval_set:
            # training dataset so add labels
            text = user_content + " " + label
        else:
            text = user_content + " "
        texts.append(text)
    return {'text': texts}
