
import functools
import operator
import os
import sys
from collections import Counter

import tiktoken
from datasets import load_from_disk
from tqdm import tqdm

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-instruct"):
    """Returns the number of tokens used by a list of messages.
       Taken and modified from OpenAI's website.
    """
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def num_tokens_for_example(example, model="gpt-3.5-turbo-instruct"):
    PROMPT = "Passage Before: {}\nPassage After\n{}Write a passage that connects the two.\n"
    num_tokens = 0
    for indices in example["changepoint_indices"]:
        passage_before = example["units"][indices-1]
        passage_after = example["units"][indices+1]
        prompt = PROMPT.format(passage_before, passage_after)
        
        message = [
            {"role": "system", "content": "You are a helpful assistant that helps connect two passages."},
            {"role": "user", "content": prompt},
        ]
        num_tokens += num_tokens_from_messages(message, model=model)

    # assume that the output will be roughly the lenght of a single passage
    return {
        "input": num_tokens,
        "output": num_tokens // 2,
    }

def estimate_cost(num_tokens, model="gpt-3.5-turbo-instruct"):
    assert model in ["gpt-3.5-turbo-instruct", "gpt-4o"]
    # https://openai.com/api/pricing/
    cost_per_million = {
        "input": {
            "gpt-3.5-turbo-instruct": 1.5,
            "gpt-4o": 5.00,
        },
        "output": {
            "gpt-3.5-turbo-instruct": 2.0,
            "gpt-4o": 15.00,
        },
    }
    
    cost = (num_tokens["input"] / 1_000_000) * cost_per_million["input"][model]
    cost += (num_tokens["output"] / 1_000_000) * cost_per_million["output"][model]
    return cost

def main():
    dataset = load_from_disk("/data1/yubnub/changepoint/s2orc_changepoint/unit/validation")
    
    model_name = "gpt-3.5-turbo-instruct"
    num_tokens_per_example = [num_tokens_for_example(example, model=model_name) for example in tqdm(dataset)]
    num_tokens = dict(functools.reduce(operator.add, map(Counter, num_tokens_per_example)))
    cost = estimate_cost(num_tokens, model=model_name)
    print(f"{model_name}: ${cost:.2f}")

    model_name = "gpt-4o"
    num_tokens_per_example = [num_tokens_for_example(example, model=model_name) for example in tqdm(dataset)]
    num_tokens = dict(functools.reduce(operator.add, map(Counter, num_tokens_per_example)))
    cost = estimate_cost(num_tokens, model=model_name)
    print(f"{model_name}: ${cost:.2f}")

    return 0

if __name__ == "__main__":
    sys.exit(main())