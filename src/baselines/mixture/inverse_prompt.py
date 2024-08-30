"""Use GPT-4o to:
1. Rephrase N Units of Text.
2. Attempt to Inverse the Units:
    - Simple Prompt
    - Few-Shot Simple Prompt
    - Simple Prompt with Masked / Kept Tokens
    - Few-Shot Prompt with Masked / Kept Tokens
    
Evaluate:
- ROUGE
- Hamming Distance
"""

import json
import os
import random
import sys
from copy import deepcopy

import pandas as pd
import tiktoken
from datasets import load_from_disk
from openai import OpenAI
from tqdm import tqdm

from utils import get_levenshtein_tags

DIRNAME = "/data1/yubnub/changepoint/s2orc_changepoint/unit_128/train_clean_and_joined"
SAVE_DIRNAME = "./prompting_data"

REPHRASE_PROMPT="""Rephrase the following passage: {}

Only output the rephrased-passage, do not include any other details.
"""

INVERSE_PROMPT="""The following passage is a LLM rephrase of a human-written passage: {}

Please write the original passage, as you believe it was written by the human.

Only output the rephrased-passage, do not include any other details.
"""

INVERSE_PROMPT_WITH_TOKENS_TO_KEEP="""The following passage is a LLM rephrase of a human-written passage: {}

Tokens to keep when re-writing the passage: {}

Please write the original passage, as you believe it was written by the human, while keeping the tokens specified.

Only output the rephrased-passage, do not include any other details.
"""

FEWSHOT_HEADER="""Here are some examples of original passages and their rephrases:

"""

TOKENIZER = tiktoken.encoding_for_model("gpt-4o")
DO_PROMPT = True

def file_exists(
    filename: str
) -> bool:
    os.makedirs(SAVE_DIRNAME, exist_ok=True)
    path = os.path.join(SAVE_DIRNAME, filename)
    return os.path.exists(path)

def query(
    client: OpenAI, 
    prompt: str,
) -> str:
    if not DO_PROMPT:
        response = "Debug."
        return response
    
    completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            )
    response = completion.choices[0].message.content
    return response

def load_s2orc_dataset() -> pd.DataFrame:
    N = 100
    records = []
    dataset = load_from_disk(DIRNAME)
    for i in range(N):
        example = dataset[i]
        samples = random.sample(example["units"], k=11)
        record = {
            "id": example["id"],
            "fewshot_units": samples[:-1],
            "unit": samples[-1],
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    return df

def create_rephrase_data(
    original_data: list[dict], 
    client: OpenAI, 
) -> pd.DataFrame:
    original_data = deepcopy(original_data)
    for i, record in enumerate(tqdm(original_data)):
        original_data[i]["rephrase"] = query(client, REPHRASE_PROMPT.format(record["unit"]))
        original_data[i]["rephrase_prompt"] = REPHRASE_PROMPT.format(record["unit"])

        original_data[i]["rephrase_fewshot"] = [None] * len(record["fewshot_units"])
        original_data[i]["fewshot_rephrase_prompts"] = [None] * len(record["fewshot_units"])
        for j, unit in enumerate(record["fewshot_units"]):
            original_data[i]["rephrase_fewshot"][j] = query(client, REPHRASE_PROMPT.format(unit))
            original_data[i]["fewshot_rephrase_prompts"][j] = REPHRASE_PROMPT.format(unit)
        
    df = pd.DataFrame(original_data)
    return df

def build_inverse_prompt(
    rephrase: str,
    header: str = None,
    unit: str = None,
) -> str:
    out = ""
    if header:
        out += header
    
    if unit:
        tokens_to_keep = get_tokens_to_keep(rephrase, unit)
        out += INVERSE_PROMPT_WITH_TOKENS_TO_KEEP.format(rephrase, tokens_to_keep)
    else:
        out += INVERSE_PROMPT.format(rephrase)

    return out

def build_fewshot_header(
    units: list[str],
    rephrases: list[str],
) -> str:
    header = FEWSHOT_HEADER
    for unit, rephrase in zip(units, rephrases):
        header += f"Rephrased: {rephrase}\nOriginal: {unit}\n\n"
    return header

def get_tokens_to_keep(
    rephrase: str,
    original: str,
):
    def tok_fn(text):
        encodings = TOKENIZER.encode(text, allowed_special="all")
        encodings = [[x] for x in encodings]
        decoded = [TOKENIZER.decode(x) for x in encodings]
        return decoded
    
    tags = get_levenshtein_tags(rephrase, original, tok_fn)
    tokens = tok_fn(rephrase)
    tokens_to_keep = [tok for tok, tag in zip(tokens, tags) if tag == "KEEP"]
    return tokens_to_keep

def prompt_inverse(
    rephrase_data: list[dict], 
    client: OpenAI,
    fewshot_N: int = 5,
    fewshot_mode: str = None,
    with_tokens_to_keep: bool = False,
) -> pd.DataFrame:
    assert fewshot_mode in [None, "same_paper", "random"]
    rephrase_data = deepcopy(rephrase_data)

    if fewshot_mode == "random":
        # Sample `fewshot_N` random rephrases / originals and use them
        # as the few-shot examples for every record:
        original_units = [rephrase_data["fewshot_units"] for rephrase_data in rephrase_data]
        original_units = [item for sublist in original_units for item in sublist]
        original_units = random.sample(original_units, k=fewshot_N)

        rephrased_units = [rephrase_data["rephrase_fewshot"] for rephrase_data in rephrase_data]
        rephrased_units = [item for sublist in rephrased_units for item in sublist]
        rephrased_units = random.sample(rephrased_units, k=fewshot_N)
        
        fewshot_header = build_fewshot_header(original_units, rephrased_units)

    for i, record in enumerate(tqdm(rephrase_data)):
        unit = record["unit"] if with_tokens_to_keep else None
        if fewshot_mode == "random":
            prompt = build_inverse_prompt(
                record["rephrase"], 
                unit=unit,
                header=fewshot_header,
            )
            inverse = query(client, prompt)
        elif fewshot_mode == "same_paper":
            # We use a few-shot prompt composed of units from the same paper:
            fewshot_header = build_fewshot_header(
                record["fewshot_units"][:fewshot_N], 
                record["rephrase_fewshot"][:fewshot_N]
            )
            prompt = build_inverse_prompt(
                record["rephrase"], 
                unit=unit,
                header=fewshot_header, 
            )
            inverse = query(client, prompt)
        else:
            prompt = build_inverse_prompt(
                record["rephrase"], 
                unit=unit,
            ) 
            inverse = query(client, prompt)
        
        rephrase_data[i]["inverse"] = inverse
        rephrase_data[i]["inverse_prompt"] = prompt
    
    df = pd.DataFrame(rephrase_data)
    return df

def load_or_create_data(
    filename: str,
    load_fn: callable,
    debug: bool = False,
    save: bool = True,
    **kwargs,
):
    if not file_exists(filename):
        data = load_fn(**kwargs)
        if save and not debug:
            data.to_json(os.path.join(SAVE_DIRNAME, filename), orient="records", lines=True)
    else:
        data = pd.read_json(os.path.join(SAVE_DIRNAME, filename), lines=True)
    
    data = data.to_dict(orient="records")
    return data

if __name__ == "__main__":
    random.seed(43)
    debug = False
    client = OpenAI()
    
    # rephrase: str = "Parental actions like directing attention, clarifying information, expanding on their children's statements, offering feedback, and asking wh-questions contribute to the quality of storybook reading experiences (Chang and Luo 2020; Neuman 1996; Ninio and Bruner 1978). Additionally, when parents engage in print referencing behaviors, such as discussing the form and features of the print in the storybook, they enhance the quality of parent-child interactions during storybook reading (Justice and Ezell 2000)."
    # original: str = "Parent behaviors such as directing attention, clarifying information, expanding on their children's utterances, providing feedback, and asking wh-questions are among behaviors that contribute to high-quality storybook reading experiences (Chang and Luo 2020;Neuman 1996;Ninio and Bruner 1978). Parents' use of print referencing behaviors, such as talking about the print form and characteristics in the storybook, also promote the quality of parent-child interactions during storybook reading (Justice and Ezell 2000)."
    # tokens_to_keep = get_tokens_to_keep(rephrase, original)
    # test_prompt = INVERSE_PROMPT_WITH_TOKENS_TO_KEEP.format(rephrase, tokens_to_keep)
    # print(test_prompt)
    
    print("Loading or Creating Data...")
    original_data = load_or_create_data(
        "original_units.jsonl", 
        load_s2orc_dataset,
        debug=debug,
    )
    
    print("Prompting for Rephrases...")
    rephrase_data = load_or_create_data(
        "rephrases.jsonl", 
        create_rephrase_data, 
        debug=debug,
        original_data=original_data,
        client=client,
    )
    
    with_tokens_to_keep = [False, True]
    for tokens_to_keep in with_tokens_to_keep:
        print("Prompting for Inverses, KEEP={}...".format(tokens_to_keep))
        inverse_data = load_or_create_data(
            f"inverse_prompts_keep={tokens_to_keep}.jsonl", 
            prompt_inverse, 
            debug=debug,
            rephrase_data=rephrase_data,
            client=client,
            with_tokens_to_keep=tokens_to_keep,
        )

    modes = ["random", "same_paper"]
    Ns = [1, 5]
    for tokens_to_keep in with_tokens_to_keep:
        for mode in modes:
            for N in Ns:
                print("Prompting for Few-Shot Inverses, mode={}, N={}, KEEP={}".format(mode, N, tokens_to_keep))
                inverse_data_fewshot = load_or_create_data(
                    f"inverse_prompts_fewshot_{mode}_{N}_keep={tokens_to_keep}.jsonl", 
                    prompt_inverse, 
                    debug,
                    rephrase_data=rephrase_data,
                    client=client,
                    fewshot_N=N,
                    fewshot_mode=mode,
                    with_tokens_to_keep=tokens_to_keep,
                )