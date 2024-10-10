
import json
import os
import sys
sys.path.append("../changepoint")

import pandas as pd

from prompts import *

def build_inverse_prompt_gpt4(
    generation: str,
    examples: list[str] = None
) -> str:
    if examples is not None:
        seperator = "\n-----\n"
        header = "Here are examples of the original author:\n"
        for example in examples:
            header += f"Example: {example}{seperator}"
    else:
        header = ""

    base_instruction = "The following passage is a mix of human and machine text, recover the original human text:"
    instruction = base_instruction
    prompt = f"{header}{instruction} {generation}"
    return prompt


"""
Prompts to Support:
1. Rephrase
2. Inversion
3. Inversion (In-Context)
"""

def write(records, output_file):
    with open(output_file, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

file_path = "/data1/yubnub/changepoint/MUD_inverse/data/data.jsonl.filtered.cleaned_kmeans_100/inverse_output/test.small.jsonl"
endpoint = "https://api.openai.com/v1/chat/completions"
os.makedirs("./outputs", exist_ok=True)

df = pd.read_json(file_path, lines=True)

# 1. Rephrase Prompt
df_no_duplicates = df.drop_duplicates(subset=["unit"])
records = []
for index, row in df_no_duplicates.iterrows():
    prompt = REPHRASE_PROMPT.format(row["unit"])
    record = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    records.append(record)
    
write(records, "./outputs/rephrase_prompts.jsonl")

# 2. Inversion Prompt
records = []
for index, row in df.iterrows():
    prompt = build_inverse_prompt_gpt4(row["rephrase"])
    record = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    for _ in range(100):
        records.append(record)
        
write(records, "./outputs/inverse_prompts.jsonl")

# 3. Inversion (In-Context) Prompt
df_author = df.groupby("author_id").agg(list).reset_index()
records = []
for index, row in df.iterrows():
    author_id = row["author_id"]
    in_context_examples = list(set(df_author[df_author["author_id"] == author_id]["unit"].values[0]))
    in_context_examples = [unit for unit in in_context_examples if unit != row["unit"]]
    prompt = build_inverse_prompt_gpt4(row["rephrase"], in_context_examples)
    record = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    for _ in range(5):
        records.append(record)
        
write(records, "./outputs/inverse_in_context_prompts.jsonl")