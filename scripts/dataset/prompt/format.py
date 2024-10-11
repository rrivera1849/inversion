
import json
import os
import sys
from collections import defaultdict

import pandas as pd

BASE_INSTRUCTION = "The following passage is a mix of human and machine text, recover the original human text:"

def get_rephrase_to_unit():
    test_file = "/data1/yubnub/changepoint/MUD_inverse/data/data.jsonl.filtered.cleaned_kmeans_100/inverse_output/test.small.jsonl"
    df = pd.read_json(test_file, lines=True)
    rephrase_to_unit_d = {}
    for _, row in df.iterrows():
        rephrase_to_unit_d[row["rephrase"]] = row["unit"]
    return rephrase_to_unit_d

def to_dataframe(rephrase_to_unit_d, rephrase_to_inverse_d):
    units = []
    rephrases = []
    inverses = []
    for rephrase, unit in rephrase_to_unit_d.items():
        inverse = rephrase_to_inverse_d[rephrase]
        units.append(unit)
        rephrases.append(rephrase)
        inverses.append(inverse)
    df = pd.DataFrame({
        "unit": units,
        "rephrase": rephrases,
        "inverse": inverses
    })
    return df

def format(rephrase_to_unit_d, fname, cutoff=BASE_INSTRUCTION):
    rephrase_to_inverse_d = defaultdict(list)
    with open(fname) as fin:
        for line in fin.readlines():
            record = json.loads(line)
            prompt = record[0]['messages'][0]['content']
            index = prompt.index(cutoff)
            rephrase = prompt[index+len(cutoff)+1:]
            inverse = record[1]['choices'][0]['message']['content']
            rephrase_to_inverse_d[rephrase].append(inverse)
    for rephrase, inverse_list in rephrase_to_inverse_d.items():
        if "in_context" in fname:
            assert len(inverse_list) == 5
        elif "inverse" in fname:
            assert len(inverse_list) == 100
        else:
            assert len(inverse_list) == 1
    df = to_dataframe(rephrase_to_unit_d, rephrase_to_inverse_d)
    return df

def main():
    os.makedirs("./ready", exist_ok=True)
    rephrase_to_unit_d = get_rephrase_to_unit()

    df = format(rephrase_to_unit_d, "./outputs/inverse_prompts.jsonl.result")
    df.to_json("./ready/gpt4_inverse.jsonl", orient="records", lines=True)

    df = format(rephrase_to_unit_d, "./outputs/inverse_in_context_prompts.jsonl.result")
    df.to_json("./ready/gpt4_inverse_in_context.jsonl", orient="records", lines=True)

    units = []
    rephrases = []
    with open("./outputs/rephrase_prompts.jsonl.result") as fin:
        for line in fin.readlines():
            record = json.loads(line)
            prompt = record[0]['messages'][0]['content']
            
            index = prompt.index("\nOnly output the rephrased-passage, do not include any other details.")
            unit = prompt[:index][len("Rephrase the following passage: "):]
            rephrase = record[1]['choices'][0]['message']['content']
            units.append(unit)
            rephrases.append(rephrase)
            
    df = pd.DataFrame({
        "unit": units,
        "rephrase": rephrases
    })
    df.to_json("./ready/test.small.gpt4.jsonl", orient="records", lines=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())