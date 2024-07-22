
import os
import sys
from argparse import ArgumentParser
from collections import Counter

import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk

sys.path.append("../changepoint")
from prompts import PROMPT_NAMES

np.random.seed(43)

parser = ArgumentParser()
parser.add_argument("--dirname", type=str,
                    default="/data1/yubnub/changepoint/s2orc_changepoint/unit_128",
                    help="Directory where the dataset is stored.")
parser.add_argument("--debug", default=False, action="store_true",
                    help="Work on the debug split.")
args = parser.parse_args()

MODEL_NAMES = [
    "Mistral-7B-Instruct-v0.3",
    "Meta-Llama-3-8B-Instruct",
    "Phi-3-mini-4k-instruct"
]

def main():
    N = 1000 * len(MODEL_NAMES) * len(PROMPT_NAMES)
    split = "test_clean_and_joined"
    split += "_debug" if args.debug else ""
    dataset = load_from_disk(os.path.join(args.dirname, split))
    indices = np.random.permutation(len(dataset)).tolist()

    records = []
    counts = Counter()
    seen_human = set()
    seen_machine = set()
    num_duplicates = 0
    for i in indices:
        
        for units in dataset[i]["units"]:
            if counts[("human", None)] < N and units not in seen_human and units not in seen_machine:
                records.append({
                    "text": units,
                    "label": 0,
                    "prompt": None,
                    "model": "human",
                })
                counts[("human", None)] += 1
                seen_human.add(units)
            else:
                break
        
        for model_name in MODEL_NAMES:
            for prompt in PROMPT_NAMES:
                generations_key = f"{model_name}_prompt={prompt}_generations"
                cp_key = f"{model_name}_prompt={prompt}_changepoint_indices"
                
                for j, generation in enumerate(dataset[i][generations_key]):
                    unit_index = dataset[i][cp_key][j]
                    if prompt == "continuation":
                        unit_index -= 1
                        
                    if dataset[i]["units"][unit_index] == generation:
                        num_duplicates += 1
                        continue


                    if counts[(model_name, prompt)] < N // (len(MODEL_NAMES) * len(PROMPT_NAMES)) and generation not in seen_machine and generation not in seen_human:
                        records.append({
                            "text": generation,
                            "label": 1,
                            "prompt": prompt,
                            "model": model_name,
                        })
                        seen_machine.add(generation)
                        counts[(model_name, prompt)] += 1
                    else:
                        break
                
        num_left = 2 * N - sum(counts.values())
        print("{}/{} records left".format(num_left, 2 * N))
        if num_left == 0:
            break
    
    print(f"Number of duplicates: {num_duplicates}")
    records_df = pd.DataFrame(records)
    MTD_dataset = Dataset.from_pandas(records_df)
    MTD_dataset.save_to_disk(os.path.join(args.dirname, "MTD_dataset"))
    return 0

if __name__ == "__main__":
    sys.exit(main())