
import os
import sys
from argparse import ArgumentParser

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
    N = 1000 # 1000 per LLM per prompt, equal amount of humans
    split = "test_clean_and_joined"
    split += "_debug" if args.debug else ""
    dataset = load_from_disk(os.path.join(args.dirname, split))
    indices = np.random.permutation(len(dataset))[:N].tolist()

    records = []
    for i in indices:
        for model_name in MODEL_NAMES:
            for prompt in PROMPT_NAMES:
                generations_key = f"{model_name}_prompt={prompt}_generations"
                changepoints_key = f"{model_name}_prompt={prompt}_changepoint_indices"

                unit_index = dataset[i][changepoints_key][0]
                if prompt == "continuation":
                    unit_index -= 1
                reference = dataset[i]["units"][unit_index]
                generation = dataset[i][generations_key][0]

                records.append({
                    "text": reference,
                    "label": 0,
                    "prompt": None,
                    "model": "human",
                })

                records.append({
                    "text": generation,
                    "label": 1,
                    "prompt": prompt,
                    "model": model_name,
                })
    
    records_df = pd.DataFrame(records)
    MTD_dataset = Dataset.from_pandas(records_df)
    MTD_dataset.save_to_disk(os.path.join(args.dirname, "MTD_dataset"))

    return 0

if __name__ == "__main__":
    sys.exit(main())