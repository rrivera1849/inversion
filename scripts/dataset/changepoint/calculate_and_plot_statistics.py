
import os
import sys
from argparse import ArgumentParser
from typing import Dict, Union

import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import AutoTokenizer

from prompts import PROMPT_NAMES

parser = ArgumentParser()
parser.add_argument("--dirname", type=str,
                    default="/data1/yubnub/changepoint/s2orc_changepoint/unit_128",
                    help="Directory where the dataset is stored.")
parser.add_argument("--split", type=str, default="test",
                    help="Dataset split to clean and join the generations for.")
parser.add_argument("--debug", default=False, action="store_true",
                    help="If True, will process only a few samples.")
args = parser.parse_args()

MODEL_NAMES = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct"
]
TOKENIZERS = {}
for model_name in MODEL_NAMES:
    TOKENIZERS[model_name.split("/")[1]] = \
        AutoTokenizer.from_pretrained(model_name)

def calculate_statistics(
    unit_reference: str, 
    unit_generated: str,
    model_name: str,
) -> Dict[str, Union[int, float]]:
    """Calculate statistics for a given reference and generated unit.
    """
    statistics = {}
        
    tokenizer = TOKENIZERS[model_name]
    
    tokens_in_reference = set(tokenizer.tokenize(unit_reference))
    tokens_in_generated = set(tokenizer.tokenize(unit_generated))

    token_overlap = len(set.intersection(tokens_in_reference, tokens_in_generated)) / len(tokens_in_generated)
    statistics["token_overlap"] = token_overlap
    statistics["length_difference"] = len(tokens_in_reference) - len(tokens_in_generated)

    return statistics

def main():
    dataset = load_from_disk(f"{args.dirname}/{args.split}")
    N = 100 if args.debug else len(dataset)

    statistics = {(prompt, model_name): [] for prompt, model_name in zip(PROMPT_NAMES, MODEL_NAMES)}

    for i in range(len(N)):
        for prompt in PROMPT_NAMES:
            for model_name in MODEL_NAMES:
                generations_key = f"{model_name}_prompt={prompt}_generations"
                changepoints_key = f"{model_name}_prompt={prompt}_changepoint_indices"

                for j, generation in enumerate(dataset[i][generations_key]):
                    unit_index = dataset[i][changepoints_key][j]
                    if prompt == "continuation":
                        unit_index -= 1
                        
                    reference = dataset[i]["units"][unit_index]
                    statistics[(prompt, model_name)].append(
                        calculate_statistics(reference, generation, model_name)
                    )

    os.makedirs("./statistics", exist_ok=True)
    
    for (prompt, model_name), stats in statistics.items():
        token_overlap = [stat["token_overlap"] for stat in stats]
        length_difference = [stat["length_difference"] for stat in stats]

        plt.hist(token_overlap, bins=50)
        plt.title(f"Token overlap for {model_name} with prompt={prompt}")
        plt.savefig(f"./statistics/{model_name}_prompt={prompt}_token_overlap.png")
        plt.close()

        plt.hist(length_difference, bins=50)
        plt.title(f"Length difference for {model_name} with prompt={prompt}")
        plt.savefig(f"./statistics/{model_name}_prompt={prompt}_length_difference.png")
        plt.close()

    for prompt in PROMPT_NAMES:
        for stat_name in ["token_overlap", "length_difference"]:
            plt.figure()
            for model_name in MODEL_NAMES:
                stats = statistics[(prompt, model_name)]
                values = [stat[stat_name] for stat in stats]
                plt.hist(values, bins=50, alpha=0.5, label=model_name)
            plt.title(f"{stat_name} for prompt={prompt}")
            plt.legend()
            plt.savefig(f"./statistics/prompt={prompt}_{stat_name}.png")

    return 0

if __name__ == "__main__":
    sys.exit(main())


