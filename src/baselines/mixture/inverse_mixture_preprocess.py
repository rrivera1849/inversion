
import os
import sys
from argparse import ArgumentParser

import pandas as pd

from utils import load_mixture_predictor, get_mixture_weights

parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="./datasets/s2orc_roberta-large_4150313_inverse",
                    help="Name of the model to use.")
parser.add_argument("--debug", default=False, action="store_true")
args = parser.parse_args()

def process(
    path: str,
    chunksize: int = 10_000,
):
    mixture_predictor = load_mixture_predictor()

    total_processed = 0
    chunk_it = pd.read_json(path, lines=True, chunksize=chunksize)
    
    basename = os.path.basename(path)
    fout_name = os.path.join(os.path.dirname(path), f"{basename}.mixture")
    for chunk in chunk_it:
        weights = get_mixture_weights(mixture_predictor, chunk["generation"].tolist())
        tokens = [mixture_predictor.tokenizer.tokenize(sample) for sample in chunk["generation"].tolist()]
        for i, (weight, token) in enumerate(zip(weights, tokens)):
            chunk.at[i, "mixture_probs"] = weight
            chunk.at[i, "mixture_tokens"] = token
        chunk.to_json(fout_name, lines=True, orient="records", mode="a")
        print("total_processed={}".format(total_processed))
        total_processed += len(chunk)
        
        if args.debug:
            break
    
def main():
    filenames = ["train.jsonl", "valid.jsonl"]
    for filename in filenames:
        path = os.path.join(args.dataset_path, filename)
        if os.path.isfile(path):
            print(f"Processing {filename}")
            process(path)

    return 0

if __name__ == "__main__":
    sys.exit(main())