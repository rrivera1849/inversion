
import os
import pickle
import random
import sys
from argparse import ArgumentParser
from tqdm import tqdm

from datasets import load_dataset

random.seed(43)

parser = ArgumentParser()
parser.add_argument("--metadata_file", type=str,
                    default="/data1/yubnub/changepoint/s2orc_changepoint/base/metadata.pkl",
                    help="Path to the metadata file where the indices are stored.")
parser.add_argument("--cache_dir", type=str, 
                    default="/data1/yubnub/changepoint/allenai/peS2o/",
                    help="Path to the HuggingFace datasets cache directory.")
parser.add_argument("--num_train", type=int, default=1_000_000,
                    help="Number of research papers to use for training.")
parser.add_argument("--num_val", type=int, default=5_000,
                    help="Number of research papers to use for validation.")
parser.add_argument("--num_test", type=int, default=10_000,
                    help="Number of research papers to use for testing.")
parser.add_argument("--debug", default=False, action="store_true",
                    help="Debug mode, runs in a small subset of data.")
args = parser.parse_args()

def split_queries_and_targets(
    text: list[str],
    E: int = 16,
):
    assert len(text) >= 2*E
    random.shuffle(text)
    queries = text[:E]
    targets = text[E:2*E]

    return queries, targets

def main():
    metadata = pickle.load(open(args.metadata_file, "rb"))
    used_train_indices = metadata["train_indices"]
    
    dataset = load_dataset("allenai/peS2o", cache_dir=args.cache_dir)
    N = 1000 if args.debug else args.num_train + args.num_val + args.num_test

    import pdb; pdb.set_trace()

    samples = []
    pbar = tqdm(total=N)
    for i in reversed(range(len(dataset["train"]))):
        if i in used_train_indices:
            continue
        
        if dataset["train"][i]["source"] != "s2orc/train":
            continue
        
        sample = dataset["train"][i]
        samples.append(sample)
        pbar.update(1)
        
        if len(samples) >= N:
            break

    import pdb; pdb.set_trace()


    return 0

if __name__ == "__main__":
    sys.exit(main())