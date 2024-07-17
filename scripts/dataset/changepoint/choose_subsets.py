"""Chooses a subset of the allenai/pes2o dataset to use for 
   the creation of our changepoint dataset.
"""

import pickle
import os
import sys
from argparse import ArgumentParser

import numpy as np
from datasets import load_dataset, Dataset
from termcolor import colored

np.random.seed(43)

parser = ArgumentParser()
parser.add_argument("--ntrain", type=int, default=100_000,
                    help="Number of training examples to use.")
parser.add_argument("--nval", type=int, default=25_000,
                    help="Number of validation examples to use.")

parser.add_argument("--savedir", type=str, 
                    default="/data1/yubnub/changepoint/s2orc_changepoint/base",
                    help="Path to the directory where the dataset will be saved.")
parser.add_argument("--metadata_file", type=str,
                    default=None,
                    help="Path to the metadata file where the indices are stored.")

parser.add_argument("--cache_dir", type=str, 
                    default="/data1/yubnub/changepoint/allenai/peS2o/",
                    help="Path to the HuggingFace datasets cache directory.")
args = parser.parse_args()

def main():
    if os.path.isdir(args.savedir):
        print(colored(f"Directory {args.savedir} already exists. Exiting...", "red"))
        return 1
    
    os.makedirs(args.savedir, exist_ok=False)
    
    dataset = load_dataset("allenai/peS2o", cache_dir=args.cache_dir)

    train_s2orc = dataset["train"].filter(
        lambda example: "s2orc" in example["source"],
        num_proc=40,
    )
    if args.metadata_file is not None:
        test_changepoint = dataset["validation"].filter(
            lambda example: "s2orc" in example["source"],
            num_proc=40,
        )
    else:
        test_changepoint = dataset["test"].filter(
            lambda example: "s2orc" in example["source"],
            num_proc=40,
        )

    if args.metadata_file is not None:
        metadata = pickle.load(open(args.metadata_file, "rb"))
        indices_used = metadata["train_indices"].tolist() + metadata["validation_indices"].tolist()
        indices = set(range(len(train_s2orc))) - set(indices_used)
        indices = np.random.permutation(list(indices))[:args.ntrain+args.nval]
        train_indices = indices[:args.ntrain]
        validation_indices = indices[args.ntrain:]
        train_changepoint = train_s2orc.select(train_indices)
        validation_changepoint = train_s2orc.select(validation_indices)
    else:
        indices = np.random.permutation(len(train_s2orc))[:args.ntrain+args.nval]
        train_indices = indices[:args.ntrain]
        validation_indices = indices[args.ntrain:]
        train_changepoint = train_s2orc.select(train_indices)
        validation_changepoint = train_s2orc.select(validation_indices)

    for split, data in {
        "train": train_changepoint,
        "validation": validation_changepoint,
        "test": test_changepoint,
    }.items():
        if data is None:
            continue
        assert len(data) == {
            "train": args.ntrain,
            "validation": args.nval,
            "test": len(test_changepoint),
        }[split]
        
        split_savedir = os.path.join(args.savedir, split)
        data.save_to_disk(split_savedir)
        print(colored(f"Saved {len(data)} examples to {split_savedir}", "green"))

    with open("{}".format(os.path.join(args.savedir, "metadata.pkl")), "wb") as fout:
        pickle.dump({
            "train_indices": train_indices,
            "validation_indices": validation_indices,
        }, fout)

    return 0

if __name__ == "__main__":
    sys.exit(main())