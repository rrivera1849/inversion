
import os
import sys
from argparse import ArgumentParser
from functools import partial

from datasets import load_from_disk

from utils import split_into_units

parser = ArgumentParser()
parser.add_argument("--num_max_tokens", type=int, default=128)
args = parser.parse_args()

def main():
    print("BE CAREFUL MAN, THERE ARE SOME CRAZY VARS HERE FIX LATER")
    
    # TODO
    save_dir = f"/data1/foobar/changepoint/s2orc_changepoint/author_unit_{args.num_max_tokens}"
    os.makedirs(save_dir, exist_ok=True)
    # for split in ["validation", "test", "train"]:
    for split in ["validation", "train"]:
        print(f"Processing {split}...")
        path = f"/data1/foobar/changepoint/s2orc_changepoint/base_author/{split}"
        changepoint = load_from_disk(path)
        split_fn = partial(split_into_units, num_max_tokens=args.num_max_tokens)
        changepoint = changepoint.map(split_fn, num_proc=40)
        changepoint.save_to_disk(f"{save_dir}/{split}")

    return 0

if __name__ == "__main__":
    sys.exit(main())