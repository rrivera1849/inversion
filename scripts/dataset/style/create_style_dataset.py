
import json
import os
import random
import sys
from argparse import ArgumentParser

from datasets import load_from_disk
from tqdm import tqdm

random.seed(43)

parser = ArgumentParser()
parser.add_argument("--debug", default=False, action="store_true",
                    help="Debug mode, runs in a small subset of data.")
args = parser.parse_args()

def split_queries_and_targets(
    text: list[str],
):
    random.shuffle(text)
    queries = text[:len(text)//2]
    targets = text[len(text)//2:]

    return queries, targets

def main():
    # TODO
    path = "/data1/yubnub/changepoint/s2orc_changepoint/author_unit_128"
    save_dirname = "/data1/yubnub/data/s2orc/"
    os.makedirs(save_dirname, exist_ok=True)
    
    dataset = load_from_disk(os.path.join(path, "train"))

    train_fout = open(os.path.join(save_dirname, "train.jsonl"), "w+")
    for i in tqdm(range(len(dataset))):
        record = {}
        record["author_id"] = dataset[i]["id"]
        record["syms"] = dataset[i]["units"]
        train_fout.write(json.dumps(record)); train_fout.write('\n')
        if args.debug and i >= 99:
            break
    train_fout.close()
    
    dataset = load_from_disk(os.path.join(path, "validation"))
    validation_queries_fout = open(os.path.join(save_dirname, "validation_queries.jsonl"), "w+")
    validation_targets_fout = open(os.path.join(save_dirname, "validation_targets.jsonl"), "w+")
    for i in tqdm(range(len(dataset))):
        record = {}
        record["author_id"] = dataset[i]["id"]
        queries, targets = split_queries_and_targets(dataset[i]["units"])
        record["syms"] = queries
        validation_queries_fout.write(json.dumps(record)); validation_queries_fout.write('\n')
        record["syms"] = targets
        validation_targets_fout.write(json.dumps(record)); validation_targets_fout.write('\n')
        if args.debug and i >= 99:
            break
    validation_queries_fout.close(); validation_targets_fout.close()

    return 0

if __name__ == "__main__":
    sys.exit(main())