# Stratified Sampling Copy

import json
import os
import sys

import pandas as pd
from termcolor import colored

# splitname we want to copy:
SPLIT_FILENAME = \
    "/data1/foobar/changepoint/MUD_inverse/data/data.jsonl.filtered.cleaned_kmeans_100/splitname_to_author_ids.json"

DATA_PATH = "/data1/foobar/changepoint/MUD_inverse/data"

def main():
    split_to_author_ids = json.load(open(SPLIT_FILENAME))
    
    fname_to_split = sys.argv[1]
    dirname = os.path.basename(fname_to_split)
    input(colored("Creating: {}, Continue?".format(dirname), "green"))

    os.makedirs(os.path.join(DATA_PATH, dirname), exist_ok=True)
    
    df = pd.read_json(fname_to_split, lines=True)
    to_explode = [col for col in df.columns if col != "author_id"]
    df = df.explode(to_explode)
    records = df.to_dict(orient="records")
    
    train_fout = open(os.path.join(DATA_PATH, dirname, "train.jsonl"), "w+")
    valid_fout = open(os.path.join(DATA_PATH, dirname, "valid.jsonl"), "w+")
    test_fout = open(os.path.join(DATA_PATH, dirname, "test.jsonl"), "w+")
    for record in records:
        author_id = record["author_id"]
        if author_id in split_to_author_ids["train"]:
            train_fout.write(json.dumps(record) + "\n")
        elif author_id in split_to_author_ids["valid"]:
            valid_fout.write(json.dumps(record) + "\n")
        elif author_id in split_to_author_ids["test"]:
            test_fout.write(json.dumps(record) + "\n")
    train_fout.close()
    valid_fout.close()
    test_fout.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
