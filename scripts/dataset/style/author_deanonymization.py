
import os
import random
import sys
from copy import deepcopy

import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
tqdm.pandas()

random.seed(43)

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

SUFFIX = ".politics"

def filter_by_lengths(row):
    N = len(row["syms"])
    indices_to_keep = []
    for i in range(N):
        if row["num_tokens"][i] >= 128-32 and row["num_tokens"][i] <= 128+32:
            indices_to_keep.append(i)
    for k, v in row.items():
        if isinstance(v, list):
            row[k] = [v[i] for i in indices_to_keep]
    return row

def main():
    base_path = "/data1/yubnub/data/iur_dataset"

    dirname = "author_100" + SUFFIX
    os.makedirs(os.path.join(base_path, dirname), exist_ok=True)

    df = pd.read_json(os.path.join(base_path, "train.jsonl" + SUFFIX), lines=True)
    df = df[df["syms"].apply(len) > 50]
    
    df["average_text_len"] = df["lens"].apply(lambda x: sum(x) / len(x))
    df = df.sort_values("average_text_len", ascending=False)

    df["num_tokens"] = df["syms"].progress_apply(lambda lst: [len(tokenizer.encode(s)) for s in lst])
    df = df.progress_apply(filter_by_lengths, axis=1)
    df = df[~(df["syms"].apply(len) == 0)]
    df = df.head(100)

    df = df[["author_id", "syms"]]
    df = df.explode("syms")
    df = df.drop_duplicates(subset=["syms"])
    df = df.groupby("author_id").agg(list).reset_index()
    df = df[df.syms.apply(len) >= 2]

    df.syms = df.syms.apply(lambda x: random.sample(x, k=2))
    queries = deepcopy(df)
    queries.syms = queries.syms.apply(lambda x: x[0])
    targets = deepcopy(df)
    targets.syms = targets.syms.apply(lambda x: x[1])

    save_dirname = "/home/riverasoto1/repos/changepoint/src/baselines/mixture/prompting_data/author"
    os.makedirs(save_dirname, exist_ok=True)
    queries.to_json(os.path.join(save_dirname, "queries.jsonl"), lines=True, orient="records")
    targets.to_json(os.path.join(save_dirname, "targets.jsonl"), lines=True, orient="records")

if __name__ == "__main__":
    sys.exit(main())