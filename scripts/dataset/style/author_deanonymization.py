
import os
import random
import sys
sys.path.append("../../../src/baselines/mixture/")
from copy import deepcopy

import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

from check import read_all_mixture_data, read_all_inverse_data

tqdm.pandas()
random.seed(43)

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

SUFFIX = ".politics"

MIXTURE_TRAIN_PATH = "../../../src/baselines/mixture/datasets/all_roberta-large_250000_stratified/"
INVERSE_TRAIN_PATH = "../../../src/baselines/mixture/datasets/all_roberta-large_250000_stratified_inverse/"
df_mixture = read_all_mixture_data(MIXTURE_TRAIN_PATH)
df_inverse = read_all_inverse_data(INVERSE_TRAIN_PATH)
all_data = df_mixture.union(df_inverse)

def filter_by_lengths(row):
    N = len(row["syms"])
    indices_to_keep = []
    for i in range(N):
        if row["num_tokens"][i] >= 128-32 and row["num_tokens"][i] <= 128+32 and row["syms"][i] not in all_data:
            indices_to_keep.append(i)
    for k, v in row.items():
        if isinstance(v, list):
            row[k] = [v[i] for i in indices_to_keep]
    return row

def main():
    base_path = "/data1/foobar/data/iur_dataset"

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
    queries.rename(columns={"author_id": "id", "syms": "unit"}, inplace=True)
    targets = deepcopy(df)
    targets.syms = targets.syms.apply(lambda x: x[1])
    targets.rename(columns={"author_id": "id", "syms": "unit"}, inplace=True)

    save_dirname = "/home/riverasoto1/repos/changepoint/src/baselines/mixture/prompting_data/author"
    os.makedirs(save_dirname, exist_ok=True)
    queries.to_json(os.path.join(save_dirname, "queries.jsonl"), lines=True, orient="records")
    targets.to_json(os.path.join(save_dirname, "targets.jsonl"), lines=True, orient="records")

if __name__ == "__main__":
    sys.exit(main())