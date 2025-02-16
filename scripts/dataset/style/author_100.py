
import os
import random
import sys

import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
tqdm.pandas()

random.seed(43)

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

SUFFIX = ".politics"

def split(row):
    # 15% for validation and 15% for test
    N = len(row["syms"])
    indices = list(range(N))
    random.shuffle(indices)
    N_valid = int(N * 0.15)
    N_test = int(N * 0.15)
    N_train = N - N_valid - N_test
    row["syms_train"] = [row["syms"][i] for i in indices[:N_train]]
    row["syms_valid"] = [row["syms"][i] for i in indices[N_train:N_train + N_valid]]
    row["syms_test"] = [row["syms"][i] for i in indices[N_train + N_valid:]]
    return row

def filter_by_lengths(row):
    N = len(row["syms"])
    indices_to_keep = []
    for i in range(N):
        if row["num_tokens"][i] >= 64 and row["num_tokens"][i] <= 512:
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
    # df = df.head(1_000)

    df["num_tokens"] = df["syms"].progress_apply(lambda lst: [len(tokenizer.encode(s)) for s in lst])
    df = df.progress_apply(filter_by_lengths, axis=1)
    df = df[~(df["syms"].apply(len) == 0)]
    df = df[df["syms"].apply(len) >= 50]
    df = df.head(100)

    import pdb; pdb.set_trace()

    df = df[["author_id", "syms"]]
    N = df["syms"].apply(len).min()
    df["syms"] = df["syms"].apply(lambda x: random.sample(x, N))

    df = df.progress_apply(split, axis=1)
    train = df[["author_id", "syms_train"]]
    valid = df[["author_id", "syms_valid"]]
    queries = df[["author_id", "syms_test"]]
    train = train.explode("syms_train").reset_index(drop=True)
    valid = valid.explode("syms_valid").reset_index(drop=True)
    queries = queries.explode("syms_test").reset_index(drop=True)
    train = train.rename(columns={"syms_train": "syms"})
    valid = valid.rename(columns={"syms_valid": "syms"})
    queries = queries.rename(columns={"syms_test": "syms"})
    train.to_json(os.path.join(base_path, dirname, "train.jsonl"), lines=True, orient="records")
    valid.to_json(os.path.join(base_path, dirname, "valid.jsonl"), lines=True, orient="records")
    queries.to_json(os.path.join(base_path, dirname, "test.jsonl"), lines=True, orient="records")

if __name__ == "__main__":
    sys.exit(main())