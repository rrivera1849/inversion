
import os
import random
import sys

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

random.seed(43)

def main():
    base_path = "/data1/yubnub/data/iur_dataset"
    os.makedirs(os.path.join(base_path, "author_100"), exist_ok=True)

    df_train = pd.read_json(os.path.join(base_path, "train.jsonl"), lines=True)
    df_queries = pd.read_json(os.path.join(base_path, "test_queries.jsonl"), lines=True)

    # 1. Get the intersection of the author_ids and select 100 random authors
    intersection = list(set.intersection(
        set(df_train["author_id"].tolist()),
        set(df_queries["author_id"].tolist())
    ))
    intersection = random.sample(intersection, 100)
    df_train = df_train[df_train["author_id"].isin(intersection)]
    df_queries = df_queries[df_queries["author_id"].isin(intersection)]

    # 2. Balance the number of texts per author
    df_train = df_train[["author_id", "syms"]]
    N = df_train["syms"].apply(len).min()
    df_train["syms"] = df_train["syms"].apply(lambda x: random.sample(x, N))

    # 3. Split the training data into training and validation
    def split_into_valid(row):
        # 10% of the symbols are used for validation
        n = len(row["syms"])
        random.shuffle(row["syms"])
        row["syms_valid"] = row["syms"][:n//10]
        row["syms"] = row["syms"][n//10:]
        return row
    
    df_train = df_train.progress_apply(split_into_valid, axis=1)
    df_valid = df_train[["author_id", "syms_valid"]]
    df_valid.rename(columns={"syms_valid": "syms"}, inplace=True)
    df_train = df_train[["author_id", "syms"]]

    df_queries = df_queries[["author_id", "syms"]]

    df_train = df_train.explode("syms").reset_index(drop=True)
    df_valid = df_valid.explode("syms").reset_index(drop=True)
    df_queries = df_queries.explode("syms").reset_index(drop=True)

    df_train.to_json(os.path.join(base_path, "author_100", "train.jsonl"), lines=True, orient="records")
    df_valid.to_json(os.path.join(base_path, "author_100", "valid.jsonl"), lines=True, orient="records")
    df_queries.to_json(os.path.join(base_path, "author_100", "test.jsonl"), lines=True, orient="records")
    return 0

if __name__ == "__main__":
    sys.exit(main())