
import json
import os
import sys

import pandas as pd

def get_test_authors():
    path = "/data1/foobar/changepoint/MUD_inverse/data/data.jsonl.filtered.cleaned_kmeans_100/inverse_output/test.small.jsonl"
    authors = [json.loads(line)["author_id"] for line in open(path)]
    return list(set(authors))

def main():
    base_path = "/data1/foobar/changepoint/MUD_inverse/data"
    dataset_name = sys.argv[1]
    assert os.path.isdir(os.path.join(base_path, dataset_name)), f"Dataset {dataset_name} not found"

    authors = get_test_authors()
    df = pd.read_json(os.path.join(base_path, dataset_name, "test.jsonl"), lines=True)
    df = df[df.author_id.isin(authors)]
    os.makedirs(os.path.join(base_path, dataset_name, "inverse_output"), exist_ok=True)
    df.to_json(os.path.join(base_path, dataset_name, "inverse_output", "test.small.jsonl"), lines=True, orient="records")

    return 0


if __name__ == "__main__":
    sys.exit(main())