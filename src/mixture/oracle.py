
import os
import random

import pandas as pd
from termcolor import colored
from transformers import AutoTokenizer
from tqdm import tqdm

from utils import get_levenshtein_tags

random.seed(43)

DATA_PATH = "/data1/foobar/data/iur_dataset/author_100.politics"

tokenizer = AutoTokenizer.from_pretrained("roberta-large")
split_names = ["train", "valid", "test"]
for split in split_names:
    print(colored(f"Processing {split} split...", "blue"))
    df_orig = pd.read_json(os.path.join(DATA_PATH, f"{split}.jsonl"), lines=True)
    df_gen = pd.read_json(os.path.join(DATA_PATH, f"{split}.jsonl.mistral"), lines=True)
    df_mixed = pd.read_json(os.path.join(DATA_PATH, f"{split}.jsonl.mistral.mixed"), lines=True)

    generations = df_gen["syms"].tolist()
    original = df_orig["syms"].tolist()
    assert len(generations) == len(original)

    generation_to_id = {gen: i for i, gen in enumerate(generations)}

    tags = []
    for i in tqdm(range(len(generations))):
        tags.append(get_levenshtein_tags(generations[i], original[i], tokenizer.tokenize))

    # we want those tags which are "human"
    tags = [[float(tag == "KEEP") for tag in elem] for elem in tags]
    tags = [[t, t] for t in tags]
    df_gen["token_mixture_preds"] = tags

    tags_mixed = []
    num = 0
    for index, row in df_mixed.iterrows():
        if row["syms"] in generation_to_id:
            tags_mixed.append(tags[generation_to_id[row["syms"]]])
            num += 1
        else:
            tags_mixed.append([[1.0, 1.0]] * len(tokenizer.tokenize(row["syms"])))
    
    expected = df_mixed.author_id.value_counts().unique()[0] // 2
    expected *= 100
    assert num == expected
    df_mixed["token_mixture_preds"] = tags_mixed
    
    df_orig["token_mixture_preds"] = df_orig["syms"].apply(lambda x: [[1.0, 1.0]] * len(tokenizer.tokenize(x)))
    # df_orig.to_json(os.path.join(DATA_PATH, f"{split}.jsonl.oracle"), orient="records", lines=True)
    df_orig["token_mixture_preds"] = df_orig["syms"].apply(lambda x: [[random.uniform(0, 1), random.uniform(0, 1)] for _ in tokenizer.tokenize(x)])
    # df_orig.to_json(os.path.join(DATA_PATH, f"{split}.jsonl.uniform"), orient="records", lines=True)
    
    # df_gen.to_json(os.path.join(DATA_PATH, f"{split}.jsonl.mistral.oracle"), orient="records", lines=True)
    # df_mixed.to_json(os.path.join(DATA_PATH, f"{split}.jsonl.mistral.mixed.oracle"), orient="records", lines=True)

    # While we're here we can also do our uniform baseline
    df_gen["token_mixture_preds"] = df_gen["token_mixture_preds"].apply(lambda x: [[random.uniform(0, 1), random.uniform(0, 1)] for _ in x])
    df_mixed["token_mixture_preds"] = df_mixed["token_mixture_preds"].apply(lambda x: [[random.uniform(0, 1), random.uniform(0, 1)] for _ in x])

    # df_gen.to_json(os.path.join(DATA_PATH, f"{split}.jsonl.mistral.uniform"), orient="records", lines=True)
    # df_mixed.to_json(os.path.join(DATA_PATH, f"{split}.jsonl.mistral.mixed.uniform"), orient="records", lines=True)

    