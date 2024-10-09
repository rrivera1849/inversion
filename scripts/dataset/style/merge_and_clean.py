
import os
import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from tqdm import tqdm

prefix = sys.argv[1]
DIRNAME = "/data1/yubnub/changepoint/MUD_inverse/raw/generations"
DEBUG = False
nrows = 100 if DEBUG else None
filenames = os.listdir(DIRNAME)
filenames = [fname for fname in filenames if prefix in fname]
print(filenames)
print(len(filenames))
input("Continue?")
filenames = [os.path.join(DIRNAME, fname) for fname in filenames if prefix in fname]

df = pd.concat([pd.read_json(fname, lines=True, nrows=nrows) for fname in filenames])
df.drop_duplicates(["unit", "rephrase"], inplace=True)

# ensure LLMs did something reasonable
def unit_is_rephrase(row: dict):
    return row["unit"] == row["rephrase"]

@torch.no_grad()
def semantic_similarity(units: list[str], rephrases: list[str], chunksize: int = 10_000):
    assert len(units) == len(rephrases)
    N = len(units)
    model = SentenceTransformer("all-mpnet-base-v2")
    all_results = []
    for i in tqdm(range(0, N, chunksize)):
        batch_units = units[i:i+chunksize]
        batch_rephrases = rephrases[i:i+chunksize]
        embeddings_1 = model.encode(batch_units, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
        embeddings_2 = model.encode(batch_rephrases, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
        result = cos_sim(embeddings_1, embeddings_2)
        result = np.diagonal(result).tolist()
        all_results.extend(result)
    return np.array(all_results)

mask = df.apply(unit_is_rephrase, axis=1)
df = df[~mask]

similarities = semantic_similarity(df.unit.tolist(), df.rephrase.tolist())
mask = similarities >= 0.7
df = df[mask]

df = df.groupby("author_id").agg(list).reset_index(drop=False)
mask = df.rephrase.apply(lambda x: len(x)) > 20
df = df[mask]
df.drop(columns=["dataset_index"], inplace=True)

save_dirname = os.path.dirname(DIRNAME)
save_fname = os.path.join(save_dirname, prefix + ".cleaned")
if DEBUG:
    save_fname += ".debug"
df.to_json(save_fname, lines=True, orient="records")