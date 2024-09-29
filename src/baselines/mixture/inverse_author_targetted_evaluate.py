
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from tqdm.auto import tqdm
tqdm.pandas()

from embedding_utils import *
from metric_utils import *
from sentence_transformers import util

base_path = "/data1/yubnub/changepoint/MUD_inverse/data/data.jsonl.filtered.cleaned_kmeans_100/inverse_output"
files = [
    "none_targetted=examples_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=5.targetted_mode=author",
    "none_targetted_6400_temperature=0.7_top_p=0.9.jsonln=5.targetted_mode=author",
]

def calculate_EER(labels, sims):
    fpr, tpr, _ = roc_curve(labels, sims, pos_label=1)
    fnr = 1 - tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return EER

def flatten(lst):
    return [item for sublist in lst for item in sublist]

def get_unique(df, author_key="author_y", textkey="unit_y"):
    unique_authors = []
    unique_text = []
    seen = set()
    for i in range(len(df)):
        for j in range(len(df.iloc[i][textkey])):
            try:
                author = df.iloc[i][author_key][j]
            except:
                author = df.iloc[i][author_key]
            text = df.iloc[i][textkey][j]
            
            key = []
            key.append(author)
            if isinstance(text, list):
                key.extend(text)
            else:
                key.append(text)
                
            key = tuple(key)
            if key not in seen:
                unique_authors.append(author)
                unique_text.append(text)
                seen.add(key)
    return unique_authors, unique_text

for file in files:
    path = os.path.join(base_path, file)
    df = pd.read_json(path, lines=True)
    df = df[["author_id_x", "unit_x", "rephrase_x", "inverse", "unit_y", "author_id_y"]]
    df = df.groupby("author_id_x").agg(list).reset_index()
    
    x_unique = set(df.author_id_x.tolist())
    y_unique = set(flatten(df.author_id_y.tolist()))
    to_remove = []
    for author in y_unique:
        if author not in x_unique:
            to_remove.append(author)
    def process(row):
        indices_to_remove = [i for i in range(len(row["author_id_y"])) if row["author_id_y"][i] in to_remove]
        indices_to_remove = indices_to_remove[::-1]
        for _, v in row.items():
            if isinstance(v, list):
                for i in indices_to_remove:
                    v.pop(i)
        return row
    df.progress_apply(process, axis=1)
    
    queries_author, queries_text = get_unique(df, author_key="author_id_y", textkey="unit_y")
    assert len(queries_text) == len(set.intersection(x_unique, y_unique))

    rephrase_author = df.author_id_x.tolist()
    rephrase_text = df.rephrase_x.apply(lambda x: list(set(x))).tolist()
    assert len(rephrase_text) == len(set.intersection(x_unique, y_unique))

    luar, luar_tok = load_luar_model_and_tokenizer()
    luar.to("cuda")

    query_author_embeddings = [get_luar_author_embeddings(text, luar, luar_tok) for text in tqdm(queries_text)]
    query_author_embeddings = torch.cat(query_author_embeddings, dim=0)

    rephrase_instance_embeddings = [get_luar_instance_embeddings(text, luar, luar_tok) for text in tqdm(rephrase_text)]
    rephrase_instance_embeddings = torch.cat(rephrase_instance_embeddings, dim=0)

    metrics = {}
    
    queries_author = np.array(queries_author)
    rephrase_author = np.array(rephrase_author)
    labels_author_author = (queries_author[:, None] == rephrase_author).astype(int)
    labels_author_instance = []
    for i in range(len(labels_author_author)):
        for j in range(len(labels_author_author[i])):
            if labels_author_author[i][j] == 1:
                labels_author_instance.extend([1] * len(rephrase_text[i]))
            else:
                labels_author_instance.append(0)
    
    import pdb; pdb.set_trace()
    similarities = util.pytorch_cos_sim(query_author_embeddings, rephrase_instance_embeddings).cpu().numpy().flatten()
    metrics["EER_rephrase_author_instance"] = calculate_EER(labels_author_instance, similarities)

    import pdb; pdb.set_trace()



    print(metrics)