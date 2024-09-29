
import json
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import util
from tqdm import tqdm

from embedding_utils import *
from metric_utils import *

def flatten(lst: list[list[Any]]) -> list[Any]:
    return [item for sublist in lst for item in sublist]

def read_data(path: str):
    df = pd.read_json(path, lines=True)

    # This is all very unfortunate, but I forgot to save the `author_id` in the 
    # small test set I saved, here we are recovering it:
    path_test_full = "/data1/yubnub/changepoint/MUD_inverse/data/data.jsonl.filtered.cleaned_kmeans_100/test.jsonl"
    df_test_full = pd.read_json(path_test_full, lines=True)
    df_test_full = df_test_full.groupby("author_id").agg(list).iloc[:100]
    to_explode = [col for col in df_test_full.columns if col != "author_id"]
    df_test_full = df_test_full.explode(to_explode).reset_index()
    assert df_test_full.unit.tolist() == df.unit.tolist()
    author_ids = df_test_full.author_id.tolist()
    
    df["author_id"] = author_ids
    df.drop_duplicates("unit", inplace=True)
    return df

def get_author_instance_labels(
    df: pd.DataFrame,
    N: int,
    use_inverse: bool = False,
) -> list[int]:
    if use_inverse:
        counts = df["inverse"].apply(len).tolist()
    else:
        counts = df["rephrase"].apply(len).tolist()

    counts = np.cumsum(counts)

    last = 0
    labels = []
    for c in counts:
        l = [0] * N
        l[last:c] = [1] * (c - last)
        last = c
        labels.extend(l)
    return labels

def pairwise_similarity(
    query_embeddings: torch.Tensor,
    inverse_embeddings: list[torch.Tensor],
    type: str = "expected",
) -> list[float]:
    assert type in ["max", "expected"]
    type_to_func = {
        "max": max_similarity,
        "expected": expected_similarity,
    }
    similarities = []
    pbar = tqdm(total=len(query_embeddings) * len(inverse_embeddings))
    for i in range(len(query_embeddings)):
        for j in range(len(inverse_embeddings)):
            similarities.append(type_to_func[type](query_embeddings[i:i+1], inverse_embeddings[j]))
            pbar.update(1)
    return similarities

def calculate_all(
    path: str,
    mode: str = "plagiarism",
):
    assert mode in ["plagiarism", "author"]
    
    metrics = {}
    df = read_data(path)
    df = df.groupby("author_id").agg(list)
    
    luar, luar_tok = load_luar_model_and_tokenizer()
    luar = luar.to("cuda")

    if mode == "plagiarism":
        # query = target
        # task: plagiarism detection
        df["unit"] = df["unit"].apply(lambda x: x[:len(x)//2])
        df["rephrase"] = df["rephrase"].apply(lambda x: x[:len(x)//2])
        df["inverse"] = df["inverse"].apply(lambda x: x[:len(x)//2])
    else:
        # query != target
        # task: author identification
        df["unit"] = df["unit"].apply(lambda x: x[:len(x)//2])
        df["rephrase"] = df["rephrase"].apply(lambda x: x[len(x)//2:])
        df["inverse"] = df["inverse"].apply(lambda x: x[len(x)//2:])
    
    num_author = len(df)
    num_instances = df.rephrase.apply(len).sum()

    author_author_labels = np.identity(num_author, dtype=np.int32).flatten().tolist()
    instance_instance_labels = np.identity(num_instances, dtype=np.int32).flatten().tolist()
    author_instance_labels = get_author_instance_labels(df, num_instances)

    # Compute Author Query
    query_author_embeddings = [get_luar_author_embeddings(unit, luar, luar_tok) for unit in tqdm(df.unit.tolist())]
    query_author_embeddings = torch.cat(query_author_embeddings, dim=0).cpu()

    # Compute Instance Query
    query_instance_embeddings = [get_luar_instance_embeddings(unit, luar, luar_tok) for unit in tqdm(df.unit.tolist())]
    query_instance_embeddings = torch.cat(query_instance_embeddings, dim=0).cpu()

    ##### Rephrases:
    rephrase_instance_embeddings = [get_luar_instance_embeddings(rephrases, luar, luar_tok) for rephrases in tqdm(df.rephrase.tolist())]
    rephrase_instance_embeddings = torch.cat(rephrase_instance_embeddings, dim=0).cpu()
    rephrase_author_embeddings = [get_luar_author_embeddings(rephrases, luar, luar_tok) for rephrases in tqdm(df.rephrase.tolist())]
    rephrase_author_embeddings = torch.cat(rephrase_author_embeddings, dim=0).cpu()
    assert len(rephrase_instance_embeddings) == num_instances
    assert len(rephrase_author_embeddings) == num_author
    
    # Author-Instance:
    similarities = util.pytorch_cos_sim(query_author_embeddings, rephrase_instance_embeddings)
    similarities = similarities.cpu().flatten().tolist()
    metrics.update(calculate_metrics(author_instance_labels, similarities, "rephrase_author-instance"))

    # Author-Author
    similarities = util.pytorch_cos_sim(query_author_embeddings, rephrase_author_embeddings)
    similarities = similarities.cpu().flatten().tolist()
    metrics.update(calculate_metrics(author_author_labels, similarities, "rephrase_author-author"))

    if mode == "plagiarism":
        # Instance-Instance:
        similarities = util.pytorch_cos_sim(query_instance_embeddings, rephrase_instance_embeddings)
        similarities = similarities.cpu().flatten().tolist()
        metrics.update(calculate_metrics(instance_instance_labels, similarities, "rephrase_instance-instance"))

    ##### Inverse (All):
    inverse_all_author_embeddings = [get_luar_author_embeddings(flatten(inverses), luar, luar_tok) for inverses in tqdm(df.inverse.tolist())]
    inverse_all_author_embeddings = torch.cat(inverse_all_author_embeddings, dim=0).cpu()
    inverse_all_instance_embeddings = [[get_luar_author_embeddings(inverse, luar, luar_tok).cpu() for inverse in inverses] for inverses in tqdm(df.inverse.tolist())]
    inverse_all_instance_embeddings = [torch.cat(inverses, dim=0) for inverses in inverse_all_instance_embeddings]
    inverse_all_instance_embeddings = torch.cat(inverse_all_instance_embeddings, dim=0).cpu()
    assert len(inverse_all_instance_embeddings) == num_instances
    assert len(inverse_all_author_embeddings) == num_author

    # Author-Instance:
    similarities = util.pytorch_cos_sim(query_author_embeddings, inverse_all_instance_embeddings)
    similarities = similarities.cpu().flatten().tolist()
    metrics.update(calculate_metrics(author_instance_labels, similarities, "inverse_all_author-instance"))

    # Author-Author
    similarities = util.pytorch_cos_sim(query_author_embeddings, inverse_all_author_embeddings)
    similarities = similarities.cpu().flatten().tolist()
    metrics.update(calculate_metrics(author_author_labels, similarities, "inverse_all_author-author"))

    if mode == "plagiarism":
        # Instance-Instance:
        similarities = util.pytorch_cos_sim(query_instance_embeddings, inverse_all_instance_embeddings)
        similarities = similarities.cpu().flatten().tolist()
        metrics.update(calculate_metrics(instance_instance_labels, similarities, "inverse_all_instance-instance"))
    
    ##### Inverse (Individual Embedding):
    inverse_instance_embeddings = [[get_luar_instance_embeddings(inverse, luar, luar_tok).cpu() for inverse in inverses] for inverses in tqdm(df.inverse.tolist())]
    inverse_author_embeddings = [torch.cat(inverses, dim=0) for inverses in inverse_instance_embeddings]
    inverse_instance_embeddings = flatten(inverse_instance_embeddings)
    assert len(inverse_instance_embeddings) == num_instances
    assert len(inverse_author_embeddings) == num_author
    for simtype in ["expected", "max"]:
        similarities = pairwise_similarity(query_author_embeddings, inverse_instance_embeddings, type=simtype)
        metrics.update(calculate_metrics(author_instance_labels, similarities, f"inverse-{simtype}_author-instance"))
        
        similarities = pairwise_similarity(query_author_embeddings, inverse_author_embeddings, type=simtype)
        metrics.update(calculate_metrics(author_author_labels, similarities, f"inverse-{simtype}_author-author"))

        if mode == "plagiarism":
            similarities = pairwise_similarity(query_instance_embeddings, inverse_instance_embeddings, type=simtype)
            metrics.update(calculate_metrics(instance_instance_labels, similarities, f"inverse-{simtype}_instance-instance"))

    ##### Inverse (Single Inversion):
    df["single_inversion"] = df.inverse.apply(lambda xx: [x[0] for x in xx])
    inverse_single_instance_embeddings = [get_luar_instance_embeddings(inverses, luar, luar_tok) for inverses in tqdm(df.single_inversion.tolist())]
    inverse_single_instance_embeddings = torch.cat(inverse_single_instance_embeddings, dim=0).cpu()
    inverse_single_author_embeddings = [get_luar_author_embeddings(inverses, luar, luar_tok) for inverses in tqdm(df.single_inversion.tolist())]
    inverse_single_author_embeddings = torch.cat(inverse_single_author_embeddings, dim=0).cpu()
    assert len(inverse_single_instance_embeddings) == num_instances
    assert len(inverse_single_author_embeddings) == num_author

    # Author-Instance:
    similarities = util.pytorch_cos_sim(query_author_embeddings, inverse_single_instance_embeddings)
    similarities = similarities.cpu().flatten().tolist()
    metrics.update(calculate_metrics(author_instance_labels, similarities, "inverse_single_author-instance"))
    
    # Author-Author
    similarities = util.pytorch_cos_sim(query_author_embeddings, inverse_single_author_embeddings)
    similarities = similarities.cpu().flatten().tolist()
    metrics.update(calculate_metrics(author_author_labels, similarities, "inverse_single_author-author"))

    if mode == "plagiarism":
        # Instance-Instance:
        similarities = util.pytorch_cos_sim(query_instance_embeddings, inverse_single_instance_embeddings)
        similarities = similarities.cpu().flatten().tolist()
        metrics.update(calculate_metrics(instance_instance_labels, similarities, "inverse_single_instance-instance"))

    return metrics

if __name__ == "__main__":
    os.makedirs("./metrics", exist_ok=True)
    base_path = "/data1/yubnub/changepoint/MUD_inverse/data/data.jsonl.filtered.cleaned_kmeans_100/inverse_output"
    files = [
        "none_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=100",
        "none_6400_temperature=0.3_top_p=0.9.jsonl.vllm_n=100",
        "none_6400_temperature=1.5_top_p=0.9.jsonl.vllm_n=100",
    ]
    for file in files:
        path = os.path.join(base_path, file)
        metrics_plagiarism = calculate_all(path)
        metrics_author = calculate_all(path, "author")
        
        print(f"File: {file}")
        print("Plagiarism")
        print(metrics_plagiarism)
        print("Author")
        print(metrics_author)
        print()
        
        name = file[:-len(".jsonl.vllm_n=100")]
        with open(f"./metrics/{name}_plagiarism.json", "w") as f:
            json.dump(metrics_plagiarism, f)
        with open(f"./metrics/{name}_author.json", "w") as f:
            json.dump(metrics_author, f)