
import json
import os
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from embedding_utils import (
    load_cisr_model,
    load_luar_model_and_tokenizer,
    load_sbert_model,
    get_author_embeddings,
    get_instance_embeddings,
)
from metric_utils import calculate_similarities_and_metrics

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

def calculate_all(
    path: str,
    mode: str = "plagiarism",
    model_name: str = "luar",
    debug: bool = False,
):
    assert mode in ["plagiarism", "author"]
    assert model_name in ["luar", "cisr", "sbert"]

    if model_name == "luar":
        luar, luar_tok = load_luar_model_and_tokenizer()
        luar = luar.to("cuda")
        function_kwargs = {
            "luar": luar,
            "luar_tok": luar_tok,
            "normalize": True,
        }
    elif model_name == "cisr":
        cisr = load_cisr_model()
        cisr = cisr.to("cuda")
        function_kwargs = {
            "model": cisr,
            "normalize": True,
        }
    elif model_name == "sbert":
        sbert = load_sbert_model()
        sbert = sbert.to("cuda")
        function_kwargs = {
            "model": sbert,
            "normalize": True,
        }

    author_fn = partial(get_author_embeddings, function_kwargs=function_kwargs, model_name=model_name)
    instance_fn = partial(get_instance_embeddings, function_kwargs=function_kwargs, model_name=model_name)
    
    metrics = {}
    df = read_data(path)
    df = df.groupby("author_id").agg(list)
    
    if mode == "author":
        # query != target
        # task: author identification
        df["unit"] = df["unit"].apply(lambda x: x[:len(x)//2])
        df["rephrase"] = df["rephrase"].apply(lambda x: x[len(x)//2:])
        df["inverse"] = df["inverse"].apply(lambda x: x[len(x)//2:])
        
    if debug:
        df = df.head(10)

    num_author = len(df)
    num_instances = df.rephrase.apply(len).sum()

    author_author_labels = np.identity(num_author, dtype=np.int32).flatten().tolist()
    instance_instance_labels = np.identity(num_instances, dtype=np.int32).flatten().tolist()

    # Compute Author Query
    query_author_embeddings = torch.cat(
        [author_fn(unit) for unit in tqdm(df.unit.tolist())],
        dim=0,
    ).cpu()

    # Compute Instance Query
    query_instance_embeddings = torch.cat(
        [instance_fn(unit) for unit in tqdm(df.unit.tolist())],
        dim=0
    ).cpu()

    ##### Rephrases:
    rephrase_instance_embeddings = torch.cat(
        [instance_fn(rephrases) for rephrases in tqdm(df.rephrase.tolist())],
        dim=0
    ).cpu()
    rephrase_author_embeddings = torch.cat(
        [author_fn(rephrases) for rephrases in tqdm(df.rephrase.tolist())],
        dim=0,
    ).cpu()
    assert len(rephrase_instance_embeddings) == num_instances
    assert len(rephrase_author_embeddings) == num_author

    if mode == "plagiarism":
        # Instance-Instance:
        metrics["rephrase"] = calculate_similarities_and_metrics(
            query_instance_embeddings,
            rephrase_instance_embeddings,
            instance_instance_labels,
        )
    else:
        # Author-Author
        metrics["rephrase"] = calculate_similarities_and_metrics(
            query_author_embeddings,
            rephrase_author_embeddings,
            author_author_labels,
        )

    ##### Inverse (All):
    inverse_all_author_embeddings = torch.cat(
        [author_fn(flatten(inverses)) for inverses in tqdm(df.inverse.tolist())],
        dim=0,
    ).cpu()
    inverse_all_instance_embeddings = [[author_fn(inverse).cpu() for inverse in inverses] for inverses in tqdm(df.inverse.tolist())]
    inverse_all_instance_embeddings = torch.cat(
        [torch.cat(inverses, dim=0) for inverses in inverse_all_instance_embeddings],
        dim=0
    ).cpu()
    assert len(inverse_all_instance_embeddings) == num_instances
    assert len(inverse_all_author_embeddings) == num_author

    if mode == "plagiarism":
        # Instance-Instance:
        metrics["inverse_all"] = calculate_similarities_and_metrics(
            query_instance_embeddings,
            inverse_all_instance_embeddings,
            instance_instance_labels,
        )
    else:
        # Author-Author
        metrics["inverse_all"] = calculate_similarities_and_metrics(
            query_author_embeddings,
            inverse_all_author_embeddings,
            author_author_labels,
        )
        
    ##### Inverse (Individual Embedding):
    inverse_instance_embeddings = [[instance_fn(inverse).cpu() for inverse in inverses] for inverses in tqdm(df.inverse.tolist())]
    inverse_author_embeddings = [torch.cat(inverses, dim=0) for inverses in inverse_instance_embeddings]
    inverse_instance_embeddings = flatten(inverse_instance_embeddings)
    assert len(inverse_instance_embeddings) == num_instances
    assert len(inverse_author_embeddings) == num_author
    for simtype in ["expected", "max"]:
        if mode == "plagiarism":
            metrics[f"inverse_{simtype}"] = calculate_similarities_and_metrics(
                query_instance_embeddings,
                inverse_instance_embeddings,
                instance_instance_labels,
                simtype,
            )
        else:
            metrics[f"inverse_{simtype}"] = calculate_similarities_and_metrics(
                query_author_embeddings,
                inverse_author_embeddings,
                author_author_labels,
                simtype,
            )
 
    ##### Inverse (Single Inversion):
    df["single_inversion"] = df.inverse.apply(lambda xx: [x[0] for x in xx])
    inverse_single_instance_embeddings = [instance_fn(inverses) for inverses in tqdm(df.single_inversion.tolist())]
    inverse_single_instance_embeddings = torch.cat(inverse_single_instance_embeddings, dim=0).cpu()
    inverse_single_author_embeddings = [author_fn(inverses) for inverses in tqdm(df.single_inversion.tolist())]
    inverse_single_author_embeddings = torch.cat(inverse_single_author_embeddings, dim=0).cpu()
    assert len(inverse_single_instance_embeddings) == num_instances
    assert len(inverse_single_author_embeddings) == num_author

    if mode == "plagiarism":
        # Instance-Instance:
        metrics["inverse_single"] = calculate_similarities_and_metrics(
            query_instance_embeddings,
            inverse_single_instance_embeddings,
            instance_instance_labels,
        )
    else:
        # Author-Author
        metrics["inverse_single"] = calculate_similarities_and_metrics(
            query_author_embeddings,
            inverse_single_author_embeddings,
            author_author_labels,
        )
        
    return metrics

if __name__ == "__main__":
    os.makedirs("./metrics", exist_ok=True)
    base_path = "/data1/yubnub/changepoint/MUD_inverse/data/data.jsonl.filtered.cleaned_kmeans_100/inverse_output"
    model_names = ["luar", "sbert", "cisr"]
    files = [
        "none_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=100",
        "none_6400_temperature=0.5_top_p=0.9.jsonl.vllm_n=100",
        "none_6400_temperature=0.6_top_p=0.9.jsonl.vllm_n=100",
        "none_6400_temperature=0.8_top_p=0.9.jsonl.vllm_n=100",
        "none_6400_temperature=0.9_top_p=0.9.jsonl.vllm_n=100",
        "none_6400_temperature=0.3_top_p=0.9.jsonl.vllm_n=100",
        "none_6400_temperature=1.5_top_p=0.9.jsonl.vllm_n=100",
    ]
    for model in model_names:
        for file in files:
            print(f"File: {file}")
            path = os.path.join(base_path, file)
            metrics_plagiarism = calculate_all(path, mode="plagiarism", model_name=model)
            metrics_author = calculate_all(path, mode="author", model_name=model)
            
            name = file[:-len(".jsonl.vllm_n=100")] + f"_{model}"
            with open(f"./metrics/{name}_plagiarism.json", "w") as f:
                json.dump(metrics_plagiarism, f)
            with open(f"./metrics/{name}_author.json", "w") as f:
                json.dump(metrics_author, f)