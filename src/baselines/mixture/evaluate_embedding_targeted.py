
import json
import os
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
tqdm.pandas()

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

def calculate_all(df, model_name="luar"):

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

    author_id_x = np.array(df.author_id_x.tolist())
    author_id_y = np.array(df.author_id_y.tolist())
    labels = (author_id_x == author_id_y).astype(int).tolist()
        
    metrics = {}
        
    query_author_embeddings = torch.cat(
            [author_fn(unit) for unit in tqdm(df.unit_y.tolist())],
            dim=0,
        ).cpu()
        
    ##### Inverse (All):
    inverse_all_author_embeddings = torch.cat(
            [author_fn(flatten(inverse)) for inverse in tqdm(df.inverse.tolist())],
            dim=0,
        ).cpu()
    metrics["inverse_all"] = calculate_similarities_and_metrics(
            query_author_embeddings,
            inverse_all_author_embeddings,
            labels,
            diagonal=True,
        )

    ##### Inverse (Individual Embedding):
    inverse_author_embeddings = [[instance_fn(inverse).cpu() for inverse in inverses] for inverses in tqdm(df.inverse.tolist())]
    inverse_author_embeddings = [torch.cat(inverses, dim=0) for inverses in inverse_author_embeddings]
    for simtype in ["max", "expected"]:
        metrics[f"inverse_{simtype}"] = calculate_similarities_and_metrics(
                query_author_embeddings,
                inverse_author_embeddings,
                labels,
                simtype,
                diagonal=True,
            )

    ##### Inverse (Single):
    df["inverse_single"] = df.inverse.apply(lambda xx: [x[0] for x in xx])
    inverse_single_author_embeddings = torch.cat(
            [author_fn(inverse).cpu() for inverse in tqdm(df.inverse_single.tolist())],
            dim=0,
        )
    metrics["inverse_single"] = calculate_similarities_and_metrics(
            query_author_embeddings,
            inverse_single_author_embeddings,
            labels,
            diagonal=True,
        )
    
    return metrics

if __name__ == "__main__":
    debug = False
    model_names = ["luar", "sbert", "cisr"]
    base_path = "/data1/yubnub/changepoint/MUD_inverse/data/data.jsonl.filtered.cleaned_kmeans_100/inverse_output"
    files = [
        "none_targetted=examples_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=5.targetted_mode=author_num_examples=1",
        "none_targetted=examples_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=5.targetted_mode=author_num_examples=2",
        "none_targetted=examples_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=5.targetted_mode=author_num_examples=3",
        "none_targetted=examples_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=5.targetted_mode=author",
        "none_targetted_6400_temperature=0.7_top_p=0.9.jsonln=5.targetted_mode=author",
    ]
    
    for file in files:
        path = os.path.join(base_path, file)
        
        df = pd.read_json(path, lines=True)
        df = df.groupby(["author_id_x", "author_id_y"]).agg(list).reset_index()
        df.unit_y = df.unit_y.apply(lambda x: x[0])
        if debug:
            df = df.iloc[:500]
            
        for model in model_names:
            metrics = calculate_all(df, model)

            name = file.replace(".jsonl", "")
            name += f"_{model}"
            with open(f"./metrics/{name}_author.json", "w") as f:
                json.dump(metrics, f)