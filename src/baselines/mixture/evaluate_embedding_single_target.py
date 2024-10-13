# Untargeted Models
# Targeted Models on eval_all

import json
import os
from argparse import ArgumentParser
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--filename", type=str, default=None, required=True)
parser.add_argument("--dataset_name", type=str, default="data.jsonl.filtered.cleaned_kmeans_100")
parser.add_argument("--model_name", type=str, default="crud", 
                    choices=["mud", "crud", "cisr", "sbert"])
parser.add_argument("--mode", type=str, default=["plagiarism"], nargs="+")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

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
    
    if "unit_y" in df.columns:
        df.rename(columns={"original_unit": "unit", "rephrase_x": "rephrase"}, inplace=True)
        df.drop(columns=["unit_y"], inplace=True)

    if "author_id" not in df.columns:
        # This is all very unfortunate, but I forgot to save the `author_id` in the 
        # small test set I saved, here we are recovering it:
        path_test_full = f"/data1/yubnub/changepoint/MUD_inverse/data/{args.dataset_name}/test.jsonl"
        df_test_full = pd.read_json(path_test_full, lines=True)
        df_test_full = df_test_full.groupby("author_id").agg(list).iloc[:100]
        to_explode = [col for col in df_test_full.columns if col != "author_id"]
        df_test_full = df_test_full.explode(to_explode).reset_index()
        if args.filename.endswith(".gpt4") and args.dataset_name == "data.jsonl.filtered.cleaned_kmeans_100":
            df_test_full = df_test_full.drop_duplicates("unit")
            df.unit = df.unit.apply(lambda x: x[:-1])
        assert sorted(df_test_full.unit.tolist()) == sorted(df.unit.tolist())
        author_ids = df_test_full.author_id.tolist()
        df["author_id"] = author_ids

    # to sample random LLMs, should be around the same proportion
    df = df.sample(frac=1., random_state=43)
    df.drop_duplicates("unit", inplace=True)
    return df

def calculate_all(
    path: str,
    mode: str = "plagiarism",
    model_name: str = "luar",
    debug: bool = False,
):
    assert mode in ["plagiarism", "author"]
    assert model_name in ["mud", "crud", "cisr", "sbert"]

    if model_name == "mud" or model_name == "crud":
        HF_identifier = "rrivera1849/LUAR-CRUD" if model_name == "crud" else "rrivera1849/LUAR-MUD"
        luar, luar_tok = load_luar_model_and_tokenizer(HF_identifier)
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

    if mode == "author":
        # Compute Author Query
        query_author_embeddings = torch.cat(
            [author_fn(unit) for unit in tqdm(df.unit.tolist())],
            dim=0,
        ).cpu()
    else:
        # Compute Instance Query
        query_instance_embeddings = torch.cat(
            [instance_fn(unit) for unit in tqdm(df.unit.tolist())],
            dim=0
        ).cpu()

    ##### Rephrases:
    if mode == "plagiarism":
        rephrase_instance_embeddings = torch.cat(
            [instance_fn(rephrases) for rephrases in tqdm(df.rephrase.tolist())],
            dim=0
        ).cpu()
        assert len(rephrase_instance_embeddings) == num_instances
    else:
        rephrase_author_embeddings = torch.cat(
            [author_fn(rephrases) for rephrases in tqdm(df.rephrase.tolist())],
            dim=0,
        ).cpu()
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
    if mode == "author":
        inverse_all_author_embeddings = torch.cat(
            [author_fn(flatten(inverses)) for inverses in tqdm(df.inverse.tolist())],
            dim=0,
        ).cpu()
        assert len(inverse_all_author_embeddings) == num_author
    else:
        inverse_all_instance_embeddings = [[author_fn(inverse).cpu() for inverse in inverses] for inverses in tqdm(df.inverse.tolist())]
        inverse_all_instance_embeddings = torch.cat(
            [torch.cat(inverses, dim=0) for inverses in inverse_all_instance_embeddings],
            dim=0
        ).cpu()
        assert len(inverse_all_instance_embeddings) == num_instances

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
    if mode == "plagiarism":
        inverse_single_instance_embeddings = [instance_fn(inverses) for inverses in tqdm(df.single_inversion.tolist())]
        inverse_single_instance_embeddings = torch.cat(inverse_single_instance_embeddings, dim=0).cpu()
        assert len(inverse_single_instance_embeddings) == num_instances
    else:
        inverse_single_author_embeddings = [author_fn(inverses) for inverses in tqdm(df.single_inversion.tolist())]
        inverse_single_author_embeddings = torch.cat(inverse_single_author_embeddings, dim=0).cpu()
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
    metric_dir = "./metrics/new"
    os.makedirs(metric_dir, exist_ok=True)
    base_path = "/data1/yubnub/changepoint/MUD_inverse/data"
    filename = os.path.join(base_path, args.dataset_name, "inverse_output", args.filename)

    if "plagiarism" in args.mode:
        print("Calculating Plagiarism Metrics")
        metrics_plagiarism = calculate_all(filename, mode="plagiarism", model_name=args.model_name, debug=args.debug)
        plagiarism_savename = os.path.join(metric_dir, args.dataset_name, "plagiarism", args.model_name)
        os.makedirs(plagiarism_savename, exist_ok=True)
        with open(os.path.join(plagiarism_savename, args.filename), "w") as f:
            json.dump(metrics_plagiarism, f)
        
    if "author" in args.mode:
        print("Calculating Author Metrics")
        metrics_author = calculate_all(filename, mode="author", model_name=args.model_name, debug=args.debug)
        author_savename = os.path.join(metric_dir, args.dataset_name, "author", args.model_name)
        os.makedirs(author_savename, exist_ok=True)
        with open(os.path.join(author_savename, args.filename), "w") as f:
            json.dump(metrics_author, f)