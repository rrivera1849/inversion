
import os
import random; random.seed(43)
from typing import Union

import pandas as pd
import torch
from sentence_transformers import util
from sklearn.metrics import roc_auc_score
from termcolor import colored
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from mixture.embedding_utils import *

INVERSE_DATA_PATH = "/data1/yubnub/changepoint/MUD_inverse/data"

def load_machine_paraphrase_data(
    filename: str = "none_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=100",
    pick_dissimilar: bool = False,
    debug: bool = False,
):
    data = {}
    total = 1_000 if not debug else 10
    
    human_fname = os.path.join(
        INVERSE_DATA_PATH,
        "../raw",
        "data.jsonl.filtered.cleaned",
    )
    df_human = pd.read_json(human_fname, lines=True, nrows=total)
    human = df_human.unit.tolist()
    human = [j for i in human for j in i]
    random.shuffle(human)
    human = human[:total]
    
    fname = os.path.join(
        INVERSE_DATA_PATH,
        "data.jsonl.filtered.respond_reddit.cleaned",
        "inverse_output",
        filename,
    )
    # "unit" is response to a reddit comment
    # "rephrase" is a paraphrase of "unit"
    # "inverse" is the inverse of "rephrase"
    df = pd.read_json(fname, lines=True)
    df = df.sample(frac=1., random_state=43).reset_index(drop=True)
    df = df.iloc[:total]
    
    machine = df.rephrase.tolist()
    texts = human + machine
    models = ["human"] * len(human) + df.model_name.tolist()
    data["without_inverse"] = {
        "texts": texts,
        "models": models,
    }

    if pick_dissimilar:
        inverse = df.inverse.tolist()
        luar, luar_tok = load_luar_model_and_tokenizer("rrivera1849/LUAR-CRUD")
        luar.to("cuda")
        embeddings_inverse = [get_luar_instance_embeddings(inv, luar, luar_tok, normalize=True) for inv in tqdm(inverse)]
        embeddings_rephrase = get_luar_instance_embeddings(df.rephrase.tolist(), luar, luar_tok, normalize=True)

        indices = []
        for j in range(total):
            emb = embeddings_rephrase[j:j+1]
            emb = emb.repeat(embeddings_inverse[j].size(0), 1)
            similarities = util.pytorch_cos_sim(emb, embeddings_inverse[j]).diag().cpu().tolist()
            minsim = min(similarities)
            min_index = similarities.index(minsim)
            indices.append(min_index)

        inverse = [inverse[i][indices[i]] for i in range(total)]
    else:
        inverse = df.inverse.tolist()
        inverse = [j for i in inverse for j in i]
        inverse = random.sample(inverse, len(human))

    texts = human + inverse
    models = ["human"] * len(human) + ["machine"] * len(inverse)
    data["with_inverse"] = {
        "texts": texts,
        "models": models,
    }

    return data

def load_human_paraphrase_data(
    filename: str = "none_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=100",
    debug: bool = False
):
    fname = os.path.join(
        INVERSE_DATA_PATH,
        "data.jsonl.filtered.respond_reddit.cleaned",
        "inverse_output",
        filename,
    )
    total = 1_000 if not debug else 10
    df = pd.read_json(fname, lines=True)
    df = df.sample(frac=1., random_state=43).reset_index(drop=True)
    df = df.iloc[:total]
    
    data = {}
    # unit = human
    # rephrase = paraphrase of human
    # inverse = inverse of rephrase
    human = df.unit.tolist()
    rephrase = df.rephrase.tolist()
    texts = human + rephrase
    models = ["human"] * len(human) + df.model_name.tolist()
    data["without_inverse"] = {
        "texts": texts,
        "models": models,
    }

    inverse = df.inverse.tolist()
    inverse = [j for i in inverse for j in i]
    inverse = random.sample(inverse, len(human))
    texts = human + inverse
    models = ["human"] * len(human) + ["machine"] * len(inverse)
    data["with_inverse"] = {
        "texts": texts,
        "models": models,
    }

    return data

def load_model(
    model_id_or_path: str = "mistralai/Mistral-7B-v0.3",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path,
        )
    except:
        print(colored("WARNING: Using Mistral tokenizer...", "yellow"))
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-v0.3",
        )
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer
    
def compute_metrics(
    scores: list[float], 
    models: list[str], 
    max_fpr: float = 0.01
) -> dict[str, float]:

    metrics: dict[str, float] = {}

    human_scores = [score for i, score in enumerate(scores) if models[i] == "human"]
    machine_scores = [score for i, score in enumerate(scores) if models[i] != "human"]

    labels = [0] * len(human_scores) + [1] * len(machine_scores)
    metrics[f"global AUC({max_fpr})"] = roc_auc_score(labels, human_scores + machine_scores, max_fpr=max_fpr)
    return metrics