
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from sentence_transformers import util
from sklearn.metrics import roc_curve, roc_auc_score
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

luar = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
luar.to("cuda").eval()
luar_tok = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)

def cosine_similarity(
    embeddings_1: torch.Tensor,
    embeddings_2: torch.Tensor,
):
    cosine_scores = util.pytorch_cos_sim(embeddings_1, embeddings_2)
    return cosine_scores

@torch.no_grad()
def get_luar_embeddings(
    text: str,
    batch_size: int = 32,
):
    all_outputs = []
    for i in range(0, len(text), batch_size):
        batch = text[i:i+batch_size]
        inputs = luar_tok(
            batch,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(luar.device)
        inputs["input_ids"] = inputs["input_ids"].view(len(batch), 1, 512)
        inputs["attention_mask"] = inputs["attention_mask"].view(len(batch), 1, 512)
        outputs = luar(**inputs)
        all_outputs.append(outputs)
    all_outputs = torch.cat(all_outputs, dim=0)
    return all_outputs

@torch.no_grad()
def get_author_embeddings(
    text: list[str],
):
    assert isinstance(text, list)
    inputs = luar_tok(
        text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(luar.device)
    inputs["input_ids"] = inputs["input_ids"].view(1, len(text), 512)
    inputs["attention_mask"] = inputs["attention_mask"].view(1, len(text), 512)
    outputs = luar(**inputs)
    return outputs

def calculate_luar_similarity(
    candidates: list[str],
    references: list[str],
):
    embeddings1 = get_luar_embeddings(candidates)
    embeddings2 = get_luar_embeddings(references)
    return cosine_similarity(embeddings1, embeddings2)

def read_data(path: str):
    df = pd.read_json(path, lines=True)

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

## AUC at Rephrase / Inverse Level:
def calculate_metrics(similarities):
    labels = []
    sims = []
    for i in range(len(similarities)):
        sims.extend(similarities[i].tolist())
        l = [0] * len(similarities[i])
        l[i] = 1
        labels.extend(l)
    AUC = roc_auc_score(labels, sims, max_fpr=0.01)

    fpr, tpr, _ = roc_curve(labels, sims, pos_label=1)
    fnr = 1 - tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    return AUC, EER

def process(path: str):
    metrics = {}
    df = read_data(path)
    df = df.groupby("author_id").agg(list)

    unit_embeddings = [get_author_embeddings(units) for units in df.unit.tolist()]
    rephrase_embeddings = [get_author_embeddings(rephrases) for rephrases in df.rephrase.tolist()]
    inverse_embeddings = [get_author_embeddings(inverses) for inverses in df.inverse.tolist()]

    unit_embeddings = torch.cat(unit_embeddings, dim=0)
    rephrase_embeddings = torch.cat(rephrase_embeddings, dim=0)
    inverse_embeddings = torch.cat(inverse_embeddings, dim=0)

    unit_rephrase_similarities = util.pytorch_cos_sim(unit_embeddings, rephrase_embeddings)
    unit_inverse_similarities = util.pytorch_cos_sim(unit_embeddings, inverse_embeddings)

    AUC_rephrase, EER_rephrase = calculate_metrics(unit_rephrase_similarities)
    AUC_inverse, EER_inverse = calculate_metrics(unit_inverse_similarities)

    metrics["easy"] = {
        "AUC_rephrase": AUC_rephrase,
        "EER_rephrase": EER_rephrase,
        "AUC_inverse": AUC_inverse,
        "EER_inverse": EER_inverse,
    }

    ## AUC at Author Level (Hard Case where we don't observe the rephrases for the units):

    unit_embeddings = [get_author_embeddings(units[:len(units)//2]) for units in df.unit.tolist()]
    rephrase_embeddings = [get_author_embeddings(rephrases[len(rephrases)//2:]) for rephrases in df.rephrase.tolist()]
    inverse_embeddings = [get_author_embeddings(inverses[len(inverses)//2:]) for inverses in df.inverse.tolist()]

    unit_embeddings = torch.cat(unit_embeddings, dim=0)
    rephrase_embeddings = torch.cat(rephrase_embeddings, dim=0)
    inverse_embeddings = torch.cat(inverse_embeddings, dim=0)

    unit_rephrase_similarities = util.pytorch_cos_sim(unit_embeddings, rephrase_embeddings)
    unit_inverse_similarities = util.pytorch_cos_sim(unit_embeddings, inverse_embeddings)

    AUC_rephrase, EER_rephrase = calculate_metrics(unit_rephrase_similarities)
    AUC_inverse, EER_inverse = calculate_metrics(unit_inverse_similarities)
    
    metrics["hard"] = {
        "AUC_rephrase": AUC_rephrase,
        "EER_rephrase": EER_rephrase,
        "AUC_inverse": AUC_inverse,
        "EER_inverse": EER_inverse,
    }
    
    return metrics

def get_oracle_hard(path: str):
    df = read_data(path)
    df = df.groupby("author_id").agg(list)
    
    unit_embeddings = [get_author_embeddings(units[:len(units)//2]) for units in df.unit.tolist()]
    unit_target_embeddings = [get_author_embeddings(units[len(units)//2:]) for units in df.unit.tolist()]

    unit_embeddings = torch.cat(unit_embeddings, dim=0)
    unit_target_embeddings = torch.cat(unit_target_embeddings, dim=0)

    similarities = util.pytorch_cos_sim(unit_embeddings, unit_target_embeddings)
    AUC_inverse, EER_inverse = calculate_metrics(similarities)
    return AUC_inverse, EER_inverse

base_path = "/data1/yubnub/changepoint/MUD_inverse/data/data.jsonl.filtered.cleaned_kmeans_100/inverse_output"
filenames = {
    "none":
        [
            # "none_1600_temperature=0.7_top_p=0.9.jsonl",
            # "none_3200_temperature=0.7_top_p=0.9.jsonl",
            # "none_4800_temperature=0.7_top_p=0.9.jsonl",
            # "none_6400_temperature=0.7_top_p=0.9.jsonl",
            "none_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=100",
        ],
    # "tokens":
    #     [
    #         # "tokens_3200_temperature=0.7_top_p=0.9.jsonl",
    #         # "tokens_6400_temperature=0.7_top_p=0.9.jsonl",
    #         # "tokens_9600_temperature=0.7_top_p=0.9.jsonl",
    #         # "tokens_12800_temperature=0.7_top_p=0.9.jsonl",
    #     ],
    # "probs":
    #     [
    #         # "probs_1600_temperature=0.7_top_p=0.9.jsonl",
    #         # "probs_3200_temperature=0.7_top_p=0.9.jsonl",
    #         # "probs_4800_temperature=0.7_top_p=0.9.jsonl",
    #         # "probs_6400_temperature=0.7_top_p=0.9.jsonl",
    #     ],
}

X = [0.5, 1.0, 1.5, 2.0]

rephrase = []
none_easy = []
none_hard = []
tokens_easy = []
tokens_hard = []
probs_easy = []
probs_hard = []

oracle_hard = None
for key, value in filenames.items():
    print(f"Processing {key}")
    
    for filename in tqdm(value):
        path = os.path.join(base_path, filename)
        metrics = process(path)
        if key == "none":
            none_easy.append(metrics["easy"]["EER_inverse"])
            none_hard.append(metrics["hard"]["EER_inverse"])
        elif key == "tokens":
            tokens_easy.append(metrics["easy"]["EER_inverse"])
            tokens_hard.append(metrics["hard"]["EER_inverse"])
        elif key == "probs":
            probs_easy.append(metrics["easy"]["EER_inverse"])
            probs_hard.append(metrics["hard"]["EER_inverse"])

        if not oracle_hard:
            oracle_hard = get_oracle_hard(path)

import pdb; pdb.set_trace()

# _ = plt.figure()
# rephrase_easy = metrics["easy"]["EER_rephrase"]
# plt.axhline(rephrase_easy, color="black", linestyle="--", label="Rephrase EER")
# plt.plot(X, none_easy, label="None")
# plt.plot(X, tokens_easy, label="Tokens")
# plt.plot(X, probs_easy, label="Probs")
# plt.xlabel("Epochs")
# plt.ylabel("EER")
# plt.ylim(0, 0.6)
# plt.legend()
# plt.xticks(X)
# plt.savefig("./easy_eer.png")
# plt.close()

# same for hard
# _ = plt.figure()
# rephrase_hard = metrics["hard"]["EER_rephrase"]
# plt.axhline(rephrase_hard, color="black", linestyle="--", label="Rephrase EER")
# plt.axhline(oracle_hard[1], color="purple", linestyle="--", label="Oracle EER")
# plt.plot(X, none_hard, label="None")
# plt.plot(X, tokens_hard, label="Tokens")
# plt.plot(X, probs_hard, label="Probs")
# plt.xlabel("Epochs")
# plt.ylabel("EER")
# plt.ylim(0, 0.6)
# plt.legend()
# plt.xticks(X)
# plt.savefig("./hard_eer.png")
# plt.close()