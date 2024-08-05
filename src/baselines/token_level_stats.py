"""Here to calculate the Token-Wise perplexity of a 
   base Mistral model, and a regular Mistral model.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from fast_detect_gpt import get_sampling_discrepancy_analytic
from utils import load_model, MODELS

@torch.no_grad()
def get_token_probs(
    text: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch_size: int = 32
) -> list[list[int]]:
    probabilities = []
    
    for i in range(0, len(text), batch_size):
        batch = text[i:i+batch_size]
        batch = tokenizer(
            batch,
            max_length=128+32,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = model(**batch).logits
        out = F.log_softmax(out, dim=-1)
        
        mask = batch["input_ids"] != tokenizer.pad_token_id
        instance_lengths = mask.sum(dim=1).tolist()
        
        instance_logits = torch.split(out[mask], instance_lengths)
        instance_labels = torch.split(batch["input_ids"][mask], instance_lengths)
        for logits, labels in zip(instance_logits, instance_labels):
            logits = logits[:-1, :]
            labels = labels[1:]
            probs = logits[torch.arange(len(labels)), labels]
            probs = probs.exp().tolist()
            probabilities.append(probs)
    
    return probabilities

@torch.no_grad()
def get_fast_detect_gpt_scores(
    text: list[str],
    reference_model_id: str = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model, base_tok = load_model()
    if reference_model_id is not None:
        reference_model, reference_tok = load_model(reference_model_id)

    scores = []
    for sample in tqdm(text):
        tok = base_tok(
            sample, 
            padding=True,
            truncation=True, 
            return_tensors="pt", 
        ).to(device)
        base_logits = base_model(**tok).logits[:, :-1]
        labels = tok["input_ids"][:, 1:]
        
        if reference_model_id is not None:
            tok = reference_tok(
                sample, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
            ).to(device)
            reference_logits = reference_model(**tok).logits[:, :-1]
        else:
            reference_logits = base_logits
        
        discrepancy = get_sampling_discrepancy_analytic(reference_logits, base_logits, labels, reduce=False)
        discrepancy = discrepancy.cpu().squeeze().tolist()
        scores.append(discrepancy)
    return scores

def main():
    N = 100
    seed = 43
    dataset = pd.read_json("./mixture/mixture_dataset_tokenizer=Mistral-7B-v0.3_num-samples=1000.jsonl", lines=True)
    dataset = dataset.sample(frac=1., random_state=seed).reset_index(drop=True)
    dataset = dataset.iloc[:2*N]
    
    # savenames = ["human", "rephrase", "mistral"]
    # model_id_or_paths = [MODELS["human"], MODELS["rephrase"], "mistralai/Mistral-7B-v0.3"]
    savenames = ["mistral"]
    model_id_or_paths = ["mistralai/Mistral-7B-v0.3"]
    i = 0
    for model_id_or_path in model_id_or_paths:
        # model, tokenizer = load_model(model_id_or_path)
        text = dataset["text"].tolist()
        tagger_labels = dataset["tagger_labels"].tolist()
        # probs = get_token_probs(text, model, tokenizer)

        probs = get_fast_detect_gpt_scores(text)
        human_probs = []
        machine_probs = []
        for prob, tlabels in zip(probs, tagger_labels):
            human_probs.extend([p for p, t in zip(prob, tlabels) if t == 0])
            machine_probs.extend([p for p, t in zip(prob, tlabels) if t == 1])

        N = min(len(human_probs), len(machine_probs))
        human_probs = human_probs[:N]
        machine_probs = machine_probs[:N]
        
        all_probs = human_probs + machine_probs
        labels = [0]*len(human_probs) + [1]*len(machine_probs)
        auc = roc_auc_score(labels, all_probs)
        print(f"AUC: {auc}")

        fpr, tpr, thresholds = roc_curve(labels, all_probs)
        # Youden J's Statistic: https://en.wikipedia.org/wiki/Youden%27s_J_statistic
        # https://www.kaggle.com/code/nicholasgah/obtain-optimal-probability-threshold-using-roc#
        roc_t = sorted(list(zip(np.abs(tpr - fpr), thresholds)), key=lambda i: i[0], reverse=True)[0][1]
        print(classification_report(labels, [p < roc_t for p in all_probs]))
        
        # import pdb; pdb.set_trace()
        _ = plt.figure()
        plt.hist(human_probs, bins=50, alpha=0.5, label="Human")
        plt.hist(machine_probs, bins=50, alpha=0.5, label="Machine")
        plt.legend()
        plt.savefig(f"./token_probs_{savenames[i]}.png")
        plt.close()
        i += 1

    return 0

if __name__ == "__main__":
    sys.exit(main())