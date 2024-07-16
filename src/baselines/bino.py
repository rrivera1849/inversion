
import os
import sys
from argparse import ArgumentParser
from collections import Counter
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from binoculars import Binoculars
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer
from termcolor import colored
from tqdm import tqdm

sys.path.append("../../scripts/dataset/changepoint")
from prompts import PROMPT_NAMES

parser = ArgumentParser()
parser.add_argument("--dirname", type=str,
                    default="/data1/yubnub/changepoint/s2orc_changepoint/unit_128",
                    help="Directory where the dataset is stored.")
parser.add_argument("--debug", default=False, action="store_true",
                    help="If True, will process only a few samples.")
args = parser.parse_args()

# RRS - Things that are getting copied everywhere:
# - dirname
# - MODEL_NAMES
MODEL_NAMES = [
    "Mistral-7B-Instruct-v0.3",
    "Meta-Llama-3-8B-Instruct",
    "Phi-3-mini-4k-instruct"
]

def get_bino_scores(
    text: list[str],
    batch_size: int = 32,
) -> list[float]:
    """Compute Bino scores for a list of text
    """
    bino = Binoculars()
    scores = []
    for i in tqdm(range(0, len(text), batch_size)):
        batch = text[i:i+batch_size]
        scores += bino.compute_score(batch)
    return scores

@torch.no_grad()
def get_LUAR_scores(
    text: list[str],
    fewshot_humans: list[str],
    batch_size: int = 32,
) -> list[float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    HF_id = "rrivera1849/LUAR-MUD"
    model = AutoModel.from_pretrained(HF_id, trust_remote_code=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(HF_id)

    tokenized_fewshot = tokenizer(
        fewshot_humans, 
        max_length=512,
        padding="max_length", 
        truncation=True, 
        return_tensors="pt",
    )
    tokenized_fewshot = {k: v.unsqueeze(0).to(device) for k, v in tokenized_fewshot.items()}

    fewshot_emb = model(**tokenized_fewshot)
    fewshot_emb = F.normalize(fewshot_emb, p=2.0, dim=-1)

    scores = []
    for i in tqdm(range(0, len(text), batch_size)):
        batch = text[i:i+batch_size]

        tokenized_batch = tokenizer(
            batch, 
            max_length=512,
            padding="max_length", 
            truncation=True, 
            return_tensors="pt",
        )
        tokenized_batch = {k: v.unsqueeze(1).to(device) for k, v in tokenized_batch.items()}

        batch_emb = model(**tokenized_batch)
        batch_emb = F.normalize(batch_emb, p=2.0, dim=-1)
        assert batch_emb.size(0) <= batch_size
        scores.extend(F.cosine_similarity(fewshot_emb, batch_emb).cpu().numpy().tolist())

    return scores


def compute_metrics(
    scores: list[float], 
    models: list[str], 
    prompts: list[Union[None, str]],
) -> dict[str, float]:
    assert len(scores) == len(models) == len(prompts)

    metrics: dict[str, float] = {}

    max_fpr = 0.01
    human_scores = [score for i, score in enumerate(scores) if models[i] == "human"]

    machine_scores = [score for i, score in enumerate(scores) if models[i] != "human"]
    labels = [0] * len(human_scores) + [1] * len(machine_scores)
    roc_auc = roc_auc_score(labels, human_scores + machine_scores, max_fpr=max_fpr)
    metrics["global AUC(0.01)"] = roc_auc

    for prompt in PROMPT_NAMES:
        prompt_machine_scores = [score for i, score in enumerate(scores) if models[i] != "human" and prompts[i] == prompt]
        prompt_labels = [0] * len(human_scores) + [1] * len(prompt_machine_scores)
        roc_auc = roc_auc_score(prompt_labels, human_scores + prompt_machine_scores, max_fpr=max_fpr)
        metrics[prompt + " AUC(0.01)"] = roc_auc
        
    return metrics

def main():
    dataset = load_from_disk(os.path.join(args.dirname, "MTD_dataset"))

    # set of fewshot examples for LUAR:
    fewshot_humans: list[str] = []
    num_fewshot = 10

    if args.debug:
        N = 50
        total_humans = N * len(MODEL_NAMES) * len(PROMPT_NAMES)
        print(colored("DEBUG: Subsampling dataset...", "yellow"))

        counts = Counter()
        indices_to_keep = []

        for i in range(len(dataset)):
            elem = dataset[i]
            if elem["model"] == "human" and counts[("human", None)] < total_humans + num_fewshot:
                if counts[("human", None)] < total_humans:
                    indices_to_keep.append(i)
                else:
                    fewshot_humans.append(elem["text"])

                counts[("human", None)] += 1
            elif counts[(elem["model"], elem["prompt"])] < N:
                indices_to_keep.append(i)
                counts[(elem["model"], elem["prompt"])] += 1

            if sum(counts.values()) == 2 * total_humans + num_fewshot:
                break

        assert len(indices_to_keep) == 2 * total_humans
        dataset = dataset.select(indices_to_keep)
    else:
        length_before = len(dataset)
        indices_to_remove = []
        for i, elem in enumerate(dataset):
            if elem["model"] == "human" and len(fewshot_humans) < num_fewshot:
                fewshot_humans.append(elem["text"])
                indices_to_remove.append(i)

            if len(fewshot_humans) == num_fewshot:
                break
        
        dataset = dataset.select([i for i in range(len(dataset)) if i not in indices_to_remove])
        assert len(dataset) == length_before - num_fewshot
            
    assert len(fewshot_humans) == num_fewshot

    texts = [elem["text"] for elem in dataset]
    models = [elem["model"] for elem in dataset]
    prompts = [elem["prompt"] for elem in dataset]

    scores = get_LUAR_scores(texts, fewshot_humans)
    metrics = compute_metrics(scores, models, prompts)
    print()
    print(colored("LUAR metrics:", "blue"))
    for k, v in metrics.items():
        print(colored(f"\t{k}: {v:.4f}", "green"))

    scores = get_bino_scores(texts)
    metrics = compute_metrics(scores, models, prompts)
    print()
    print(colored("Binocular metrics:", "blue"))
    for k, v in metrics.items():
        print(colored(f"\t{k}: {v:.4f}", "green"))

    return 0

if __name__ == "__main__":
    sys.exit(main())