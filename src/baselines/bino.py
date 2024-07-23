
import os
import sys
from argparse import ArgumentParser
from collections import Counter
from typing import Union

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from binoculars import Binoculars
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from termcolor import colored
from tqdm import tqdm

sys.path.append("../../scripts/dataset/changepoint")
from prompts import PROMPT_NAMES

parser = ArgumentParser()
parser.add_argument("--dirname", type=str,
                    default="/data1/yubnub/changepoint/s2orc_changepoint/unit_128",
                    help="Directory where the dataset is stored.")
parser.add_argument("--M4_dataset_subset", type=str, default=None,
                    choices=["arxiv", "peerread", "reddit", "wikipedia", "wikihow"],
                    help="If provided, will use the M4 dataset.")
parser.add_argument("--batch_size", type=int, default=32)
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

def get_rank(text, base_model, base_tokenizer, log=False):
    """From: https://github.com/eric-mitchell/detect-gpt/blob/main/run.py#L298C1-L320C43
    """
    with torch.no_grad():
        tokenized = base_tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(base_model.device)
        logits = base_model(**tokenized).logits[:,:-1]
        labels = tokenized["input_ids"][:,1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:,-1], matches[:,-2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1 # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()

def get_bino_scores(
    text: list[str],
    batch_size: int = 32,
) -> list[float]:
    """Compute Bino scores for a list of text
    """
    bino = Binoculars(
        observer_name_or_path="tiiuae/falcon-7b",
        performer_name_or_path="tiiuae/falcon-7b-instruct",
    )
    scores = []
    for i in tqdm(range(0, len(text), batch_size)):
        batch = text[i:i+batch_size]
        scores += bino.compute_score(batch)
    return scores

@torch.no_grad()
def get_openai_detector_scores(
    text: list[str],
    batch_size: int = 32,
) -> list[float]:
    """Uses OpenAI's RoBERTa Detector for AI-Text Detection
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained("openai-community/roberta-base-openai-detector")
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("openai-community/roberta-base-openai-detector")

    probs = []
    for i in tqdm(range(0, len(text), batch_size)):
        batch = tokenizer(
            text[i:i+batch_size],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        batch = {k:v.to(device) for k,v in batch.items()}

        out = model(**batch)
        prob = out.logits.softmax(dim=-1)[:, 0].cpu().numpy().tolist()
        probs.extend(prob)
    return probs

def get_logrank_scores(
    text: list[str],
    model_id: str ="openai-community/gpt2-xl",
) -> list[float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    scores = []
    for i in tqdm(range(len(text))):
        try:
            scores.append(get_rank(text[i], model, tokenizer, log=True))
        except:
            scores.append(None)

    return scores

@torch.no_grad()
def get_LUAR_scores(
    text: list[str],
    fewshot_humans: list[str],
    batch_size: int = 32,
) -> list[float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # HF_id = "rrivera1849/LUAR-MUD"
    HF_id = "/data1/yubnub/pretrained_weights/LUAR/S2ORC_baseline_fast"
    print(colored("LUAR Weights: {}".format(HF_id), "green"))

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
    metrics["global AUC(0.01)"] = roc_auc_score(labels, human_scores + machine_scores, max_fpr=max_fpr)

    for prompt in PROMPT_NAMES:
        prompt_machine_scores = [score for i, score in enumerate(scores) if models[i] != "human" and prompts[i] == prompt]
        prompt_labels = [0] * len(human_scores) + [1] * len(prompt_machine_scores)
        metrics[prompt + " AUC(0.01)"] = roc_auc_score(prompt_labels, human_scores + prompt_machine_scores, max_fpr=max_fpr)
        
    return metrics

def evaluate_on_M4():
    print(colored(f"Evaluating on M4 dataset subset: {args.M4_dataset_subset}", "blue"))

    path = "/home/riverasoto1/repos/M4/data"
    subset = args.M4_dataset_subset

    filenames = [fname for fname in os.listdir(path) if subset in fname and "bloomz" not in fname]
    filenames = [os.path.join(path, fname) for fname in filenames]

    human_text, machine_text = [], []
    for fname in filenames:
        df = pd.read_json(fname, lines=True)
        human_text.extend(df["human_text"].tolist())
        machine_text.extend(df["machine_text"].tolist())
    human_text = list(set(human_text))
    machine_text = list(set(machine_text))

    if args.debug:
        human_text = human_text[:100]
        machine_text = machine_text[:100]

    print(colored(f"Number of human examples: {len(human_text)}", "yellow"))
    print(colored(f"Number of machine examples: {len(machine_text)}", "yellow"))

    scores = get_bino_scores(human_text + machine_text, batch_size=args.batch_size)
    scores = [-score for score in scores]
    # scores = get_logrank_scores(human_text + machine_text)
    labels = [0] * len(human_text) + [1] * len(machine_text)

    nan_value_indices = [i for i, score in enumerate(scores) if score != score or score is None]
    scores = [score for i, score in enumerate(scores) if i not in nan_value_indices]
    labels = [label for i, label in enumerate(labels) if i not in nan_value_indices]
    pauc = roc_auc_score(labels, scores, max_fpr=0.01)
    print(colored(f"PAUC(0.01): {pauc:.4f}", "green"))

def main():
    if args.M4_dataset_subset is not None:
        evaluate_on_M4()
        return 0
    
    # set of fewshot examples for LUAR:
    fewshot_humans: list[str] = []
    num_fewshot = 10

    dataset = load_from_disk(os.path.join(args.dirname, "MTD_dataset"))
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

    scores = get_bino_scores(texts, batch_size=args.batch_size)
    scores = [-score for score in scores]
    metrics = compute_metrics(scores, models, prompts)
    print()
    print(colored("Binocular metrics:", "blue"))
    for k, v in metrics.items():
        print(colored(f"\t{k}: {v:.4f}", "green"))

    scores = get_openai_detector_scores(texts, batch_size=args.batch_size)
    metrics = compute_metrics(scores, models, prompts)
    print()
    print(colored("OpenAI Detector metrics:", "blue"))
    for k, v in metrics.items():
        print(colored(f"\t{k}: {v:.4f}", "green"))

    # TODO: Add MSP as a Score

    return 0

if __name__ == "__main__":
    sys.exit(main())