
import random; random.seed(43)
import json
import os
import sys
from argparse import ArgumentParser
from functools import partial

import torch
import torch.nn.functional as F
from binoculars import Binoculars
from sklearn.metrics import roc_curve
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification, 
    AutoTokenizer
)
from tqdm import tqdm

from fast_detect_gpt import get_sampling_discrepancy, get_sampling_discrepancy_analytic
from utils import load_machine_paraphrase_data, compute_metrics, load_human_paraphrase_data, load_model

parser = ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="data.jsonl.filtered.respond_reddit.cleaned")
parser.add_argument("--filename", type=str, default="MTD_all_none_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=100-with-preds")
parser.add_argument("--only_inverse", default=False, action="store_true")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--debug", default=False, action="store_true")
args = parser.parse_args()

def get_rank(
    texts: list[str], 
    log: bool = False,
):
    """From: https://github.com/eric-mitchell/detect-gpt/blob/main/run.py#L298C1-L320C43
    """
    base_model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
    base_model.cuda()
    base_tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
    base_tokenizer.pad_token = base_tokenizer.eos_token

    with torch.no_grad():
        result = []
        for text in tqdm(texts):
            tokenized = base_tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(base_model.device)
            logits = base_model(**tokenized).logits[:,:-1]
            labels = tokenized["input_ids"][:, 1:]

            # get rank of each label token in the model's likelihood ordering
            matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

            assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

            ranks, timesteps = matches[:,-1], matches[:,-2]

            # make sure we got exactly one match for each timestep in the sequence
            assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

            ranks = ranks.float() + 1 # convert to 1-indexed rank
            if log:
                ranks = torch.log(ranks)

            result.append(ranks.float().mean().item())
            
    return result

def get_entropy(texts):
    """From: https://github.com/eric-mitchell/detect-gpt/blob/main/run.py#L324C1-L332C1
    """
    base_model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
    base_tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
    result = []
    with torch.no_grad():
        for text in tqdm(texts):
            tokenized = base_tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(base_model.device)
            logits = base_model(**tokenized).logits[:,:-1]
            neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
            result.append(-neg_entropy.sum(-1).mean().item())
    return result
    
@torch.no_grad()
def get_bino_scores(
    text: list[str],
    batch_size: int = 8,
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

@torch.no_grad()
def get_RADAR_scores(
    text: list[str],
    batch_size: int = 32,
) -> list[float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector = AutoModelForSequenceClassification.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
    tokenizer = AutoTokenizer.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
    detector.to(device)
    detector.eval()
    
    probabilities = []
    for i in tqdm(range(0, len(text), batch_size)):
        batch = text[i:i+batch_size]
        batch = tokenizer(
            batch,
            max_length=512, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
        )
        batch = {k:v.to(device) for k,v in batch.items()}

        output_probs = F.log_softmax(detector(**batch).logits, dim=-1)[:, 0].exp().tolist()
        probabilities.extend(output_probs)

    return probabilities

@torch.no_grad()
def get_fast_detect_gpt_scores(
    text: list[str],
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, base_tok = load_model("gpt2-xl")

    scores = []
    for sample in tqdm(text):
        tok = base_tok(
            sample, 
            padding=True,
            truncation=True, 
            return_tensors="pt", 
        ).to(device)
        base_logits = base_model(**tok).logits[:, :-1]
        reference_logits = base_logits
        labels = tok["input_ids"][:, 1:]
        discrepancy = get_sampling_discrepancy(reference_logits, base_logits, labels)
        scores.append(discrepancy)
    return scores

def main():
    data_dirname = "/data1/yubnub/changepoint/MUD_inverse/data"
    filename = os.path.join(data_dirname, args.dataset_name, "inverse_output", args.filename)
    data = load_machine_paraphrase_data(filename=filename, pick_dissimilar=True, debug=args.debug)

    detector_map = {
        "Binoculars": get_bino_scores,
        "OpenAI": get_openai_detector_scores,
        "rank": get_rank,
        "logrank": partial(get_rank, log=True),
        "entropy": get_entropy,
        "RADAR": get_RADAR_scores,
        "FastDetectGPT": get_fast_detect_gpt_scores,
    }
    detector_invert_map = {
        "Binoculars": True,
        "OpenAI": False,
        "rank": True,
        "logrank": True,
        "entropy": True,
        "RADAR": False,
        "FastDetectGPT": False,
    }
    
    outputs = {}
    for detector_name, detector_fn in detector_map.items():
        for key in data.keys():
            if args.only_inverse and key != "with_inverse":
                continue
            
            print(key, detector_name)
            scores = detector_fn(data[key]["texts"])
            invert_score = detector_invert_map[detector_name]
            if invert_score:
                scores = [-s for s in scores]
            metrics = compute_metrics(
                scores,
                data[key]["models"],
                max_fpr=1.0,
            )
            print(key, metrics)
            labels = [model != "human" for model in data["without_inverse"]["models"]]
            fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
            fpr = fpr.tolist()
            tpr = tpr.tolist()
            thresholds = thresholds.tolist()
            
            outputs[f"{detector_name}-{key}"] = (fpr, tpr, thresholds)
            outputs[f"{detector_name}-{key}-metrics"] = metrics

    savedir = "./mtd_plotting"
    os.makedirs(savedir, exist_ok=True)
    savename = "metrics_" + args.dataset_name + "--" + args.filename
    savename += ".debug" if args.debug else ""
    with open(os.path.join(savedir, savename), "w+") as fout:
        fout.write(json.dumps(outputs))

    return 0

if __name__ == "__main__":
    sys.exit(main())