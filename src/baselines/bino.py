
import random; random.seed(43)
import sys
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from binoculars import Binoculars
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification, 
    AutoTokenizer
)
from termcolor import colored
from tqdm import tqdm

from fast_detect_gpt import get_sampling_discrepancy, get_sampling_discrepancy_analytic
from utils import load_machine_paraphrase_data, compute_metrics, load_human_paraphrase_data, load_model

parser = ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="machine_paraphrase",
                    choices=["machine_paraphrase", "human_paraphrase"])
parser.add_argument("--filename", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--debug", default=False, action="store_true")
args = parser.parse_args()

def get_rank(
    text: list[str], 
    base_model: AutoModelForCausalLM, 
    base_tokenizer: AutoTokenizer,
    log: bool = False,
):
    """From: https://github.com/eric-mitchell/detect-gpt/blob/main/run.py#L298C1-L320C43
    """
    with torch.no_grad():
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

        return ranks.float().mean().item()
    
@torch.no_grad()
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

    base_model, base_tok = load_model()

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
    if args.dataset_name == "machine_paraphrase":
        if args.filename is not None:
            data = load_machine_paraphrase_data(filename=args.filename, pick_dissimilar=True, debug=args.debug)
        else:
            data = load_machine_paraphrase_data(pick_dissimilar=True, debug=args.debug)
    else:
        if args.filename is not None:
            data = load_human_paraphrase_data(filename=args.filename, debug=args.debug)
        else:
            data = load_human_paraphrase_data(debug=args.debug)

    scores_without = get_fast_detect_gpt_scores(data["without_inverse"]["texts"])
    metrics_without_inverse = compute_metrics(
        scores_without, 
        data["without_inverse"]["models"], 
        max_fpr=0.01,
    )
    print("Without inverse:")
    print(metrics_without_inverse)
    
    scores_with = get_fast_detect_gpt_scores(data["with_inverse"]["texts"])
    metrics_with_inverse = compute_metrics(
        scores_with, 
        data["with_inverse"]["models"], 
        max_fpr=0.01,
    )
    print("With inverse:")
    print(metrics_with_inverse)
    
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    _ = plt.figure()

    labels = [model != "human" for model in data["without_inverse"]["models"]]
    fpr, tpr, _ = roc_curve(
        labels,
        scores_without, 
        pos_label=1,
    )
    plt.plot(fpr, tpr, label="Human vs Rephrase(Machine)")
    
    labels = [model != "human" for model in data["with_inverse"]["models"]]
    fpr, tpr, _ = roc_curve(
        labels,
        scores_with,
        pos_label=1,
    )
    plt.plot(fpr, tpr, label="Human vs Inverse(Rephrase)")
    plt.xscale("log")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig("./mixture/plots/mtd_roc_curve.pdf")
    plt.close()

    # scores = get_RADAR_scores(texts, batch_size=args.batch_size)
    # metrics = compute_metrics(scores, models, prompts)
    # print()
    # print(colored("RADAR metrics:", "blue"))
    # for k, v in metrics.items():
    #     print(colored(f"\t{k}: {v:.4f}", "green"))
    

    return 0

if __name__ == "__main__":
    sys.exit(main())