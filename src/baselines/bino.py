
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
from tqdm import tqdm

from mixture.model import MixturePredictor
from utils import load_MTD_data, compute_metrics

parser = ArgumentParser()
parser.add_argument("--dirname", type=str,
                    default="/data1/yubnub/changepoint/s2orc_changepoint/unit_128",
                    help="Directory where the dataset is stored.")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--debug", default=False, action="store_true",
                    help="If True, will process only a few samples.")
args = parser.parse_args()

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
def get_MSP_scores(
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
            scores.append(get_MSP(text[i], model, tokenizer))
        except:
            scores.append(None)

    return scores

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
def get_mixture_predictor_scores(
    text: list[str],
    batch_size: int = 32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MixturePredictor()
    state_dict = torch.load("./mixture/outputs/mixture_baseline_nocontinuation/checkpoints/best.pth")["model"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    probabilities = []
    for i in tqdm(range(0, len(text), batch_size)):
        batch = text[i:i+batch_size]
        sequence_mixture_preds, _ = model.predict(batch)
        sequence_mixture_preds = F.log_softmax(sequence_mixture_preds, dim=-1)[:, 1].exp().tolist()
        probabilities.extend(sequence_mixture_preds)
    
    return probabilities

def main():
    texts, models, prompts = load_MTD_data(args.dirname, debug=args.debug, debug_N=50)
    
    # TODO: Refactor to remove the repetition.
    
    # scores = get_bino_scores(texts, batch_size=args.batch_size)
    # scores = [-score for score in scores]
    # metrics = compute_metrics(scores, models, prompts)
    # print()
    # print(colored("Binocular metrics:", "blue"))
    # for k, v in metrics.items():
    #     print(colored(f"\t{k}: {v:.4f}", "green"))

    # scores = get_openai_detector_scores(texts, batch_size=args.batch_size)
    # metrics = compute_metrics(scores, models, prompts)
    # print()
    # print(colored("OpenAI Detector metrics:", "blue"))
    # for k, v in metrics.items():
    #     print(colored(f"\t{k}: {v:.4f}", "green"))
        
    # scores = get_MSP_scores(texts)
    # metrics = compute_metrics(scores, models, prompts)
    # print()
    # print(colored("MSP metrics:", "blue"))
    # for k, v in metrics.items():
    #     print(colored(f"\t{k}: {v:.4f}", "green"))
    
    # scores = get_RADAR_scores(texts, batch_size=args.batch_size)
    # metrics = compute_metrics(scores, models, prompts)
    # print()
    # print(colored("RADAR metrics:", "blue"))
    # for k, v in metrics.items():
    #     print(colored(f"\t{k}: {v:.4f}", "green"))
    
    # scores = get_mixture_predictor_scores(texts, batch_size=args.batch_size)
    # metrics = compute_metrics(scores, models, prompts)
    # print()
    # print(colored("MixturePredictor metrics:", "blue"))
    # for k, v in metrics.items():
    #     print(colored(f"\t{k}: {v:.4f}", "green"))

    return 0

if __name__ == "__main__":
    sys.exit(main())