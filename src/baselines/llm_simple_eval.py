"""TODOS
1. Token Level / Sequence Level Hypothesis Testing
2. Above but with imputed masks from the MixturePredictor
3. KL Divergence of "Human LLM" vs "Rephrase LLM"
4. Few-Shot KL-Divergence: given a few human examples, get the divergences 
   to them...

Code Misc:
1. Add a configuration file for the models.
"""

import os
import sys
from argparse import ArgumentParser
from collections import Counter

import torch
from binoculars import Binoculars
from termcolor import colored
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from mixture.model import MixturePredictor
from utils import load_MTD_data, compute_metrics, load_model, MODELS

parser = ArgumentParser()
parser.add_argument("--dirname", type=str,
                    default="/data1/yubnub/changepoint/s2orc_changepoint/unit_128",
                    help="Directory where the dataset is stored.")
parser.add_argument("--only_rephrases", default=False, action="store_true",
                    help="If True, will only process human & rephrase samples.")
parser.add_argument("--debug", default=False, action="store_true",
                    help="If True, will process only a few samples.")
args = parser.parse_args()


def KL_divergence(
    distribution: torch.Tensor,
    reference_distribution: torch.Tensor,
) -> torch.Tensor:
    # KL(P || Q)
    return (distribution * (distribution / reference_distribution).log()).sum()

@torch.inference_mode()
def get_divergence_from_human(
    texts: list[str],
    human_model: AutoModelForCausalLM,
    rephrase_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch_size: int = 1, # TODO
    partial_distribution: bool = False,
):
    divergences = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        batch = tokenizer(
            batch,
            max_length=128+32,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        batch = {k:v.to(human_model.device) for k,v in batch.items()}
        
        human_output = human_model(**batch).logits
        rephrase_output = rephrase_model(**batch).logits

        human_distribution = torch.softmax(human_output, dim=-1)
        rephrase_distribution = torch.softmax(rephrase_output, dim=-1)

        indices = batch["input_ids"] != tokenizer.pad_token_id
        indices = indices.squeeze()
        human_distribution = human_distribution[:, indices, :]
        rephrase_distribution = rephrase_distribution[:, indices, :]

        if partial_distribution:
            input_ids = batch["input_ids"][:, indices].squeeze()
            human_distribution = human_distribution[:, torch.arange(len(human_distribution)), input_ids]
            rephrase_distribution = rephrase_distribution[:, torch.arange(len(rephrase_distribution)), input_ids]

        divergence = KL_divergence(human_distribution, rephrase_distribution)
        divergences.append(divergence.item())

    return divergences



@torch.no_grad()
def hypothesis_test_with_mixture(
    text: list[str],
    human_model: AutoModelForCausalLM,
    rephrase_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch_size: int = 1, # TODO
):
    """This is essentially the PPL under the `human_model`, followed by the 
       PPL imputed by the mask predicted by the MixturePredictor.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixture_predictor = MixturePredictor()
    state_dict = torch.load("./mixture/outputs/mixture_baseline_nocontinuation/checkpoints/best.pth")["model"]
    mixture_predictor.load_state_dict(state_dict)
    mixture_predictor.to(device)
    mixture_predictor.eval()

    loss_fn = torch.nn.CrossEntropyLoss()

    ratios = []
    for i in range(0, len(text), batch_size):
        batch = text[i:i+batch_size]
        human_PPLs = PPL(batch, human_model, tokenizer)

        _, token_mixture_mask = mixture_predictor.predict(batch)

        batch = tokenizer(
            batch,
            max_length=128+32,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        human_out = human_model(**batch).logits
        rephrase_out = rephrase_model(**batch).logits
        
        for j in range(len(batch["input_ids"])):
            # 1 x D
            input_ids = batch["input_ids"][j:j+1]
            # 1 x D x VOCAB
            human_logits = human_out[j:j+1]
            rephrase_logits = rephrase_out[j:j+1]
            # 1 x D
            mask = input_ids != tokenizer.pad_token_id

            input_ids = input_ids[mask].unsqueeze(0)
            human_logits = human_logits[mask].unsqueeze(0)
            rephrase_logits = rephrase_logits[mask].unsqueeze(0)
            
            mixture_mask = token_mixture_mask[j].argmax(dim=1).unsqueeze(0)
            
            import pdb; pdb.set_trace()
            logits = (human_logits * ~mixture_mask) + (rephrase_logits * mixture_mask)
            logits = logits[..., :-1, :].contiguous()
            labels = input_ids[..., 1:].contiguous()
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

            loss_out = loss_fn(logits, labels)

    return ratios

@torch.no_grad()
def PPL(
    text: list[str],
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
):
    PPLs = []
    for i in tqdm(range(0, len(text), batch_size)):
        batch = text[i:i+batch_size]
        batch = tokenizer(
            batch,
            max_length=128+32,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["labels"] == tokenizer.pad_token_id] = -100
        batch = {k:v.to(model.device) for k,v in batch.items()}
        
        losses = model(**batch).loss
        PPLs.append(torch.exp(losses).item())

    return PPLs


@torch.no_grad()
def get_bino_scores(
    text: list[str],
    batch_size: int = 32,
    use_human_and_rephrase: bool = False,
) -> list[float]:
    """Compute Bino scores for a list of text
    """
    if use_human_and_rephrase:
        observer = MODELS["human"]
        performer = MODELS["rephrase"]
        tokenizer_model_id = "mistralai/Mistral-7B-v0.3"
    else:
        observer = "tiiuae/falcon-7b"
        performer = "tiiuae/falcon-7b-instruct"
        tokenizer_model_id = None

    bino = Binoculars(
        observer_name_or_path=observer,
        performer_name_or_path=performer,
        tokenizer_model_id=tokenizer_model_id,
    )
    scores = []
    for i in tqdm(range(0, len(text), batch_size)):
        batch = text[i:i+batch_size]
        scores += bino.compute_score(batch)
    return [-score for score in scores]

def main():
    texts, models, prompts = load_MTD_data(args.dirname, debug=args.debug)

    if args.only_rephrases:
        indices_to_keep = [i for i, prompt in enumerate(prompts) if prompt is None or prompt == "rephrase"]
        texts = [texts[i] for i in indices_to_keep]
        models = [models[i] for i in indices_to_keep]
        prompts = [prompts[i] for i in indices_to_keep]

    print("Dataset Prompt Counts:")
    for key, value in Counter(prompts).items():
        print(f"\t{key}: {value}")
    print("Dataset Model Counts:")
    for key, value in Counter(models).items():
        print(f"\t{key}: {value}")
    print()
    
    human_model, tokenizer = load_model(MODELS["human"])
    # rephrase_model, _ = load_model(MODELS["rephrase"])
    rephrase_model, _ = load_model()

    ppl_human_scores = PPL(texts, human_model, tokenizer, batch_size=1)
    ppl_rephrase_scores = PPL(texts, rephrase_model, tokenizer, batch_size=1)
    predictions = []
    for i in range(len(texts)):
        if ppl_human_scores[i] < ppl_rephrase_scores[i]:
            predictions.append(0)
        else:
            predictions.append(1)
    ground_truth_labels = [0 if model == "human" else 1 for model in models]
    from sklearn.metrics import classification_report
    print(classification_report(ground_truth_labels, predictions))    

    ratios = [h / r for h, r in zip(ppl_human_scores, ppl_rephrase_scores)]
    metrics = compute_metrics(ratios, models, prompts, max_fpr=0.1)
    print()
    print(colored("Ratios of PPL:", "blue"))
    for k, v in metrics.items():
        print(colored(f"\t{k}: {v:.4f}", "green"))

    return 0

if __name__ == "__main__":
    sys.exit(main())