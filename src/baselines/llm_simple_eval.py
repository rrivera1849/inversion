"""Evaluation Protocol
"""

import os
import sys
from argparse import ArgumentParser
from collections import Counter

import matplotlib.pyplot as plt
import torch
from datasets import load_from_disk
from peft import PeftModel 
from termcolor import colored
from transformers import AutoModel, AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM
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

MODELS = {
    "human": "/scratch1/yubnub/changepoint/output/Mistral-7B-v0.3-QLoRA-prompt=none-perc=0.1-ns=100000-debug=False/checkpoint-3000",
    "rephrase": "/scratch1/yubnub/changepoint/output/Mistral-7B-v0.3-QLoRA-prompt=rephrase-perc=0.1-ns=100000-debug=False/checkpoint-3000",
}

MODEL_NAMES = [
    "Mistral-7B-Instruct-v0.3",
    "Meta-Llama-3-8B-Instruct",
    "Phi-3-mini-4k-instruct"
]

@torch.no_grad()
def PPL(
    text: list[str],
    model: AutoModel, 
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
        batch = {k:v.to(model.device) for k,v in batch.items()}
        
        losses = model(**batch).loss
        PPLs.append(torch.exp(losses).item())

    return PPLs

@torch.no_grad()
def joint_mixture_predict(
    texts: list[str],
    human_model: AutoModel, 
    rephrase_model: AutoModel, 
    tokenizer: AutoTokenizer,
):
    device = human_model.device
    
    scores = []
    for i in tqdm(range(len(texts))):
        batch = texts[i]

        batch = tokenizer(
            batch,
            return_tensors="pt",
        )
        batch = {k:v.to(device) for k,v in batch.items()}

        out_1 = human_model(**batch)
        out_2 = rephrase_model(**batch)
        
        num_above = 0
        for pos_index in range(1, batch["input_ids"].size(1)):
            token_index = batch["input_ids"][0, pos_index].item()
            if out_2["logits"][:, pos_index, token_index] > out_1["logits"][:, pos_index, token_index]:    
                num_above += 1

        scores.append(num_above / batch["input_ids"].size(1))
    return scores

def load_model(
    peft_path: str,
):
    MODEL_NAME = "mistralai/Mistral-7B-v0.3"
    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_NAME,
    #     device_map="auto",
    #     torch_dtype=torch.float16,
    # )
    # model = PeftModel.from_pretrained(
    #     model,
    #     model_id=peft_path,
    # )
    # # merge the LoRA Adapter layers into the base model
    # model.merge_and_unload()
    # model.eval()
    
    model = AutoModelForCausalLM.from_pretrained(
        peft_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side="left", # RRS - do we need this when running evaluations?
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

from typing import Union
from sklearn.metrics import roc_auc_score
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
        if len(prompt_machine_scores) == 0:
            continue
        prompt_labels = [0] * len(human_scores) + [1] * len(prompt_machine_scores)
        metrics[prompt + " AUC(0.01)"] = roc_auc_score(prompt_labels, human_scores + prompt_machine_scores, max_fpr=max_fpr)
        
    return metrics


def main():
    dataset = load_from_disk(os.path.join(args.dirname, "MTD_dataset"))
    
    if args.debug:
        N = 5
        total_humans = N * len(MODEL_NAMES) * len(PROMPT_NAMES)
        print(colored("DEBUG: Subsampling dataset...", "yellow"))

        counts = Counter()
        indices_to_keep = []

        for i in range(len(dataset)):
            elem = dataset[i]
            if elem["model"] == "human" and counts[("human", None)] < total_humans:
                if counts[("human", None)] < total_humans:
                    indices_to_keep.append(i)

                counts[("human", None)] += 1
            elif counts[(elem["model"], elem["prompt"])] < N:
                indices_to_keep.append(i)
                counts[(elem["model"], elem["prompt"])] += 1

            if sum(counts.values()) == 2 * total_humans:
                break

        assert len(indices_to_keep) == 2 * total_humans
        dataset = dataset.select(indices_to_keep)

    texts = [elem["text"] for elem in dataset]
    models = [elem["model"] for elem in dataset]
    prompts = [elem["prompt"] for elem in dataset]
    
    human_model, tokenizer = load_model(MODELS["human"])
    rephrase_model, _ = load_model(MODELS["rephrase"])

    scores = joint_mixture_predict(texts, human_model, rephrase_model, tokenizer)
    metrics = compute_metrics(scores, models, prompts)
    print()
    print(colored("LLM Simple Eval metrics:", "blue"))
    for k, v in metrics.items():
        print(colored(f"\t{k}: {v:.4f}", "green"))
    
    # TODO RRS - Think about whether we can use a variant of the Binoculars score
    # directly instead of using PPL.
    
    return 0

if __name__ == "__main__":
    sys.exit(main())