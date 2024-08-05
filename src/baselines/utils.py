
import os
import sys
from collections import Counter
from typing import Union

import torch
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
from termcolor import colored
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("../../scripts/dataset/changepoint")
from prompts import PROMPT_NAMES

MODEL_NAMES = [
    "Mistral-7B-Instruct-v0.3",
    "Meta-Llama-3-8B-Instruct",
    "Phi-3-mini-4k-instruct"
]

LLM_PATH = "/data1/yubnub/changepoint/models/llm"
MODELS = {
    "human": "Mistral-7B-v0.3-QLoRA-prompt=none-perc=0.1-ns=100000-debug=False/checkpoint-3000",
    "rephrase": "Mistral-7B-v0.3-QLoRA-prompt=rephrase-perc=0.1-ns=100000-debug=False/checkpoint-3000",
}
MODELS = {k: os.path.join(LLM_PATH, v) for k,v in MODELS.items()}

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

def load_MTD_data(
    dirname: str, 
    debug: bool = False, 
    debug_N: int = 50,
) -> tuple[list[str], list[str], list[str]]:
    """Load the Machine-Text Detection dataset.
    """
    dataset = load_from_disk(os.path.join(dirname, "MTD_dataset"))
    if debug:
        N = debug_N
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

    return texts, models, prompts

# def load_RAID_data(
#     dirname: str, 
#     debug: bool = False, 
#     debug_N: int = 50,
# ):
#     dataset = load_from_disk(os.path.join("RAID_rephrase", "train_human_unit_128"))
#     # TODO

def compute_metrics(
    scores: list[float], 
    models: list[str], 
    prompts: list[Union[None, str]],
    max_fpr: float = 0.01
) -> dict[str, float]:
    assert len(scores) == len(models) == len(prompts)

    metrics: dict[str, float] = {}
    
    human_scores = [score for i, score in enumerate(scores) if models[i] == "human"]

    machine_scores = [score for i, score in enumerate(scores) if models[i] != "human"]
    labels = [0] * len(human_scores) + [1] * len(machine_scores)
    metrics[f"global AUC({max_fpr})"] = roc_auc_score(labels, human_scores + machine_scores, max_fpr=max_fpr)

    for prompt in PROMPT_NAMES:
        prompt_machine_scores = [score for i, score in enumerate(scores) if models[i] != "human" and prompts[i] == prompt]
        if len(prompt_machine_scores) == 0:
            continue
        prompt_labels = [0] * len(human_scores) + [1] * len(prompt_machine_scores)
        metrics[prompt + f" AUC({max_fpr})"] = roc_auc_score(prompt_labels, human_scores + prompt_machine_scores, max_fpr=max_fpr)
        
    return metrics