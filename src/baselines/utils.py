
import json
import os
import sys
from collections import Counter
from typing import Union

import numpy as np
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

def load_s2orc_MTD_data(
    dirname: str = "/data1/yubnub/changepoint/s2orc_changepoint/unit_128", 
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

def load_author_data(
    author_data_dirname: str = "/data1/yubnub/data/iur_dataset/author_100.politics",
    debug: bool = False,
    debug_N: int = 50,
):
    """Loads a split from the Author 100 Politics dataset.
    """
    N = None
    if debug:
        N = debug_N
        
    # suffixes = ["", ".mistral", ".mistral.inverse", ".mistral.inverse-mixture", ".inverse-mixture-simple"]
    # model_names = ["human", "machine", "machine", "machine", "machine"]
    # prompt_names = ["human", "mistral", "inverse", "inverse-mixture", "inverse-mixture-simple"]
    suffixes = ["", ".mistral.inverse-mixture-simple"]
    model_names = ["human", "machine"]
    prompt_names = ["human", "inverse-mixture-simple"]
    texts, models, prompts = [], [], []
    for i, suffix in enumerate(suffixes):
        with open(os.path.join(author_data_dirname, f"test.jsonl{suffix}")) as fin:
            for j, line in enumerate(fin):
                if N is not None and j >= N:
                    break
                texts.append(json.loads(line)["syms"])
                models.append(model_names[i])
                prompts.append(prompt_names[i])

    return texts, models, prompts    

def load_inverse_data(
    inverse_data_dirname: str = "/home/riverasoto1/repos/changepoint/src/baselines/mixture/prompting_data",
    debug: bool = False,
    debug_N: int = 50,
):
    texts = []
    models = []
    prompts = []
    with open(os.path.join(inverse_data_dirname, "rephrases_gpt-4o-mini.jsonl")) as fin:
        data = [json.loads(line) for line in fin]
        texts.extend([elem["rephrase"] for elem in data])
        models.extend(["gpt-4o"] * len(data))
        prompts.extend(["rephrase"] * len(data))
    with open(os.path.join(inverse_data_dirname, "inverse_prompts_gpt-4o-mini_keep=True.jsonl")) as fin:
        data = [json.loads(line) for line in fin]
        texts.extend([elem["inverse"] for elem in data])
        models.extend(["gpt-4o"] * len(data))
        prompts.extend(["inverse"] * len(data))
        texts.extend([elem["unit"] for elem in data])
        models.extend(["human"] * len(data))
        prompts.extend(["human"] * len(data))
        
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
    prompts: list[Union[None, str]] = None,
    max_fpr: float = 0.01
) -> dict[str, float]:

    metrics: dict[str, float] = {}
    
    human_scores = [score for i, score in enumerate(scores) if models[i] == "human"]

    machine_scores = [score for i, score in enumerate(scores) if models[i] != "human"]
    labels = [0] * len(human_scores) + [1] * len(machine_scores)
    metrics[f"global AUC({max_fpr})"] = roc_auc_score(labels, human_scores + machine_scores, max_fpr=max_fpr)

    if prompts is not None:
        for prompt in np.unique(prompts):
            prompt_machine_scores = [score for i, score in enumerate(scores) if models[i] != "human" and prompts[i] == prompt]
            if len(prompt_machine_scores) == 0:
                continue
            prompt_labels = [0] * len(human_scores) + [1] * len(prompt_machine_scores)
            metrics[prompt + f" AUC({max_fpr})"] = roc_auc_score(prompt_labels, human_scores + prompt_machine_scores, max_fpr=max_fpr)
        
    return metrics