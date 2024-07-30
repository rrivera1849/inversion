
import os
import sys
from collections import Counter
from typing import Union

from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
from termcolor import colored

sys.path.append("../../scripts/dataset/changepoint")
from prompts import PROMPT_NAMES

MODEL_NAMES = [
    "Mistral-7B-Instruct-v0.3",
    "Meta-Llama-3-8B-Instruct",
    "Phi-3-mini-4k-instruct"
]

def load_MTD_data(
    dirname: str, 
    debug: bool = False, 
    debug_N: int = 50,
) -> tuple[list[str], list[str], list[str]]:
    """Load the Machine-Text Detection dataset.
    """
    # set of fewshot examples for LUAR:
    fewshot_humans: list[str] = []

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
                else:
                    fewshot_humans.append(elem["text"])

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