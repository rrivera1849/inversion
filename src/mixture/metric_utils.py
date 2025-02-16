
from typing import Union

import numpy as np
import torch
from sentence_transformers import util
from sklearn.metrics import (
    roc_curve, 
    roc_auc_score,
)
from tqdm.auto import tqdm

def calculate_metrics(
    labels: list[int], 
    sims: list[float], 
) -> dict:
    metrics = {}
    fpr, tpr, thresholds = roc_curve(labels, sims, pos_label=1)
    fnr = 1 - tpr
    metrics[f"ROC"] = (fpr.tolist(), tpr.tolist(), thresholds.tolist())
    metrics[f"AUC"] = roc_auc_score(labels, sims)
    metrics[f"AUC(0.01)"] = roc_auc_score(labels, sims, max_fpr=0.01)
    eer_1 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    metrics[f"EER"] = (eer_1 + eer_2) / 2
    return metrics

def calculate_similarities_and_metrics(
    emb1: torch.Tensor,
    emb2: Union[torch.Tensor, list[torch.Tensor]],
    labels: list[int],
    simtype: str = None,
    diagonal: bool = False
):
    if simtype:
        similarities = pairwise_similarity(emb1, emb2, type=simtype, diagonal=diagonal)
    else:
        similarities = util.pytorch_cos_sim(emb1, emb2)
        if diagonal:
            similarities = similarities.diag()
        similarities = similarities.cpu().flatten().tolist()

    metrics = calculate_metrics(labels, similarities)
    metrics["similarities"] = similarities
    metrics["labels"] = labels
    return metrics

def max_similarity(
    embeddings1: torch.Tensor,  
    embeddings2: torch.Tensor,
    return_max_index: bool = False,
) -> float:
    embeddings1 = embeddings1.repeat(embeddings2.size(0), 1)
    similarities = util.pytorch_cos_sim(embeddings1, embeddings2).diag().cpu().tolist()

    maxsim = max(similarities)
    if return_max_index:
        max_index = similarities.index(maxsim)
        return max_index
    return maxsim

def expected_similarity(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor,
) -> float:
    embeddings1 = embeddings1.repeat(embeddings2.size(0), 1)
    similarities = util.pytorch_cos_sim(embeddings1, embeddings2).diag().cpu().tolist()
    avg_cosine_similarity = sum(similarities) / len(similarities)
    return avg_cosine_similarity

def pairwise_similarity(
    query_embeddings: torch.Tensor,
    inverse_embeddings: list[torch.Tensor],
    type: str = "expected",
    progress_bar: bool = False,
    diagonal: bool = False,
) -> list[float]:
    assert type in ["max", "expected"]
    type_to_func = {
        "max": max_similarity,
        "expected": expected_similarity,
    }
    similarities = []
    if progress_bar:
        pbar = tqdm(total=len(query_embeddings) * len(inverse_embeddings))
    for i in range(len(query_embeddings)):
        for j in range(len(inverse_embeddings)):
            if diagonal and i == j:
                similarities.append(type_to_func[type](query_embeddings[i:i+1], inverse_embeddings[j]))
            if not diagonal:
                similarities.append(type_to_func[type](query_embeddings[i:i+1], inverse_embeddings[j]))
            if progress_bar:
                pbar.update(1)
    return similarities
