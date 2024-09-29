
import numpy as np
import torch
from sentence_transformers import util
from sklearn.metrics import (
    roc_curve, 
    roc_auc_score,
)

def calculate_metrics(
    labels: list[int], 
    sims: list[float], 
    suffix: str,
) -> dict:
    metrics = {}
    fpr, tpr, thresholds = roc_curve(labels, sims, pos_label=1)
    fnr = 1 - tpr
    metrics[f"ROC_{suffix}"] = (fpr.tolist(), tpr.tolist(), thresholds.tolist())
    metrics[f"AUC_{suffix}"] = roc_auc_score(labels, sims)
    metrics[f"AUC(0.01)_{suffix}"] = roc_auc_score(labels, sims, max_fpr=0.01)
    metrics[f"EER_{suffix}"] = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return metrics

def max_similarity(
    embeddings1: torch.Tensor,  
    embeddings2: torch.Tensor,
) -> float:
    embeddings1 = embeddings1.repeat(embeddings2.size(0), 1)
    similarities = util.pytorch_cos_sim(embeddings1, embeddings2).diag().cpu().tolist()
    maxsim = max(similarities)
    return maxsim

def expected_similarity(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor,
) -> float:
    embeddings1 = embeddings1.repeat(embeddings2.size(0), 1)
    similarities = util.pytorch_cos_sim(embeddings1, embeddings2).diag().cpu().tolist()
    avg_cosine_similarity = sum(similarities) / len(similarities)
    return avg_cosine_similarity

