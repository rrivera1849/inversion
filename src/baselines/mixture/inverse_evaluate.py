
import os
import sys

import evaluate
import nltk
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import roc_auc_score

from utils import load_mixture_predictor, get_mixture_weights

bleu = evaluate.load("bleu")
sbert = SentenceTransformer("all-mpnet-base-v2")
mixture_predictor = load_mixture_predictor()

def calculate_sbert_similarity(
    candidates: list[str], 
    references: list[str],
):
    embeddings1 = sbert.encode(candidates, convert_to_tensor=True)
    embeddings2 = sbert.encode(references, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2).diag().cpu().tolist()
    avg_sbert_similarity = sum(cosine_scores) / len(cosine_scores)
    return avg_sbert_similarity, cosine_scores

def token_f1(
    candidates: list[str], 
    references: list[str],
):
    F1s = []
    for r, p in zip(references, candidates):
        true_words = nltk.tokenize.word_tokenize(r)
        pred_words = nltk.tokenize.word_tokenize(p)

        true_words_set = set(true_words)
        pred_words_set = set(pred_words)
        TP = len(true_words_set & pred_words_set)
        FP = len(true_words_set) - len(true_words_set & pred_words_set)
        FN = len(pred_words_set) - len(true_words_set & pred_words_set)

        precision = (TP) / (TP + FP + 1e-20)
        recall = (TP) / (TP + FN + 1e-20)
        try:
            f1 = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1 = 0.0
        F1s.append(f1)
    return sum(F1s) / len(F1s), F1s

def get_MTD_score(
    candidates: list[str], 
    units: list[str], 
    max_fpr: int = 0.01
):
    out_candidates = get_mixture_weights(
        mixture_predictor,
        candidates, key=None, 
        return_sequence_probs=True,
        progress_bar=False,
    )[0]
    out_units = get_mixture_weights(
        mixture_predictor,
        units, 
        key=None,
        return_sequence_probs=True,
        progress_bar=False,
    )[0]
    
    score_candidates = [out[1] for out in out_candidates]
    score_units = [out[1] for out in out_units]
    scores = score_candidates + score_units
    labels = [1] * len(score_candidates) + [0] * len(score_units)
    AUC = roc_auc_score(labels, scores, max_fpr=max_fpr)
    return AUC

def main():
    metrics = {}
    dataset_path = "./prompting_data/results_to_evaluate"
    for filename in os.listdir(dataset_path):
        path = os.path.join(dataset_path, filename)
        if not path.endswith(".jsonl"):
            continue

        print("Evaluating", filename)
        df = pd.read_json(path, lines=True)

        if "inverse" in df.columns:
            candidates = df.inverse.tolist()
            references = df.unit.tolist()
        else:
            candidates = df.rephrase.tolist()
            references = df.unit.tolist()

        name = os.path.splitext(os.path.basename(filename))[0]
        metrics[name] = {}
        metrics[name]["bleu"] = bleu.compute(predictions=candidates, references=references)["bleu"]
        metrics[name]["token_f1"] = token_f1(candidates, references)
        metrics[name]["mtd_score"] = get_MTD_score(candidates, references)
        metrics[name]["sbert_similarity"] = calculate_sbert_similarity(candidates, references)

    df = pd.DataFrame.from_dict(metrics, orient="index")
    print(df.to_markdown())
    df.to_json("results.json", orient="index")

if __name__ == "__main__":
    sys.exit(main())