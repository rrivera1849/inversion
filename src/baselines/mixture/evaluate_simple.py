
import json
import os
from functools import partial
from multiprocessing import Pool
from argparse import ArgumentParser

import evaluate
import numpy as np
import pandas as pd
import nltk
from tqdm.auto import tqdm

parser = ArgumentParser()
parser.add_argument("--filename", type=str, default=None, required=True)
parser.add_argument("--dataset_name", type=str, default="data.jsonl.filtered.cleaned_kmeans_100")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

BLEU = evaluate.load("bleu")
DEBUG = args.debug

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
    return F1s

def BLEU_score(
    predictions: list[str], 
    references: list[str],
):
    try:
        return BLEU.compute(predictions=predictions, references=references)["bleu"]
    except ZeroDivisionError:
        return 0.0

def compute_metrics(input):
    _, row = input
    metrics = {}
    metrics["token_f1"] = {}
    metrics["bleu"] = {}
    
    unit = row["unit"]
    rephrase = row["rephrase"]
    inverse = row["inverse"]

    metrics["token_f1"]["rephrase"] = token_f1([rephrase], [unit])[0]
    inverse_token_f1 = token_f1(inverse, [unit for _ in range(len(inverse))])
    metrics["token_f1"]["inverse_single"] = inverse_token_f1[0]
    metrics["token_f1"]["inverse_expected"] = np.mean(inverse_token_f1)
    metrics["token_f1"]["inverse_max"] = np.max(inverse_token_f1)

    metrics["bleu"]["rephrase"] = BLEU_score(predictions=[rephrase], references=[unit])
    inverse_bleu = [BLEU_score(predictions=[inv], references=[unit]) for inv in inverse]
    metrics["bleu"]["inverse_single"] = inverse_bleu[0]
    metrics["bleu"]["inverse_expected"] = np.mean(inverse_bleu)
    metrics["bleu"]["inverse_max"] = np.max(inverse_bleu)
    
    return metrics

def untargeted():
    base_path = f"/data1/yubnub/changepoint/MUD_inverse/data/{args.dataset_name}/inverse_output"
    
    path = os.path.join(base_path, args.filename)
    df = pd.read_json(path, lines=True)
    if DEBUG:
        df = df.iloc[:10]
        
    with Pool(40) as pool:
        all_metrics = list(tqdm(pool.imap(compute_metrics, df.iterrows()), total=len(df)))

    avg_metrics = {}
    avg_metrics["token_f1"] = {}
    avg_metrics["bleu"] = {}
    for metric in ["token_f1", "bleu"]:
        for key in all_metrics[0][metric]:
            avg_metrics[metric][key] = np.mean([m[metric][key] for m in all_metrics])

    os.makedirs(f"./metrics/new/{args.dataset_name}/basic", exist_ok=True)
    with open(f"./metrics/new/{args.dataset_name}/basic/{args.filename}", "w") as f:
        f.write(json.dumps(avg_metrics, indent=4))
    

if __name__ == "__main__":
    os.makedirs("./metrics", exist_ok=True)

    untargeted()