
import json
import os
from functools import partial
from multiprocessing import Pool

import evaluate
import numpy as np
import pandas as pd
import nltk
from tqdm.auto import tqdm

BLEU = evaluate.load("bleu")
DEBUG = False

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

def compute_metrics(input, targeted=False):
    _, row = input
    
    metrics = {}
    metrics["token_f1"] = {}
    metrics["bleu"] = {}
    
    if targeted:
        unit = row["unit_x"]
        rephrase = row["rephrase_x"]
        inverse = row["inverse"]
    else:
        unit = row["unit"]
        rephrase = row["rephrase"]
        inverse = row["inverse"]

    metrics["token_f1"]["rephrase"] = token_f1([rephrase], [unit])[0]
    inverse_token_f1 = token_f1(inverse, [unit for _ in range(len(inverse))])
    metrics["token_f1"]["inverse_single"] = inverse_token_f1[0]
    metrics["token_f1"]["inverse_expected"] = np.mean(inverse_token_f1)
    metrics["token_f1"]["inverse_max"] = np.max(inverse_token_f1)

    metrics["bleu"]["rephrase"] = BLEU.compute(predictions=[rephrase], references=[unit])["bleu"]
    inverse_bleu = [BLEU.compute(predictions=[inv], references=[unit])["bleu"] for inv in inverse]
    metrics["bleu"]["inverse_single"] = inverse_bleu[0]
    metrics["bleu"]["inverse_expected"] = np.mean(inverse_bleu)
    metrics["bleu"]["inverse_max"] = np.max(inverse_bleu)
    
    return metrics

def untargeted():
    base_path = "/data1/yubnub/changepoint/MUD_inverse/data/data.jsonl.filtered.cleaned_kmeans_100/inverse_output"
    files = [
        "none_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=100",
        "none_6400_temperature=0.5_top_p=0.9.jsonl.vllm_n=100",
        "none_6400_temperature=0.6_top_p=0.9.jsonl.vllm_n=100",
        "none_6400_temperature=0.8_top_p=0.9.jsonl.vllm_n=100",
        "none_6400_temperature=0.9_top_p=0.9.jsonl.vllm_n=100",
        "none_6400_temperature=0.3_top_p=0.9.jsonl.vllm_n=100",
        "none_6400_temperature=1.5_top_p=0.9.jsonl.vllm_n=100",
    ]
    if DEBUG:
        files = [files[0]]
    
    pool = Pool(40)
    
    for fname in files:
        print(f"Processing {fname}")
        path = os.path.join(base_path, fname)
        df = pd.read_json(path, lines=True)
        if DEBUG:
            df = df.iloc[:10]
            
        all_metrics = list(tqdm(pool.imap(compute_metrics, df.iterrows()), total=len(df)))

        avg_metrics = {}
        avg_metrics["token_f1"] = {}
        avg_metrics["bleu"] = {}
        for metric in ["token_f1", "bleu"]:
            for key in all_metrics[0][metric]:
                avg_metrics[metric][key] = np.mean([m[metric][key] for m in all_metrics])
                
        name = fname[:-len(".jsonl.vllm_n=100")] + "_simple_untargeted"
        if DEBUG:
            name += "_debug"
        with open(f"./metrics/{name}.json", "w") as f:
            f.write(json.dumps(avg_metrics, indent=4))
    
    pool.close()
    
def targeted():
    
    base_path = "/data1/yubnub/changepoint/MUD_inverse/data/data.jsonl.filtered.cleaned_kmeans_100/inverse_output"
    files = [
        # "none_targetted=examples_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=5.targetted_mode=author_num_examples=1",
        # "none_targetted=examples_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=5.targetted_mode=author_num_examples=2",
        # "none_targetted=examples_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=5.targetted_mode=author_num_examples=3",
        "none_targetted=examples_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=5.targetted_mode=author",
        "none_targetted_6400_temperature=0.7_top_p=0.9.jsonln=5.targetted_mode=author",
    ]
    if DEBUG:
        files = [files[0]]

    pool = Pool(40)
    
    for fname in files:
        path = os.path.join(base_path, fname)
        
        df = pd.read_json(path, lines=True)
        df = df[df.author_id_x == df.author_id_y]
        if DEBUG:
            df = df.iloc[:100]

        fn = partial(compute_metrics, targeted=True)
        all_metrics = list(tqdm(pool.imap(fn, df.iterrows()), total=len(df)))

        avg_metrics = {}
        avg_metrics["token_f1"] = {}
        avg_metrics["bleu"] = {}
        for metric in ["token_f1", "bleu"]:
            for key in all_metrics[0][metric]:
                avg_metrics[metric][key] = np.mean([m[metric][key] for m in all_metrics])
                
        name = fname.replace(".jsonl", "") + "_simple_targeted"
        if DEBUG:
            name += "_debug"
        with open(f"./metrics/{name}.json", "w") as f:
            f.write(json.dumps(avg_metrics, indent=4))

    pool.close()
            

if __name__ == "__main__":
    os.makedirs("./metrics", exist_ok=True)

    untargeted()
    targeted()