
import os
import sys

import evaluate
import nltk
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

bleu = evaluate.load("bleu")
sbert = SentenceTransformer("all-mpnet-base-v2")
luar = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
luar.to("cuda").eval()
luar_tok = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)

def cosine_similarity(
    embeddings_1: torch.Tensor,
    embeddings_2: torch.Tensor,
):
    cosine_scores = util.pytorch_cos_sim(embeddings_1, embeddings_2).diag().cpu().tolist()
    avg_cosine_similarity = sum(cosine_scores) / len(cosine_scores)
    return avg_cosine_similarity, cosine_scores

def calculate_sbert_similarity(
    candidates: list[str], 
    references: list[str],
):
    embeddings1 = sbert.encode(candidates, convert_to_tensor=True)
    embeddings2 = sbert.encode(references, convert_to_tensor=True)
    return cosine_similarity(embeddings1, embeddings2)

@torch.no_grad()
def get_luar_embeddings(
    text: str,
    batch_size: int = 32,
):
    all_outputs = []
    for i in range(0, len(text), batch_size):
        batch = text[i:i+batch_size]
        inputs = luar_tok(
            batch,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(luar.device)
        inputs["input_ids"] = inputs["input_ids"].view(len(batch), 1, 512)
        inputs["attention_mask"] = inputs["attention_mask"].view(len(batch), 1, 512)
        outputs = luar(**inputs)
        all_outputs.append(outputs)
    all_outputs = torch.cat(all_outputs, dim=0)
    return all_outputs

def calculate_luar_similarity(
    candidates: list[str],
    references: list[str],
):
    embeddings1 = get_luar_embeddings(candidates)
    embeddings2 = get_luar_embeddings(references)
    return cosine_similarity(embeddings1, embeddings2)

def calculate_embedding_similarity(
    candidates: list[str],
    references: list[str],
    model_name: str = "sbert",
):
    if model_name == "sbert":
        return calculate_sbert_similarity(candidates, references)
    elif model_name == "luar":
        return calculate_luar_similarity(candidates, references)
    else:
        raise ValueError(f"Model name {model_name} not supported.")

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

def main():
    metrics = {}
    dataset_path = "/data1/yubnub/changepoint/MUD_inverse/data/data.jsonl.filtered.cleaned_kmeans_100/inverse_output"
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
        metrics[name]["sbert_similarity"] = calculate_embedding_similarity(candidates, references, "sbert")
        metrics[name]["luar_similarity"] = calculate_embedding_similarity(candidates, references, "luar")

        print('\t', "BLEU", metrics[name]["bleu"])
        print('\t', "Token F1", metrics[name]["token_f1"][0])
        print('\t', "LUAR Sim.", metrics[name]["luar_similarity"][0])
        print('\t', "SBERT Sim.", metrics[name]["sbert_similarity"][0])

    df = pd.DataFrame.from_dict(metrics, orient="index")
    # df.to_json("results.json", orient="index")

if __name__ == "__main__":
    sys.exit(main())