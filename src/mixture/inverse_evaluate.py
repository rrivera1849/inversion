
import os
import sys
from typing import Union

import evaluate
import nltk
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

bleu = evaluate.load("bleu")
sbert = SentenceTransformer("all-mpnet-base-v2")
luar = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
luar.to("cuda").eval()
luar_tok = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
# cisr = SentenceTransformer("AnnaWegmann/Style-Embedding")
# cisr.to("cuda").eval()
# cisr_tok = AutoTokenizer.from_pretrained("AnnaWegmann/Style-Embedding")

def cosine_similarity(
    embeddings_1: Union[torch.Tensor, list[torch.Tensor]],
    embeddings_2: torch.Tensor,
    type: str = "expected",
):
    # assuming `embeddings_1` are the inversions and `embeddings_2` are the references
    if isinstance(embeddings_1, list):
        assert len(embeddings_1) == len(embeddings_2)

        cosine_scores = []
        for j in range(len(embeddings_1)):
            emb1 = embeddings_1[j]
            emb2 = embeddings_2[j:j+1].repeat(emb1.size(0), 1)
            similarities = util.pytorch_cos_sim(emb1, emb2).diag().cpu().tolist()
            if type == "expected":
                similarity = sum(similarities) / len(similarities)
            elif type == "max":
                similarity = max(similarities)
            elif type == "first":
                similarity = similarities[0]
            else:
                raise ValueError(f"Type {type} not supported.")
            
            cosine_scores.append(similarity)
    else:
        cosine_scores = util.pytorch_cos_sim(embeddings_1, embeddings_2).diag().cpu().tolist()

    avg_cosine_similarity = sum(cosine_scores) / len(cosine_scores)
    
    return avg_cosine_similarity, cosine_scores

@torch.no_grad()
def calculate_sbert_similarity(
    candidates: Union[list[str], list[list[str]]], 
    references: list[str],
):
    if isinstance(candidates[0], list):
        N = [len(c) for c in candidates]
        candidates = [j for i in candidates for j in i]
        embeddings1 = sbert.encode(candidates, convert_to_tensor=True, show_progress_bar=True)
        embeddings1 = list(torch.split(embeddings1, N))
    else:
        embeddings1 = sbert.encode(candidates, convert_to_tensor=True, show_progress_bar=True)

    embeddings2 = sbert.encode(references, convert_to_tensor=True)

    similarities = {}
    similarities["expected"] = cosine_similarity(embeddings1, embeddings2, type="expected")
    similarities["max"] = cosine_similarity(embeddings1, embeddings2, type="max")
    similarities["first"] = cosine_similarity(embeddings1, embeddings2, type="first")
    return similarities

# @torch.no_grad()
# def calculate_cisr_similarity(
#     candidates: list[str],
#     references: list[str],
# ):
#     embeddings1 = cisr.encode(candidates)
#     embeddings2 = cisr.encode(references)
#     return cosine_similarity(embeddings1, embeddings2)

@torch.no_grad()
def get_luar_embeddings(
    text: list[str],
    batch_size: int = 1024,
):
    all_outputs = []
    for i in tqdm(range(0, len(text), batch_size)):
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
    candidates: Union[list[str], list[list[str]]],
    references: list[str],
):
    if isinstance(candidates[0], list):
        N = [len(c) for c in candidates]
        candidates = [j for i in candidates for j in i]
        embeddings1 = get_luar_embeddings(candidates)
        embeddings1 = list(torch.split(embeddings1, N))
    else:
        embeddings1 = get_luar_embeddings(candidates)

    embeddings2 = get_luar_embeddings(references)

    similarities = {}
    similarities["expected"] = cosine_similarity(embeddings1, embeddings2, type="expected")
    similarities["max"] = cosine_similarity(embeddings1, embeddings2, type="max")
    similarities["first"] = cosine_similarity(embeddings1, embeddings2, type="first")
    return similarities

def calculate_embedding_similarity(
    candidates: Union[list[str], list[list[str]]],
    references: list[str],
    model_name: str = "sbert",
):
    if model_name == "sbert":
        return calculate_sbert_similarity(candidates, references)
    elif model_name == "luar":
        return calculate_luar_similarity(candidates, references)
    # elif model_name == "cisr":
    #     return calculate_cisr_similarity(candidates, references)
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
    dataset_path = "/data1/foobar/changepoint/MUD_inverse/data/data.jsonl.filtered.cleaned_kmeans_100/inverse_output"
    for filename in os.listdir(dataset_path):
        path = os.path.join(dataset_path, filename)
        # if not path.endswith(".jsonl"):
            # continue
        if "vllm" not in filename:
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
        # metrics[name]["bleu"] = bleu.compute(predictions=candidates, references=references)["bleu"]
        # metrics[name]["token_f1"] = token_f1(candidates, references)
        metrics[name]["sbert_similarity"] = calculate_embedding_similarity(candidates, references, "sbert")
        metrics[name]["luar_similarity"] = calculate_embedding_similarity(candidates, references, "luar")

        # print('\t', "BLEU", metrics[name]["bleu"])
        # print('\t', "Token F1", metrics[name]["token_f1"][0])
        print('\t', "LUAR Sim. (Expected)", metrics[name]["luar_similarity"]["expected"][0])
        print('\t', "LUAR Sim. (Max)", metrics[name]["luar_similarity"]["max"][0])
        print('\t', "LUAR Sim. (First)", metrics[name]["luar_similarity"]["first"][0])

        print('\t', "SBERT Sim. (Expected)", metrics[name]["sbert_similarity"]["expected"][0])
        print('\t', "SBERT Sim. (Max)", metrics[name]["sbert_similarity"]["max"][0])
        print('\t', "SBERT Sim. (First)", metrics[name]["sbert_similarity"]["first"][0])

    df = pd.DataFrame.from_dict(metrics, orient="index")
    df.to_json("results.json", orient="index")

if __name__ == "__main__":
    sys.exit(main())