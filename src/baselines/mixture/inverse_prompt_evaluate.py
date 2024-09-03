
import os
import random
random.seed(43)

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer
import evaluate
import pandas as pd

def string_exact_match(s1: str, s2: str) -> bool:
    return s1 == s2

rouge = evaluate.load('rouge')
sbert = SentenceTransformer('all-mpnet-base-v2')
luar = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
luar.to("cuda:0")
luar.eval()
luar_tok = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)

# cisr = AutoModel.from_pretrained("AnnaWegmann/Style-Embedding")
# cisr.to("cuda:1")
# cisr.eval()
# cisr_tok = AutoTokenizer.from_pretrained("AnnaWegmann/Style-Embedding")

metrics = {}
it = 0

@torch.no_grad()
def calculate_luar_embeddings(text: str):
    inputs = luar_tok(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(luar.device)
    
    N = len(text)
    inputs["input_ids"] = inputs["input_ids"].reshape(N, 1, 512)
    inputs["attention_mask"] = inputs["attention_mask"].reshape(N, 1, 512)
    
    outputs = luar(**inputs)
    outputs = F.normalize(outputs, dim=-1, p=2)
    return outputs

# def calculate_cisr_embeddings(text: str):
#     inputs = cisr_tok(
#         text,
#         max_length=512,
#         padding="max_length",
#         truncation=True,
#         return_tensors="pt",
#     ).to(cisr.device)
    
#     outputs = cisr(**inputs)
#     outputs = F.normalize(outputs, dim=-1, p=2)
#     return outputs

def calculate_luar_similarity(candidates, references):
    embeddings1 = calculate_luar_embeddings(candidates)
    embeddings2 = calculate_luar_embeddings(references)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    avg_luar_similarity = cosine_scores.diag().mean().item()
    return avg_luar_similarity

# def calculate_cisr_similarity(candidates, references):
#     embeddings1 = calculate_cisr_embeddings(candidates)
#     embeddings2 = calculate_cisr_embeddings(references)
#     cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
#     avg_cisr_similarity = cosine_scores.diag().mean().item()
#     return avg_cisr_similarity

def calculate_sbert_similarity(candidates, references):
    embeddings1 = sbert.encode(candidates, convert_to_tensor=True)
    embeddings2 = sbert.encode(references, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    avg_sbert_similarity = cosine_scores.diag().mean().item()
    return avg_sbert_similarity

import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
def get_token_overlap(text1, text2):
    tokens1 = enc.encode(text1)
    tokens2 = enc.encode(text2)
    overlap = len(set(tokens1).intersection(set(tokens2))) / len(set(tokens1).union(set(tokens2)))
    return overlap

for filename in os.listdir("./prompting_data/"):

    if "inverse" not in filename and "rephrase" not in filename:
        continue
    
    if "inverse" in filename and "gpt-4o-mini" in filename:
        continue
    if "rephrase" in filename and "gpt-4o-mini" in filename:
        continue
    
    print("Evaluating", filename)
    filename = os.path.join("./prompting_data/", filename)
    df = pd.read_json(filename, lines=True)

    if "inverse" in filename:
        candidates = df.inverse.tolist()
        references = df.unit.tolist()
    else:
        candidates = df.rephrase.tolist()
        references = df.unit.tolist()

    name = os.path.basename(filename).split(".")[0]

    metrics[name] = {}
    # results = rouge.compute(predictions=candidates, references=references)
    # metrics[name]["rouge1"] = results["rouge1"]
    # metrics[name]["rouge2"] = results["rouge2"]
    # metrics[name]["rougeL"] = results["rougeL"]
    # metrics[name]["sbert_similarity"] = calculate_sbert_similarity(candidates, references)
    # metrics[name]["luar_similarity"] = calculate_luar_similarity(candidates, references)
    metrics[name]["token_overlap"] = sum([get_token_overlap(c, r) for c, r in zip(candidates, references)]) / len(candidates)
    # metrics[name]["cisr_similarity"] = calculate_cisr_similarity(candidates, references)
    # metrics[name]["exact_match"] = sum([string_exact_match(c, r) for c, r in zip(candidates, references)]) / len(candidates)

    # if it == 0:
    #     random.shuffle(candidates)
    #     metrics["random"] = {}
    #     results = rouge.compute(predictions=candidates, references=references)
    #     metrics["random"]["rouge1"] = results["rouge1"]
    #     metrics["random"]["rouge2"] = results["rouge2"]
    #     metrics["random"]["rougeL"] = results["rougeL"]
    #     metrics["random"]["sbert_similarity"] = calculate_sbert_similarity(candidates, references)
    #     metrics["random"]["luar_similarity"] = calculate_luar_similarity(candidates, references)
        # metrics["random"]["cisr_similarity"] = calculate_cisr_similarity(candidates, references)
        # metrics["random"]["exact_match"] = sum([string_exact_match(c, r) for c, r in zip(candidates, references)]) / len(candidates)
        
    it += 1
    
df = pd.DataFrame.from_dict(metrics, orient="index")
print(df.to_markdown())
df.to_json("inverse_prompt_results.json", orient="index")
