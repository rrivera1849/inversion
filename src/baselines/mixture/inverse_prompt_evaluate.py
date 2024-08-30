
import os
import random
random.seed(43)

from sentence_transformers import SentenceTransformer, util

import evaluate
import pandas as pd

def string_exact_match(s1: str, s2: str) -> bool:
    return s1 == s2

rouge = evaluate.load('rouge')
model = SentenceTransformer('all-MiniLM-L6-v2')
metrics = {}
it = 0
for filename in os.listdir("./prompting_data/"):
    if "inverse" not in filename and "rephrase" not in filename:
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

    results = rouge.compute(predictions=candidates, references=references)
    name = os.path.basename(filename).split(".")[0]
    metrics[name] = {}
    metrics[name]["rouge1"] = results["rouge1"]
    metrics[name]["rouge2"] = results["rouge2"]
    metrics[name]["rougeL"] = results["rougeL"]

    # Compute SBERT similarity
    embeddings1 = model.encode(candidates, convert_to_tensor=True)
    embeddings2 = model.encode(references, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    avg_sbert_similarity = cosine_scores.diag().mean().item()
    metrics[name]["sbert_similarity"] = avg_sbert_similarity

    if it == 0:
        # Compute SBERT similarity for random shuffle
        embeddings1_random = model.encode(candidates, convert_to_tensor=True)
        embeddings2_random = model.encode(references, convert_to_tensor=True)
        cosine_scores_random = util.pytorch_cos_sim(embeddings1_random, embeddings2_random)
        avg_sbert_similarity_random = cosine_scores_random.diag().mean().item()
        metrics["random"]["sbert_similarity"] = avg_sbert_similarity_random
        random.shuffle(candidates)
        metrics["random"] = {}
        results = rouge.compute(predictions=candidates, references=references)
        metrics["random"]["rouge1"] = results["rouge1"]
        metrics["random"]["rouge2"] = results["rouge2"]
        metrics["random"]["rougeL"] = results["rougeL"]
        
    it += 1
    
df = pd.DataFrame.from_dict(metrics, orient="index")
print(df.to_markdown())
df.to_json("inverse_prompt_results.json", orient="index")
