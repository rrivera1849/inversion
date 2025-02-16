
import json
import os
from typing import Union

import mauve
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import umap
import umap.plot
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

AUTHOR_DATA_PATH = "/data1/foobar/data/iur_dataset/author_100.politics"
TOKENIZER = AutoTokenizer.from_pretrained("roberta-large")

device = "cuda" if torch.cuda.is_available() else "cpu"
# sbert = SentenceTransformer("all-mpnet-base-v2")
# sbert.to(device)
# sbert.eval()

luar = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
luar.to(device)
luar.eval()
luar_tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD")

def get_uar_embedding(data: list[str], batch_size=128):
    all_out = []
    for batch_start in tqdm(range(0, len(data), batch_size)):
        batch = data[batch_start:batch_start + batch_size]
        tok = luar_tokenizer.batch_encode_plus(
            batch,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        tok["input_ids"] = tok["input_ids"].reshape(-1, 1, 512).to(device)
        tok["attention_mask"] = tok["attention_mask"].reshape(-1, 1, 512).to(device)
        with torch.inference_mode():
            out = luar(**tok)
            out = F.normalize(out, p=2.0)
        all_out.append(out)
    all_out = torch.cat(all_out, dim=0)
    return all_out

def load_author_data(
    name,
):
    """Loads a split from the Author 100 Politics dataset.
    
       TODO: copied from inverse_inference.py
    """
    with open(os.path.join(AUTHOR_DATA_PATH, name)) as fin:
        data = [json.loads(line) for line in fin]
    return data

def get_cossim(emb_1, emb_2, average=True):
    cossim = torch.nn.CosineSimilarity(dim=-1)
    if average:
        out = cossim(emb_1, emb_2).mean().item()
        return out
    else:
        out = cossim(emb_1, emb_2)
        return out

for split_name in ["train"]:
    reference_data = load_author_data("test.jsonl")
    reference_data = [data["syms"] for data in reference_data]
    
    mistral_data = load_author_data("test.jsonl.mistral")
    mistral_data = [data["syms"] for data in mistral_data]

    inverse_data = load_author_data("test.jsonl.mistral.inverse")
    inverse_data = [data["syms"] for data in inverse_data]
    inverse_data = [text.strip() for text in inverse_data]
    
    inverse_with_mixture_data = load_author_data("test.jsonl.mistral.inverse-mixture")
    inverse_with_mixture_data = [data["syms"] for data in inverse_with_mixture_data]
    inverse_with_mixture_data = [text.strip() for text in inverse_with_mixture_data]

    inverse_with_simple_mixture = load_author_data("test.jsonl.mistral.inverse-mixture-simple")
    inverse_with_simple_mixture = [data["syms"] for data in inverse_with_simple_mixture]
    inverse_with_simple_mixture = [text.strip() for text in inverse_with_simple_mixture]

    # mauve_mistral = mauve.compute_mauve(p_text=reference_data, q_text=mistral_data, device_id=0, max_text_length=512, verbose=False)
    # mauve_inverse = mauve.compute_mauve(p_text=reference_data, q_text=inverse_data, device_id=0, max_text_length=512, verbose=False)
    # mauve_inverse_with_mixture = mauve.compute_mauve(p_text=reference_data, q_text=inverse_with_mixture_data, device_id=0, max_text_length=512, verbose=False)
    
    # print(f"MAUVE Mistral: {mauve_mistral.mauve:.2f}")
    # print(f"MAUVE Inverse: {mauve_inverse.mauve:.2f}")
    # print(f"MAUVE Inverse with Mixture: {mauve_inverse_with_mixture.mauve:.2f}")
    
    luar_reference = get_uar_embedding(reference_data)
    luar_mistral = get_uar_embedding(mistral_data)
    luar_inverse = get_uar_embedding(inverse_data)
    luar_inverse_with_mixture = get_uar_embedding(inverse_with_mixture_data)
    luar_inverse_with_simple_mixture = get_uar_embedding(inverse_with_simple_mixture)

    luar_reference_centroid = luar_reference.mean(dim=0, keepdim=True).repeat(luar_reference.size(0), 1)

    mistral_sim = get_cossim(luar_reference_centroid, luar_mistral, average=False)
    inverse_sim = get_cossim(luar_reference_centroid, luar_inverse, average=False)
    inverse_with_mixture_sim = get_cossim(luar_reference_centroid, luar_inverse_with_mixture, average=False)
    inverse_with_simple_mixture_sim = get_cossim(luar_reference_centroid, luar_inverse_with_simple_mixture, average=False)
    inverse_success = (inverse_sim > mistral_sim).float().mean()
    inverse_with_mixture_success = (inverse_with_mixture_sim > mistral_sim).float().mean()
    inverse_with_simple_mixture_success = (inverse_with_simple_mixture_sim > mistral_sim).float().mean()
    print(f"Inverse Success: {inverse_success:.2f}")
    print(f"Inverse with Mixture Success: {inverse_with_mixture_success:.2f}")
    print(f"Inverse with Simple Mixture Success: {inverse_with_simple_mixture_success:.2f}")

    all_embeddings = torch.cat([luar_reference, luar_mistral, luar_inverse, luar_inverse_with_mixture, luar_inverse_with_simple_mixture], dim=0)
    all_embeddings = all_embeddings.cpu().detach().numpy()
    labels = ["Reference"] * len(reference_data) + ["Mistral"] * len(mistral_data) + ["Inverse"] * len(inverse_data) + ["Inverse with Mixture"] * len(inverse_with_mixture_data)
    mapper = umap.UMAP(metric="cosine").fit(all_embeddings)
    umap.plot.points(mapper, labels=np.array(labels))
    plt.savefig(f"umap_with_inverses.png")
    
    # print(f"Mean cosine similarity for Mistral: {get_cossim(luar_reference, luar_mistral):.2f}")
    # print(f"Mean cosine similarity for inverse: {get_cossim(luar_reference, luar_inverse):.2f}")
    # print(f"Mean cosine similarity for inverse with mixture: {get_cossim(luar_reference, luar_inverse_with_mixture):.2f}")

    # sbert_reference = sbert.encode(reference_data, show_progress_bar=True, convert_to_tensor=True)
    # sbert_mistral = sbert.encode(mistral_data, show_progress_bar=True, convert_to_tensor=True)
    # sbert_inverse = sbert.encode(inverse_data, show_progress_bar=True, convert_to_tensor=True)
    # sbert_inverse_with_mixture = sbert.encode(inverse_with_mixture_data, show_progress_bar=True, convert_to_tensor=True)

    # print(f"Mean cosine similarity for Mistral: {get_cossim(sbert_reference, sbert_mistral):.2f}")
    # print(f"Mean cosine similarity for inverse: {get_cossim(sbert_reference, sbert_inverse):.2f}")
    # print(f"Mean cosine similarity for inverse with mixture: {get_cossim(sbert_reference, sbert_inverse_with_mixture):.2f}")

    # mistral_token_overlaps = []
    # inverse_token_overlaps = []
    # inverse_with_mixture_token_overlaps = []
    # for reference, mistral, inverse, inverse_with_mixture in zip(reference_data, mistral_data, inverse_data, inverse_with_mixture_data):
    #     reference = TOKENIZER.tokenize(reference)
    #     mistral = TOKENIZER.tokenize(mistral)
    #     inverse = TOKENIZER.tokenize(inverse)
    #     inverse_with_mixture = TOKENIZER.tokenize(inverse_with_mixture)

    #     token_overlap_mistral = len(set.intersection(set(reference), set(mistral))) / min(len(set(mistral)), len(set(reference)))
    #     token_overlap_inverse = len(set.intersection(set(reference), set(inverse))) / min(len(set(inverse)), len(set(reference)))
    #     token_overlap_inverse_with_mixture = len(set.intersection(set(reference), set(inverse_with_mixture))) / min(len(set(inverse_with_mixture)), len(set(reference)))

    #     mistral_token_overlaps.append(token_overlap_mistral)
    #     inverse_token_overlaps.append(token_overlap_inverse)
    #     inverse_with_mixture_token_overlaps.append(token_overlap_inverse_with_mixture)

    
    # _ = plt.figure()
    # plt.hist(mistral_token_overlaps, bins=20, alpha=0.5, label="Mistral Rephrase")
    # plt.hist(inverse_with_mixture_token_overlaps, bins=20, alpha=0.5, label="Inverse with Mixture")
    # plt.legend()
    # plt.title("Token Overlap of Inverse Rephrases against Human Reference")
    # plt.savefig(f"token_overlap_inverse_with_mixture_{split_name}.png")
    # plt.close()

    # _ = plt.figure()
    # plt.hist(mistral_token_overlaps, bins=20, alpha=0.5, label="Mistral Rephrase")
    # plt.hist(inverse_token_overlaps, bins=20, alpha=0.5, label="Inverse")
    # plt.legend()
    # plt.title("Token Overlap of Inverse Rephrases against Human Reference")
    # plt.savefig(f"token_overlap_inverse_{split_name}.png")
    # plt.close()
    
    # print(f"Mean token overlap for Mistral: {np.mean(mistral_token_overlaps):.2f}")
    # print(f"Mean token overlap for inverse: {np.mean(inverse_token_overlaps):.2f}")
    # print(f"Mean token overlap for inverse with mixture: {np.mean(inverse_with_mixture_token_overlaps):.2f}")

    # print(f"Median token overlap for Mistral: {np.median(mistral_token_overlaps):.2f}")
    # print(f"Median token overlap for inverse: {np.median(inverse_token_overlaps):.2f}")
    # print(f"Median token overlap for inverse with mixture: {np.median(inverse_with_mixture_token_overlaps):.2f}")