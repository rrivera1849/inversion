
import os

import pandas as pd
import torch

from embedding_utils import (
    load_luar_model_and_tokenizer,
    get_instance_embeddings,
    get_author_embeddings,
)
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

DATA_DIR = "/data1/foobar/changepoint/MUD_inverse/data"

abstracts_filenames = [
    "test_final_none_3000_temperature=0.7_top_p=0.9.jsonl.vllm_n=100-edit-detector",
    "test_final_none_3000_temperature=0.7_top_p=0.9.jsonl.vllm_n=100-neural-pred-abstracts",
]

reviews_filenames = [
    "test_final_none_1600_temperature=0.7_top_p=0.9.jsonl.vllm_n=100-edit-detector",
    "test_final_none_1600_temperature=0.7_top_p=0.9.jsonl.vllm_n=100-neural-pred-reviews",
]

reddit_filenames = [
    "MTD_all_none_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=100-with-preds",
    "MTD_all_none_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=100-neural-pred-data.jsonl.filtered.respond_reddit.cleaned",
]

data = {
    "data.jsonl.filtered.respond_reddit.cleaned": reddit_filenames,
    "abstracts": abstracts_filenames,
    "reviews": reviews_filenames,
}

luar, luar_tok = load_luar_model_and_tokenizer()
luar.cuda()
cossim = torch.nn.CosineSimilarity(dim=-1)

function_kwargs = {
    "luar": luar,
    "luar_tok": luar_tok,
}

debug = False

for dataset_name in data.keys():
    try:
        valid_fname = os.path.join(DATA_DIR, dataset_name, "valid_final.jsonl")
        valid = pd.read_json(valid_fname, lines=True)
    except:
        valid_fname = os.path.join(DATA_DIR, dataset_name, "inverse_output", "valid_with_all_none_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=1")
        valid = pd.read_json(valid_fname, lines=True)
        
    valid = valid[~((valid.is_machine) | (valid.is_human_paraphrase))]
    valid = valid.rephrase.tolist()[:1000]
    
    background_emb = get_author_embeddings(valid, function_kwargs, "mud")

    files_to_evaluate = data[dataset_name]

    for fname in files_to_evaluate:
        path = os.path.join(DATA_DIR, dataset_name, "inverse_output", fname)
        df = pd.read_json(path, lines=True)
        df = df.sample(frac=1.).reset_index(drop=True)
        
        text = []
        for _, row in df.iterrows():
            if row["use_inverse"] == True:
                text.append(row["inverse"])
            else:
                text.append([row["rephrase"]])

        if debug:
            text = text[:100]

        embeddings = [get_author_embeddings(t, function_kwargs, "mud") for t in tqdm(text)]
        embeddings = torch.cat(embeddings, dim=0)
        labels = df.is_machine.tolist()
        if debug:
            labels = labels[:100]
        scores = cossim(embeddings, background_emb).tolist()
        auc = roc_auc_score(labels, scores)
        print(dataset_name, fname, "{:.2f}".format(auc))

    # now get baseline which is with all rephrases
    text = df.rephrase.tolist()
    if debug:
        text = text[:100]
    embeddings = get_instance_embeddings(text, function_kwargs, "mud")
    labels = df.is_machine.tolist()
    if debug:
        labels = labels[:100]
    scores = cossim(embeddings, background_emb).tolist()
    auc = roc_auc_score(labels, scores)
    print(dataset_name, "baseline", "{:.2f}".format(auc))