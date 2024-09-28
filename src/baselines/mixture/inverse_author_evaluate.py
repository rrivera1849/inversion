
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import util
from sklearn.metrics import roc_curve
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

def read_data(path: str):
    df = pd.read_json(path, lines=True)

    path_test_full = "/data1/yubnub/changepoint/MUD_inverse/data/data.jsonl.filtered.cleaned_kmeans_100/test.jsonl"
    df_test_full = pd.read_json(path_test_full, lines=True)
    df_test_full = df_test_full.groupby("author_id").agg(list).iloc[:100]
    to_explode = [col for col in df_test_full.columns if col != "author_id"]
    df_test_full = df_test_full.explode(to_explode).reset_index()
    assert df_test_full.unit.tolist() == df.unit.tolist()
    author_ids = df_test_full.author_id.tolist()
    df["author_id"] = author_ids
    df.drop_duplicates("unit", inplace=True)
    
    return df

luar = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
luar.to("cuda").eval()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
luar_tok = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)

@torch.no_grad()
def get_luar_author_embeddings(
    text: list[str],
):
    # output: tensor of shape (1, 512)
    assert isinstance(text, list)
    inputs = luar_tok(
        text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(luar.device)
    inputs["input_ids"] = inputs["input_ids"].view(1, len(text), 512)
    inputs["attention_mask"] = inputs["attention_mask"].view(1, len(text), 512)
    outputs = luar(**inputs)
    outputs = F.normalize(outputs, p=2, dim=-1)
    return outputs

@torch.no_grad()
def get_luar_instance_embeddings(
    text: list[str],
    batch_size: int = 32,
):
    # output: tensor of shape (len(text), 512)
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
    all_outputs = F.normalize(all_outputs, p=2, dim=-1)
    return all_outputs

def calculate_EER(labels, sims):
    fpr, tpr, _ = roc_curve(labels, sims, pos_label=1)
    fnr = 1 - tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return EER

def max_similarity(
    embeddings1: torch.Tensor,  
    embeddings2: torch.Tensor,
):
    embeddings1 = embeddings1.repeat(embeddings2.size(0), 1)
    similarities = util.pytorch_cos_sim(embeddings1, embeddings2).diag().cpu().tolist()
    maxsim = max(similarities)
    return maxsim

def expected_similarity(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor,
):
    embeddings1 = embeddings1.repeat(embeddings2.size(0), 1)
    similarities = util.pytorch_cos_sim(embeddings1, embeddings2).diag().cpu().tolist()
    avg_cosine_similarity = sum(similarities) / len(similarities)
    return avg_cosine_similarity

def get_author_instance_labels(
    df: pd.DataFrame,
    N: int,
    use_inverse: bool = False,
):
    if use_inverse:
        counts = df["inverse"].apply(len).tolist()
    else:
        counts = df["rephrase"].apply(len).tolist()

    counts = np.cumsum(counts)

    last = 0
    labels = []
    for c in counts:
        l = [0] * N
        l[last:c] = [1] * (c - last)
        last = c
        labels.extend(l)
    return labels

def flatten(lst):
    return [item for sublist in lst for item in sublist]

def pairwise_similarity(
    query_embeddings: torch.Tensor,
    inverse_embeddings: list[torch.Tensor],
    type: str = "expected",
):
    # TODO: parallelize this
    assert type in ["max", "expected"]
    type_to_func = {
        "max": max_similarity,
        "expected": expected_similarity,
    }

    similarities = []
    pbar = tqdm(total=len(query_embeddings) * len(inverse_embeddings))
    for i in range(len(query_embeddings)):
        for j in range(len(inverse_embeddings)):
            similarities.append(type_to_func[type](query_embeddings[i:i+1], inverse_embeddings[j]))
            pbar.update(1)
    return similarities

def calculate_metric(
    path: str,
    mode: str = "plagiarism",
):
    assert mode in ["plagiarism", "author"]
    
    metrics = {}
    df = read_data(path)
    df = df.groupby("author_id").agg(list)
    
    if mode == "plagiarism":
        # query = target
        # task: plagiarism detection
        df["unit"] = df["unit"].apply(lambda x: x[:len(x)//2])
        df["rephrase"] = df["rephrase"].apply(lambda x: x[:len(x)//2])
        df["inverse"] = df["inverse"].apply(lambda x: x[:len(x)//2])
    else:
        # query != target
        # task: author identification
        df["unit"] = df["unit"].apply(lambda x: x[:len(x)//2])
        df["rephrase"] = df["rephrase"].apply(lambda x: x[len(x)//2:])
        df["inverse"] = df["inverse"].apply(lambda x: x[len(x)//2:])
    
    num_author = len(df)
    num_instances = df.rephrase.apply(len).sum()

    author_author_labels = np.identity(num_author, dtype=np.int32).flatten().tolist()
    instance_instance_labels = np.identity(num_instances, dtype=np.int32).flatten().tolist()
    author_instance_labels = get_author_instance_labels(df, num_instances)

    # Compute Author Query
    query_author_embeddings = [get_luar_author_embeddings(unit) for unit in tqdm(df.unit.tolist())]
    query_author_embeddings = torch.cat(query_author_embeddings, dim=0).cpu()
    
    # Compute Instance Query
    query_instance_embeddings = [get_luar_instance_embeddings(unit) for unit in tqdm(df.unit.tolist())]
    query_instance_embeddings = torch.cat(query_instance_embeddings, dim=0).cpu()

    ##### Rephrases:
    rephrase_instance_embeddings = [get_luar_instance_embeddings(rephrases) for rephrases in tqdm(df.rephrase.tolist())]
    rephrase_instance_embeddings = torch.cat(rephrase_instance_embeddings, dim=0).cpu()
    rephrase_author_embeddings = [get_luar_author_embeddings(rephrases) for rephrases in tqdm(df.rephrase.tolist())]
    rephrase_author_embeddings = torch.cat(rephrase_author_embeddings, dim=0).cpu()
    assert len(rephrase_instance_embeddings) == num_instances
    assert len(rephrase_author_embeddings) == num_author
    
    # Author-Instance:
    similarities = util.pytorch_cos_sim(query_author_embeddings, rephrase_instance_embeddings)
    similarities = similarities.cpu().flatten().tolist()
    metrics["EER_rephrase_author-instance"] = calculate_EER(author_instance_labels, similarities)


    # Author-Author
    similarities = util.pytorch_cos_sim(query_author_embeddings, rephrase_author_embeddings)
    similarities = similarities.cpu().flatten().tolist()
    metrics["EER_rephrase_author-author"] = calculate_EER(author_author_labels, similarities)

    if mode == "plagiarism":
        # Instance-Instance:
        similarities = util.pytorch_cos_sim(query_instance_embeddings, rephrase_instance_embeddings)
        similarities = similarities.cpu().flatten().tolist()
        metrics["EER_rephrase_instance-instance"] = calculate_EER(instance_instance_labels, similarities)

    ##### Inverse (All):
    inverse_all_author_embeddings = [get_luar_author_embeddings(flatten(inverses)) for inverses in tqdm(df.inverse.tolist())]
    inverse_all_author_embeddings = torch.cat(inverse_all_author_embeddings, dim=0).cpu()
    inverse_all_instance_embeddings = [[get_luar_author_embeddings(inverse).cpu() for inverse in inverses] for inverses in tqdm(df.inverse.tolist())]
    inverse_all_instance_embeddings = [torch.cat(inverses, dim=0) for inverses in inverse_all_instance_embeddings]
    inverse_all_instance_embeddings = torch.cat(inverse_all_instance_embeddings, dim=0).cpu()
    assert len(inverse_all_instance_embeddings) == num_instances
    assert len(inverse_all_author_embeddings) == num_author

    # Author-Instance:
    similarities = util.pytorch_cos_sim(query_author_embeddings, inverse_all_instance_embeddings)
    similarities = similarities.cpu().flatten().tolist()
    metrics[f"EER_all_author-instance"] = calculate_EER(author_instance_labels, similarities)
    
    # Author-Author
    similarities = util.pytorch_cos_sim(query_author_embeddings, inverse_all_author_embeddings)
    similarities = similarities.cpu().flatten().tolist()
    metrics[f"EER_all_author-author"] = calculate_EER(author_author_labels, similarities)

    if mode == "plagiarism":
        # Instance-Instance:
        similarities = util.pytorch_cos_sim(query_instance_embeddings, inverse_all_instance_embeddings)
        similarities = similarities.cpu().flatten().tolist()
        metrics[f"EER_all_instance-instance"] = calculate_EER(instance_instance_labels, similarities)
    
    ##### Inverse (Individual Embedding):
    inverse_instance_embeddings = [[get_luar_instance_embeddings(inverse).cpu() for inverse in inverses] for inverses in tqdm(df.inverse.tolist())]
    inverse_author_embeddings = [torch.cat(inverses, dim=0) for inverses in inverse_instance_embeddings]
    inverse_instance_embeddings = flatten(inverse_instance_embeddings)
    assert len(inverse_instance_embeddings) == num_instances
    assert len(inverse_author_embeddings) == num_author
    for simtype in ["expected", "max"]:
        similarities = pairwise_similarity(query_author_embeddings, inverse_instance_embeddings, type=simtype)
        metrics[f"EER_inverse-{simtype}_author-instance"] = calculate_EER(author_instance_labels, similarities)
        
        similarities = pairwise_similarity(query_author_embeddings, inverse_author_embeddings, type=simtype)
        metrics[f"EER_inverse-{simtype}_author-author"] = calculate_EER(author_author_labels, similarities)

        if mode == "plagiarism":
            similarities = pairwise_similarity(query_instance_embeddings, inverse_instance_embeddings, type=simtype)
            metrics[f"EER_inverse-{simtype}_instance_instance"] = calculate_EER(instance_instance_labels, similarities)

    ##### Inverse (Single Inversion):
    df["single_inversion"] = df.inverse.apply(lambda xx: [x[0] for x in xx])
    inverse_single_instance_embeddings = [get_luar_instance_embeddings(inverses) for inverses in tqdm(df.single_inversion.tolist())]
    inverse_single_instance_embeddings = torch.cat(inverse_single_instance_embeddings, dim=0).cpu()
    inverse_single_author_embeddings = [get_luar_author_embeddings(inverses) for inverses in tqdm(df.single_inversion.tolist())]
    inverse_single_author_embeddings = torch.cat(inverse_single_author_embeddings, dim=0).cpu()
    assert len(inverse_single_instance_embeddings) == num_instances
    assert len(inverse_single_author_embeddings) == num_author

    # Author-Instance:
    similarities = util.pytorch_cos_sim(query_author_embeddings, inverse_single_instance_embeddings)
    similarities = similarities.cpu().flatten().tolist()
    metrics[f"EER_single_author-instance"] = calculate_EER(author_instance_labels, similarities)
    
    # Author-Author
    similarities = util.pytorch_cos_sim(query_author_embeddings, inverse_single_author_embeddings)
    similarities = similarities.cpu().flatten().tolist()
    metrics[f"EER_single_author-author"] = calculate_EER(author_author_labels, similarities)

    if mode == "plagiarism":
        # Instance-Instance:
        similarities = util.pytorch_cos_sim(query_instance_embeddings, inverse_single_instance_embeddings)
        similarities = similarities.cpu().flatten().tolist()
        metrics[f"EER_single_instance-instance"] = calculate_EER(instance_instance_labels, similarities)

    return metrics

os.makedirs("./metrics", exist_ok=True)
base_path = "/data1/yubnub/changepoint/MUD_inverse/data/data.jsonl.filtered.cleaned_kmeans_100/inverse_output"
files = [
    "none_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=100",
    "none_6400_temperature=0.3_top_p=0.9.jsonl.vllm_n=100",
    "none_6400_temperature=1.5_top_p=0.9.jsonl.vllm_n=100",
]
for file in files:
    path = os.path.join(base_path, file)
    metrics_plagiarism = calculate_metric(path)
    metrics_author = calculate_metric(path, "author")
    
    print(f"File: {file}")
    print("Plagiarism")
    print(metrics_plagiarism)
    print("Author")
    print(metrics_author)
    print()
    
    import pdb; pdb.set_trace()
    name = file[:-len(".jsonl.vllm_n=100")]
    with open(f"./metrics/{name}_plagiarism.json", "w") as f:
        json.dump(metrics_plagiarism, f)
    with open(f"./metrics/{name}_author.json", "w") as f:
        json.dump(metrics_author, f)