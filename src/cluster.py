
import json
import os
import sys
from typing import List
from collections import Counter
from itertools import chain
from time import time

import torch
from kmeans_pytorch import kmeans
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

import file_utils as fu
import nn_utils as nnu
import search_utils as su

DEBUG = True

def get_changepoints(
    author_partition: List[int]
):
    """Returns a list of 1s and 0s, where 1 indicates a change in the author partition.
    """
    changes = []
    for i in range(1, len(author_partition)):
        if author_partition[i] != author_partition[i-1]:
            changes.append(1)
        else:
            changes.append(0)
    return changes

def main():
    start = time()

    samples = fu.read_PAN_dataset(fu.PAN23_paths["hard"])
    savedir = "./results/clustering/PAN23_hard"
    savedir_majority = os.path.join(savedir, "majority")
    savedir_largest_score = os.path.join(savedir, "largest_score")
    os.makedirs(savedir_majority, exist_ok=True)
    os.makedirs(savedir_largest_score, exist_ok=True)

    # TODO: Train larger IUR, and use it here:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "uar"
    LUAR_id = "/data1/yubnub/pretrained_weights/LUAR/LUAR-IUR"
    model = AutoModel.from_pretrained(LUAR_id, trust_remote_code=True)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(LUAR_id)
    
    num_trials = 100    # number of clustering trials
    pred_majority, pred_largest_score, ground_truth = [], [], []
    
    if DEBUG:
        to_print = []
        max_samples = 10
    else:
        max_samples = len(samples)

    for i in tqdm(range(max_samples)):
        text = samples[i]["text"]
        true_changes = samples[i]["changes"]
        
        # 1. get the embeddings under some style model
        embeddings = torch.cat(
            [nnu.get_embedding(sample, model, tokenizer, device, model_name) for sample in text],
            dim=0
        )
        
        # 2. cluster the embeddings using k-means `num_trials` times
        all_predicted_partitions, all_proposal_scores = [], []
        for _ in range(num_trials):
            # 2.1 cluster:
            author_partition, _ =  kmeans(
                embeddings, 
                num_clusters=samples[i]["authors"], 
                distance="cosine", 
                device=device
            )
            author_partition = author_partition.cpu().numpy().tolist()

            # 2.2 get proposal score:
            proposal_score = su.get_proposal_score(
                text, author_partition, 
                model, tokenizer, device, 
                model_name=model_name
            )

            all_predicted_partitions.append((tuple(get_changepoints(author_partition)), tuple(author_partition), proposal_score))
            all_proposal_scores.append(proposal_score)
            
        # 3. get the majority vote and the largest proposal score
        changes_majority = Counter(all_predicted_partitions).most_common(1)[0][0][0]
        changes_largest_score = [change for change, _, score in all_predicted_partitions if score == max(all_proposal_scores)][0]

        pred_majority.append(changes_majority)
        pred_largest_score.append(changes_largest_score)
        ground_truth.append(true_changes)
        
        if DEBUG:
            debug_s = f"Majority Vote: {changes_majority}\tLargest Proposal Score: {changes_largest_score}\tTrue Changes: {true_changes}"
            to_print.append(debug_s)

    if DEBUG:
        for p in to_print:
            print(p)

    f1_majority = f1_score(list(chain.from_iterable(ground_truth)), list(chain.from_iterable(pred_majority)), labels=[0, 1], average="macro")
    f1_largest_score = f1_score(list(chain.from_iterable(ground_truth)), list(chain.from_iterable(pred_largest_score)), labels=[0, 1], average="macro")
    print(f"F1 Score Majority Vote: {f1_majority:.2f}\tF1 Score Largest Proposal Score: {f1_largest_score:.2f}")

    for i in range(max_samples):
        with open(os.path.join(savedir_majority, f"solution-problem-{i+1}.json"), "w") as f:
            f.write(json.dumps({"changes": list(pred_majority[i])}))
        with open(os.path.join(savedir_largest_score, f"solution-problem-{i+1}.json"), "w") as f:
            f.write(json.dumps({"changes": list(pred_largest_score[i])}))
        
    print(f"Total Time Taken: {time() - start:.2f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())