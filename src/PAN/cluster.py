
import json
import os
import random
import sys
from typing import List
from collections import Counter, defaultdict
from itertools import chain
from time import time

import numpy as np
import torch
from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

import file_utils as fu
import nn_utils as nnu
import search_utils as su

DEBUG = True
CALCULATE_BEST_PERFORMANCE = False
SAVE_BEST_PERFORMANCE_TRAINING_DATA = False

CALCULATE_BEST_PERFORMANCE |= SAVE_BEST_PERFORMANCE_TRAINING_DATA

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
    if SAVE_BEST_PERFORMANCE_TRAINING_DATA:
        score_train_savepath = os.path.dirname(fu.PAN23_train_paths["hard"])
        score_train_savepath = os.path.join(score_train_savepath, "train_best_performance_scores.jsonl")
        score_train_savepath_fout = open(score_train_savepath, "w+")

        score_validation_savepath = os.path.dirname(fu.PAN23_validation_paths["hard"])
        score_validation_savepath = os.path.join(score_validation_savepath, "validation_best_performance_scores.jsonl")
        score_validation_savepath_fout = open(score_validation_savepath, "w+")
        
        samples  = fu.read_PAN_dataset(fu.PAN23_train_paths["hard"])
        num_train_samples = len(samples)
        samples += fu.read_PAN_dataset(fu.PAN23_validation_paths["hard"])
    else:
        samples = fu.read_PAN_dataset(fu.PAN23_validation_paths["hard"])

    savedir = "./results/clustering/PAN23_hard"
    if CALCULATE_BEST_PERFORMANCE:
        savedir_best_performance = os.path.join(savedir, "best_performance")
        os.makedirs(savedir_best_performance, exist_ok=True)
    else:
        savedir_majority = os.path.join(savedir, "majority")
        savedir_largest_score = os.path.join(savedir, "largest_score")
        os.makedirs(savedir_majority, exist_ok=True)
        os.makedirs(savedir_largest_score, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "uar"
    LUAR_id = "/data1/foobar/pretrained_weights/LUAR/LUAR-IUR"
    print(f"Loading: {LUAR_id}")
    model = AutoModel.from_pretrained(LUAR_id, trust_remote_code=True)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(LUAR_id)
    
    num_trials = 200    # number of clustering trials
    if CALCULATE_BEST_PERFORMANCE:
        pred_best_performance = []
    else:
        pred_majority, pred_largest_score = [], []
    ground_truth = []
    
    if DEBUG:
        to_print = []
        max_samples = 10
    else:
        max_samples = len(samples)
        
    for i in tqdm(range(max_samples)):
        text = samples[i]["text"]
        true_changes = samples[i]["changes"]
        
        if len(text) != len(true_changes) + 1:
            # some data is buggy
            continue
        
        # 1. get the embeddings under some style model
        embeddings = torch.cat(
            [nnu.get_embedding(sample, model, tokenizer, device, model_name) for sample in text],
            dim=0
        )
        
        # 2. cluster the embeddings using k-means `num_trials` times
        if CALCULATE_BEST_PERFORMANCE:
            all_best_partitions = []
        else:
            all_predicted_partitions, all_proposal_scores = [], []
        for _ in range(num_trials):
            # 2.1 cluster:
            # author_partition, _ =  kmeans(
            #     embeddings,
            #     num_clusters=samples[i]["authors"], 
            #     distance="cosine", 
            #     device=device,
            #     tqdm_flag=False,
            # )
            # author_partition = author_partition.cpu().numpy().tolist()
            author_partition = KMeans(n_clusters=samples[i]["authors"]).fit(embeddings.cpu().numpy()).labels_
            
            # some weirdness in the kmeans implementation:
            if len(np.unique(author_partition)) != samples[i]["authors"]:
                continue
            
            if CALCULATE_BEST_PERFORMANCE:
                all_best_partitions.append(author_partition)
            else:
                # 2.2 get proposal score:
                proposal_score = su.get_proposal_score(
                    text, author_partition,
                    model, tokenizer, device,
                    model_name=model_name,
                )

                all_predicted_partitions.append((tuple(get_changepoints(author_partition)), tuple(author_partition), proposal_score))
                all_proposal_scores.append(proposal_score)

        if CALCULATE_BEST_PERFORMANCE:
            all_best_partitions = list(set([tuple(x) for x in all_best_partitions]))
            all_best_changepoints = [get_changepoints(partition) for partition in all_best_partitions]
            
            if SAVE_BEST_PERFORMANCE_TRAINING_DATA:
                changepoints_to_partition = defaultdict(list)
                for partition, changes in zip(all_best_partitions, all_best_changepoints):
                    changepoints_to_partition[tuple(changes)].append(partition)
                    
                for changes, partitions in changepoints_to_partition.items():
                    sample = {
                        "embeddings": su.get_partition_embeddings(text, partitions[0], model, tokenizer, device, model_name),
                        "f1_score": f1_score(true_changes, changes, labels=[0, 1], average="macro", zero_division=0.0),
                    }
                    sample["embeddings"] = [embedding.cpu().numpy().tolist() for embedding in sample["embeddings"]]
                    # score_train_savepath_fout.write(json.dumps(sample) + "\n")
                    # TODO
                    if i < num_train_samples:
                        score_train_savepath_fout.write(json.dumps(sample) + "\n")
                    else:
                        score_validation_savepath_fout.write(json.dumps(sample) + "\n")

            all_best_partitions = [
                (change, f1_score(true_changes, change, labels=[0, 1], average="macro", zero_division=0.0))
                for change in all_best_changepoints
            ]
            pred_best = max(all_best_partitions, key=lambda x: x[1])[0]
            pred_best_performance.append(pred_best)
        else:
            # 3. get the majority vote and the largest proposal score
            changes_majority = Counter(all_predicted_partitions).most_common(1)[0][0][0]
            changes_largest_score = [change for change, _, score in all_predicted_partitions if score == max(all_proposal_scores)][0]
            pred_majority.append(changes_majority)
            pred_largest_score.append(changes_largest_score)

        ground_truth.append(true_changes)
        
        if DEBUG:
            if CALCULATE_BEST_PERFORMANCE:
                debug_s = f"Best Performance: {pred_best}\tTrue Changes: {true_changes}"
            else:
                debug_s = f"Majority Vote: {changes_majority}\tLargest Proposal Score: {changes_largest_score}\tTrue Changes: {true_changes}"
            to_print.append(debug_s)

    if DEBUG:
        for p in to_print:
            print(p)

    if CALCULATE_BEST_PERFORMANCE:
        f1_best_performance = f1_score(list(chain.from_iterable(ground_truth)), list(chain.from_iterable(pred_best_performance)), labels=[0, 1], average="macro")
        print(f"F1 Score Best Performance: {f1_best_performance:.2f}")
    else:
        f1_majority = f1_score(list(chain.from_iterable(ground_truth)), list(chain.from_iterable(pred_majority)), labels=[0, 1], average="macro")
        f1_largest_score = f1_score(list(chain.from_iterable(ground_truth)), list(chain.from_iterable(pred_largest_score)), labels=[0, 1], average="macro")
        print(f"F1 Score Majority Vote: {f1_majority:.2f}\tF1 Score Largest Proposal Score: {f1_largest_score:.2f}")

    if CALCULATE_BEST_PERFORMANCE:
        for i in range(len(pred_best_performance)):
            with open(os.path.join(savedir_best_performance, f"solution-problem-{i+1}.json"), "w") as f:
                f.write(json.dumps({"changes": list(pred_best_performance[i])}))
    else:
        for i in range(len(pred_majority)):
            with open(os.path.join(savedir_majority, f"solution-problem-{i+1}.json"), "w") as f:
                f.write(json.dumps({"changes": list(pred_majority[i])}))
        for i in range(len(pred_largest_score)):
            with open(os.path.join(savedir_largest_score, f"solution-problem-{i+1}.json"), "w") as f:
                f.write(json.dumps({"changes": list(pred_largest_score[i])}))
        
    print(f"Total Time Taken: {time() - start:.2f}")
    return 0

if __name__ == "__main__":
    random.seed(43)
    sys.exit(main())