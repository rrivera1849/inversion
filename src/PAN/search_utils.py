
from typing import List

import numpy as np
import torch
from sentence_transformers.util import cos_sim

import nn_utils as nnu

def get_proposal_score(
    text: List[str], 
    author_partition: List[int], 
    model,
    tokenizer,
    device,
    model_name="uar",
):
    """A given "proposal" is a partition of the text into author segments. 
       The score is the sum of the pairwise distances between the embeddings of the segments.
       Our assumption is that the higher the score, the better the proposal.
    """
    assert model_name in ["cisr", "uar"]

    all_embeddings = get_partition_embeddings(text, author_partition, model, tokenizer, device, model_name)
        
    all_embeddings = torch.cat(all_embeddings, dim=0)
    distances = 1 - cos_sim(all_embeddings, all_embeddings)
    
    adjacent_distances = 0.
    for i in range(distances.shape[0] - 1):
        adjacent_distances += distances[i, i+1].item()
    
    return adjacent_distances

def get_proposal_score_pairwise(
    text: List[str], 
    author_partition: List[int], 
    model,
    tokenizer,
    device,
    model_name="uar",
):
    """A given "proposal" is a partition of the text into author segments. 
       The score is the sum of the pairwise distances between the embeddings of the segments.
       Our assumption is that the higher the score, the better the proposal.
    """
    assert model_name in ["cisr", "uar"]

    all_embeddings = get_partition_embeddings(text, author_partition, model, tokenizer, device, model_name)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    distances = 1 - cos_sim(all_embeddings, all_embeddings)
    
    score = distances[np.triu_indices(distances.shape[0], k=1)].sum()
    return score   

def get_proposal_score_by_intra_and_inter(
    text: List[str], 
    author_partition: List[int], 
    model,
    tokenizer,
    device,
    model_name="uar",
):
    """A given "proposal" is a partition of the text into author segments. 
       The score is the sum of the pairwise distances between the embeddings of the segments.
       Our assumption is that the higher the score, the better the proposal.
    """
    assert model_name in ["cisr", "uar"]

    score = 0.
    author_id = max(author_partition)
    for idx in range(author_id+1):
        author_text = [text[i] for i in range(len(text)) if author_partition[i] == idx]
        if len(author_text) == 1:
            score += 0.
            # no intra similarities for a single author
        else:
            embeddings = [
                nnu.get_embedding(author_text[i], model, tokenizer, device, model_name=model_name)
                for i in range(len(author_text))
            ]
            embeddings = torch.cat(embeddings, dim=0)
            sims = cos_sim(embeddings, embeddings)
            score += sims[np.triu_indices(sims.shape[0], k=1)].sum()
            # otherwise, sum of all intra similarities
    
    score += get_proposal_score_pairwise(text, author_partition, model, tokenizer, device, model_name)
    return score

def get_partition_embeddings(text, author_partition, model, tokenizer, device, model_name):
    """Get the embeddings of the text partitioned by the author_partition.
    """
    author_id = max(author_partition)
    all_embeddings = []
    for idx in range(author_id+1):
        author_text = [text[i] for i in range(len(text)) if author_partition[i] == idx]
        embedding = nnu.get_embedding(author_text, model, tokenizer, device, model_name=model_name)
        all_embeddings.append(embedding)
    return all_embeddings