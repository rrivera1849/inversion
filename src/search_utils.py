
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

    author_id = max(author_partition)
    all_embeddings = []
    for idx in range(author_id+1):
        author_text = [text[i] for i in range(len(text)) if author_partition[i] == idx]
        embedding = nnu.get_embedding(author_text, model, tokenizer, device, model_name=model_name)
        all_embeddings.append(embedding)
        
    all_embeddings = torch.cat(all_embeddings, dim=0)
    distances = 1 - cos_sim(all_embeddings, all_embeddings)
    pairwise_distances = distances[np.triu_indices(distances.shape[0], 1)].sum()

    return pairwise_distances.item()
