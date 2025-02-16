# Filters data based on:
#  - min_docs_per_author
#  - max_docs_per_author
#  - min_words_in_doc
#  - removes duplicates
# For each author, select the pair of documents which have the least sbert similarity
#  - select only if the similarity is less than a margin.

import os
import argparse
import json
from typing import Dict, List
from sentence_transformers import SentenceTransformer, util
import torch
import logging
from datasets import load_from_disk
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("roberta-large")

def write_jsonl(data: List[Dict], output_file: str) -> None:
    with open(output_file, 'w') as f:
        for d in data:
            print(json.dumps(d, ensure_ascii=False), file=f)

def count_num_tokens(text: str) -> int:
    return len(TOKENIZER(text)['input_ids'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_file', type=str,
                        default="/data1/foobar/changepoint/s2orc_changepoint/unit_128/train_clean_and_joined")
    parser.add_argument('--output_data_file', type=str,
                        default="./s2orc_hard_positives.jsonl")
    parser.add_argument(
        '--model_name',
        default='all-mpnet-base-v2', help='sbert model used to calculate semantic similarity'
    )
    parser.add_argument(
        '--margin',
        type=float,
        default=0.2,
        help='doc pair by an author with more than this margin of cosine similarity will be discarded'
    )
    args = parser.parse_args()
    assert not os.path.exists(args.output_data_file)

    # logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%m/%d/%Y %H:%M:%S',
        filename=f"{args.output_data_file}.log"
    )
    logging.info(f"args = {args}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(args.model_name).to(device)

    all_data = {}
    dataset = load_from_disk(args.input_data_file)
    for i in range(len(dataset)):
        example = dataset[i]
        units = [unit for unit in example["units"] if count_num_tokens(unit) <= 128]
        all_data[i] = [{"author_id": i, "unit": unit} for unit in units]
    total_num_docs = sum([len(docs) for _, docs in all_data.items()])
    logging.info(f"Total number of docs initially: {total_num_docs}")

    # remove duplicate docs
    filtered_all_data = {}
    for author_id, docs in all_data.items():
        filtered_all_data[author_id] = []
        texts = [d['unit'] for d in docs]
        for idx in range(len(texts) - 1):
            text = texts[idx]
            if text not in texts[idx + 1:]:
                filtered_all_data[author_id].append(docs[idx])
        filtered_all_data[author_id].append(docs[-1])
    all_data = filtered_all_data

    # Of all the docs written by the same author, select the pair of docs which has the least sbert similarity.
    # ignore the author if the least similarity is more than the margin.
    hard_positives = []
    for author_id, docs in all_data.items():
        embeddings = model.encode([d['unit'] for d in docs])
        similarity_scores = util.dot_score(embeddings, embeddings)
        index_for_minimum = torch.argmin(similarity_scores).item()
        anchor_idx = index_for_minimum // len(embeddings)
        hard_positive_idx = index_for_minimum % len(embeddings)
        if similarity_scores[anchor_idx][hard_positive_idx] > args.margin:
            continue
        hard_positives.append(docs[anchor_idx])
        hard_positives.append(docs[hard_positive_idx])
    current_num_docs = len(hard_positives)
    docs_filtered = total_num_docs - current_num_docs
    logging.info(
        f"After selecting a single hard_positive pair for each author: "
        f"current_num_docs = {current_num_docs}, docs_filtered = {docs_filtered}"
    )

    write_jsonl(data=hard_positives, output_file=args.output_data_file)


if __name__ == '__main__':
    main()
