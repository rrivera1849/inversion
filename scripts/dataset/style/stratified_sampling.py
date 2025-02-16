
import json
import os
import sys
from argparse import ArgumentParser
from multiprocessing import Pool

import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.metrics import pairwise_distances

parser = ArgumentParser()
parser.add_argument("--embeddings_dirname", type=str, default="data.jsonl.filtered.cleaned_LUAR-MUD")
parser.add_argument("--clustering_algorithm", type=str, default="kmeans",
                    choices=["kmeans", "affinity_propagation"])
parser.add_argument("--n_clusters", type=int, default=100)
parser.add_argument("--seed", type=int, default=43)
parser.add_argument("--debug", default=False, action="store_true")
args = parser.parse_args()

np.random.seed(43)

BASE_DIR = "/data1/foobar/changepoint/MUD_inverse"
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw")

def main():
    data_fname = os.path.join(RAW_DATA_DIR, f"{args.embeddings_dirname.split('_')[0]}")
    embeddings_dir = os.path.join(EMBEDDINGS_DIR, args.embeddings_dirname)

    if not os.path.isdir(embeddings_dir):
        print(f"Embeddings directory {embeddings_dir} does not exist.")
        return 1
    if not os.path.isfile(data_fname):
        print(f"Dataset file {data_fname} does not exist.")
        return 1

    if args.debug:
        N = 1_000
    else:
        N = len(os.listdir(embeddings_dir))

    embedding_fnames = [os.path.join(embeddings_dir, fname) for fname in os.listdir(embeddings_dir)[:N]]
    with Pool(40) as pool:
        embeddings = pool.map(np.load, embedding_fnames)
    embeddings = np.stack(embeddings)
    embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]

    import pdb; pdb.set_trace()
    
    if args.clustering_algorithm == "affinity_propagation":
        similarities = 1. - pairwise_distances(embeddings, metric="cosine")
        cluster_alg = AffinityPropagation(
            random_state=args.seed,
            affinity="precomputed"
        )
        cluster_alg.fit(similarities)
    elif args.clustering_algorithm == "kmeans":
        cluster_alg = KMeans(
            n_clusters=args.n_clusters,
            random_state=args.seed
        )
        cluster_alg.fit(embeddings)

    label_to_author_ids = {}
    for filename, label in zip(embedding_fnames, cluster_alg.labels_):
        author_id = os.path.basename(filename)
        author_id = os.path.splitext(author_id)[0]
        if label not in label_to_author_ids:
            label_to_author_ids[label] = []
        label_to_author_ids[label].append(author_id)

    splitname_to_author_ids = {
        "train": [],
        "valid": [],
        "test": []
    }
    for label, author_ids in label_to_author_ids.items():
        np.random.shuffle(author_ids)
        n_train = int(0.8 * len(author_ids))
        n_valid = int(0.1 * len(author_ids))
        splitname_to_author_ids["train"].extend(author_ids[:n_train])
        splitname_to_author_ids["valid"].extend(author_ids[n_train:n_train+n_valid])
        splitname_to_author_ids["test"].extend(author_ids[n_train+n_valid:])

    assert len(set(splitname_to_author_ids["train"])) == len(splitname_to_author_ids["train"])
    assert len(set(splitname_to_author_ids["valid"])) == len(splitname_to_author_ids["valid"])
    assert len(set(splitname_to_author_ids["test"])) == len(splitname_to_author_ids["test"])
    assert len(set.intersection(
        set(splitname_to_author_ids["train"]),
        set(splitname_to_author_ids["valid"]),
        set(splitname_to_author_ids["test"]),
    )) == 0

    split_data_dir = os.path.join(DATA_DIR, os.path.basename(data_fname)) + f"_{args.clustering_algorithm}"
    if args.clustering_algorithm == "kmeans":
        split_data_dir += f"_{args.n_clusters}"
    os.makedirs(split_data_dir, exist_ok=True)

    fout_train = open(os.path.join(split_data_dir, "train.jsonl"), "w+")
    fout_valid = open(os.path.join(split_data_dir, "valid.jsonl"), "w+")
    fout_test = open(os.path.join(split_data_dir, "test.jsonl"), "w+")

    def reformat(data: dict) -> list[dict]:
        data_df = pd.DataFrame(data)
        to_explode = [col for col in data_df.columns if col != "author_id"]
        data_df = data_df.explode(to_explode)
        return data_df.to_dict(orient="records")

    with open(data_fname, "r") as fin:
        for line in fin:
            data = json.loads(line)
            author_id = data["author_id"]
            data = reformat(data)
            
            if author_id in splitname_to_author_ids["train"]:
                for datum in data:
                    fout_train.write(json.dumps(datum) + "\n")
            elif author_id in splitname_to_author_ids["valid"]:
                for datum in data:
                    fout_valid.write(json.dumps(datum) + "\n")
            elif author_id in splitname_to_author_ids["test"]:
                for datum in data:
                    fout_test.write(json.dumps(datum) + "\n")

    fout_train.close()
    fout_valid.close()
    fout_test.close()

    with open(os.path.join(split_data_dir, "splitname_to_author_ids.json"), "w+") as fout:
        json.dump(splitname_to_author_ids, fout)
    np.save(os.path.join(split_data_dir, "cluster_centers.npy"), cluster_alg.cluster_centers_)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())