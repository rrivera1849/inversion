
import json
import os

import numpy as np
from sklearn.metrics import euclidean_distances

DATA_PATH = "/data1/yubnub/changepoint/MUD_inverse/data/data.jsonl.filtered.cleaned_kmeans_100"
EMBEDDINGS_PATH = "/data1/yubnub/changepoint/MUD_inverse/embeddings/data.jsonl.filtered.cleaned_LUAR-MUD"

cluster_centers = np.load(os.path.join(DATA_PATH, "cluster_centers.npy"))

style_embeddings = np.stack([np.load(os.path.join(EMBEDDINGS_PATH, fname)) for fname in sorted(os.listdir(EMBEDDINGS_PATH))])
style_embeddings /= np.linalg.norm(style_embeddings, axis=1)[:, np.newaxis]

distances = euclidean_distances(style_embeddings, cluster_centers)
closest_cluster = np.argmin(distances, axis=1)

author_id_to_cluster_center = {}
for i, fname in enumerate(sorted(os.listdir(EMBEDDINGS_PATH))):
    author_id = os.path.splitext(fname)[0]
    author_id_to_cluster_center[author_id] = int(closest_cluster[i])

with open(os.path.join(DATA_PATH, "author_id_to_cluster_center.json"), "w") as fout:
    json.dump(author_id_to_cluster_center, fout)