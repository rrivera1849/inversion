
import random
import sys

import pandas as pd
import numpy as np
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm

seed = 43
N = 100
TEST_DATA_PATH = "/data1/yubnub/changepoint/s2orc_changepoint/unit_128/test_clean_and_joined"

random.seed(seed)

def main():
    samples = []
    dataset = load_from_disk(TEST_DATA_PATH)
    # RRS - 10k should be enough for us to get the data we need:
    K = 10_000
    print("Reading data...")
    for i in tqdm(range(K)):
        example = dataset[i]
        units = example["units"]
        
        generation_keys = [key for key in example.keys() if "prompt=rephrase_generations" in key]
        changepoint_indices_keys = [key for key in example.keys() if "prompt=rephrase_changepoint_indices" in key]

        indices = set.intersection(*[set(example[ckey]) for ckey in changepoint_indices_keys])
        for index in indices:
            unit = units[index]
            
            generations = []
            model_names = []
            for gkey, ckey in zip(generation_keys, changepoint_indices_keys):
                changepoint_indices = example[ckey]
                generation_index = changepoint_indices.index(index)
                model_name = gkey.split("_prompt")[0]
                generation = example[gkey][generation_index]
                generations.append(generation)
                model_names.append(model_name)

            samples.append((example["id"], unit, generations, model_names))

    units = [sample[1] for sample in samples]
    print("Number of Units: ", len(units))
    model = SentenceTransformer("all-mpnet-base-v2")
    features = model.encode(units, show_progress_bar=True, batch_size=512, convert_to_numpy=True, normalize_embeddings=False)

    print("Clustering...")
    sampled_data = []
    seen = set()
    kmeans = KMeans(n_clusters=N, random_state=seed).fit(features)
    for label in np.unique(kmeans.labels_):
        # randomly sample one unit from each cluster
        idx = np.where(kmeans.labels_ == label)[0]

        for sample_idx in idx:
            if samples[sample_idx][0] in seen:
                continue

            sample = samples[sample_idx]
            sampled_data.append(sample)
            seen.add(samples[sample_idx][0])
            break
        
    print("Saving data...")
    sampled_data = pd.DataFrame(data={
        "id": [data[0] for data in sampled_data],
        "unit": [data[1] for data in sampled_data],
        "generation": [data[2] for data in sampled_data],
        "model_name": [data[3] for data in sampled_data],
    })
    sampled_data = sampled_data.explode(["generation", "model_name"])
    sampled_data.to_json("../test_data/author.jsonl", orient="records", lines=True)

    return 0

if __name__ == "__main__":
    sys.exit(main())