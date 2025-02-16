
import json
import os
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm
DATA_FNAME = "/home/riverasoto1/repos/raid/data/raid.jsonl"
SAVEDIR = "/data1/foobar/changepoint/MUD_inverse/data"

df = pd.read_json(DATA_FNAME, lines=True)
pool = Pool(40)

def subsample_data(domain, debug):
    domain_df = df[df.domain == domain]
    paraphrases_df = domain_df[domain_df.attack == "paraphrase"]

    records = []
    pbar = tqdm(total=len(paraphrases_df))
    i = 0
    for _, row in paraphrases_df.iterrows():
        record = {
            "author_id": i,
            "rephrase": row["generation"],
            "unit": domain_df[domain_df["id"] == row["adv_source_id"]]["generation"].iloc[0],
        }
        records.append(record)
        pbar.update(1)
        i += 1
        if i > 10 and debug:
            break
    
    return records

def save(fname, records):
    with open(fname, "w+") as fout:
        for record in records:
            fout.write(json.dumps(record) + "\n")

# domains = df.domain.unique().tolist()
domains = ["reddit", "reviews"]
for domain in tqdm(domains):
    print(domain)
    records = subsample_data(domain, False)
    
    savenames = ["train.jsonl", "valid.jsonl", "test.jsonl"]
    train_num_samples = int(0.8 * len(records))
    valid_num_samples = int(0.1 * len(records))
    train = records[:train_num_samples]
    valid = records[train_num_samples:train_num_samples+valid_num_samples]
    test = records[train_num_samples+valid_num_samples:]
    
    domain_dir = os.path.join(SAVEDIR, domain)
    os.makedirs(domain_dir, exist_ok=True)
    save(os.path.join(domain_dir, "train.jsonl"), train)
    save(os.path.join(domain_dir, "valid.jsonl"), valid)
    save(os.path.join(domain_dir, "test.jsonl"), test)