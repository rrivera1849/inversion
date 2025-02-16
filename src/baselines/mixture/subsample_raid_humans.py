
import json
import os
import pandas as pd

RAID_HUMAN_PATH = "/home/riverasoto1/repos/raid/data/train_human.jsonl"
DATA_DIR = "/data1/yubnub/changepoint/MUD_inverse/data/"

raid_human = pd.read_json(RAID_HUMAN_PATH, lines=True)

domain_name = [
    "abstracts", "reddit", "reviews"
]

# Sub-sample humans for test set (500), the rest for validation set
# Even out the number of validation data.

def to_records_and_save(data, fname):
    fout = open(fname, "w+")
    for d in data:
        record = {"unit": d}
        fout.write(json.dumps(record))
        fout.write("\n")

for domain in domain_name:
    df_domain = raid_human[raid_human["domain"] == domain]
    text = df_domain["generation"].tolist()
    test_generations = text[:500]
    valid_generations = text[500:]
    
    to_records_and_save(
        test_generations,
        os.path.join(DATA_DIR, domain, "test_human.jsonl"),
    )
    to_records_and_save(
        valid_generations,
        os.path.join(DATA_DIR, domain, "valid_human.jsonl"),
    )
    
    df = pd.read_json(os.path.join(DATA_DIR, domain, "valid.jsonl"), lines=True)    
    df = df.sample(frac=1.).reset_index(drop=True)
    df = df.iloc[:len(valid_generations)]
    df.to_json(os.path.join(DATA_DIR, domain, "valid.small.jsonl"), lines=True, orient="records")