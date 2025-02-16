
import os
import sys

import pandas as pd

DATA_DIR = "/data1/foobar/changepoint/MUD_inverse/data/"

domain_name = sys.argv[1]

def merge(
    human_paraphrase_fname,
    machine_paraphrase_fname,
):
    human_paraphrase = pd.read_json(human_paraphrase_fname, lines=True)
    human_paraphrase["is_human_paraphrase"] = True
    human_paraphrase["is_machine"] = False
    N = len(human_paraphrase)
    human_paraphrase = pd.concat([human_paraphrase, human_paraphrase]).reset_index(drop=True)

    indices = human_paraphrase.iloc[len(human_paraphrase)//2:].index
    human_paraphrase.loc[indices, "is_machine"] = False
    human_paraphrase.loc[indices, "is_human_paraphrase"] = False
    human_paraphrase.loc[indices, "rephrase"] = human_paraphrase.loc[indices, "unit"]
    
    machine_paraphrase = pd.read_json(machine_paraphrase_fname, lines=True)
    machine_paraphrase["is_human_paraphrase"] = False
    machine_paraphrase["is_machine"] = True
    
    machine_paraphrase = machine_paraphrase.iloc[:N]
    
    cols = ["unit", "rephrase", "is_human_paraphrase", "is_machine"]
    human_paraphrase = human_paraphrase[cols]
    machine_paraphrase = machine_paraphrase[cols]
    
    merged = pd.concat([human_paraphrase, machine_paraphrase])
    return merged

def main():
    df = merge(
        os.path.join(DATA_DIR, domain_name, "valid_human.jsonl.paraphrased"),
        os.path.join(DATA_DIR, domain_name, "valid.jsonl"),
    )
    df.to_json(os.path.join(DATA_DIR, domain_name, "valid_final.jsonl"), lines=True, orient="records")
    
    # repeat for test
    df = merge(
        os.path.join(DATA_DIR, domain_name, "test_human.jsonl.paraphrased"),
        os.path.join(DATA_DIR, domain_name, "test.small.jsonl"),
    )
    df.to_json(os.path.join(DATA_DIR, domain_name, "test_final.jsonl"), lines=True, orient="records")
    
main()