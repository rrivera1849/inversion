"""Checks that there is no overlap between training and testing data.
"""

import os
import sys

import pandas as pd
from datasets import load_from_disk

def read_all_mixture_data(path: str):
    df_mixture = pd.read_json(os.path.join(path, "train.jsonl"), lines=True)
    df_mixture_valid = pd.read_json(os.path.join(path, "valid.jsonl"), lines=True)
    df_mixture_test = pd.read_json(os.path.join(path, "test.jsonl"), lines=True)
    df_mixture = pd.concat([df_mixture, df_mixture_valid, df_mixture_test])
    return set(df_mixture[df_mixture["label"] == 0].text.tolist())

def read_all_inverse_data(path: str):
    df_inverse = pd.read_json(os.path.join(path, "train.jsonl"), lines=True)
    df_inverse_valid = pd.read_json(os.path.join(path, "valid.jsonl"), lines=True)
    df_inverse_test = pd.read_json(os.path.join(path, "test.jsonl"), lines=True)
    df_inverse = pd.concat([df_inverse, df_inverse_valid, df_inverse_test])
    return set(df_inverse.original.tolist())

def main():
    MIXTURE_TRAIN_PATH = "./datasets/all_roberta-large_250000_stratified/"
    INVERSE_TRAIN_PATH = "./datasets/all_roberta-large_250000_stratified_inverse/"

    print("Checking for overlap between training and testing data.")
    mixture_data = read_all_mixture_data(MIXTURE_TRAIN_PATH)
    inverse_data = read_all_inverse_data(INVERSE_TRAIN_PATH)
    all_data = mixture_data.union(inverse_data)

    df_test = pd.read_json("./prompting_data/all_units.jsonl", lines=True)
    test_data = set(df_test.unit.tolist())
    overlap = test_data.intersection(all_data)

    print("len(overlap):", len(overlap))
    if overlap:
        print("Fixing overlap...")
        records = {
            "id": [],
            "unit": [],
            "domain": [],
        }
        
        raid = load_from_disk("/data1/foobar/changepoint/RAID_rephrase/train_human_unit_128_clean_and_joined")
        raid_df = raid.to_pandas()
        for domain in raid_df.domain.unique():
            print("domain:", domain)
            raid_df_domain = raid_df[raid_df.domain == domain]
            raid_df_domain = raid_df_domain.explode("units")
            raid_df_domain.drop_duplicates(subset=["units"], inplace=True)
            mask = raid_df_domain.units.apply(lambda x: x not in all_data)
            raid_df_domain = raid_df_domain[mask]
            raid_df_domain = raid_df_domain.sample(frac=1.).reset_index(drop=True)
            raid_df_domain = raid_df_domain.head(100)
            records["id"].extend(raid_df_domain.id.tolist())
            records["unit"].extend(raid_df_domain.units.tolist())
            records["domain"].extend(raid_df_domain.domain.tolist())

        s2orc = load_from_disk("/data1/foobar/changepoint/s2orc_changepoint/unit_128/test")
        index = 0
        to_sample = 100
        while to_sample > 0:
            for unit in s2orc[index]["units"]:
                if unit not in all_data and unit not in records["unit"]:
                    records["id"].append(s2orc[index]["id"])
                    records["unit"].append(unit)
                    records["domain"].append("s2orc")
                    to_sample -= 1
                    break
                            
            index += 1

        assert len(set(records["unit"])) == 900
        assert set(records["unit"]).isdisjoint(all_data)
        df_no_overlap = pd.DataFrame(records)
        df_no_overlap.to_json("./prompting_data/no_overlap.jsonl", orient="records", lines=True)

    return 0

if __name__ == "__main__":
    sys.exit(main())