
import os

import pandas as pd

DATA_PATH = "/data1/yubnub/changepoint/MUD_inverse"

def main():
    valid_df = pd.read_json(os.path.join(DATA_PATH, "data/data.jsonl.filtered.respond_reddit.cleaned/valid.jsonl"), lines=True)

    raw_df = pd.read_json(os.path.join(DATA_PATH, "raw", "data.jsonl.filtered.cleaned"), lines=True)
    raw_df = raw_df[raw_df["author_id"].isin(valid_df["author_id"])]
    raw_df = raw_df[["unit"]].explode("unit").reset_index(drop=True)
    raw_df = raw_df.sample(frac=1.).iloc[:10_000]
    raw_df.rename(columns={"unit":"rephrase"}, inplace=True)
    raw_df["is_machine"] = False

    valid_df = valid_df.loc[:, ["rephrase"]]
    valid_df = pd.DataFrame(valid_df["rephrase"].sample(frac=1.).iloc[:10_000])
    valid_df["is_machine"] = True
    
    new_valid_df = pd.concat([raw_df, valid_df])
    # this is really just calibration data for our Gaussian Mixture Model:
    new_valid_df.to_json(
        os.path.join(DATA_PATH, "data/data.jsonl.filtered.respond_reddit.cleaned/valid_with_human.jsonl"),
        lines=True,
        orient="records",
    )
    
    return 0

main()