
import os
import sys
from glob import glob

import pandas as pd

from utils import clean_generation
from termcolor import colored
from tqdm.auto import tqdm
tqdm.pandas()

DATA_PATH = "/data1/yubnub/data/iur_dataset/author_100.politics/"

print(colored("RRS - This is only reading mistral generations right now!", "red"))

def clean_and_join(
    split_fname: str, 
    generation_split_fname: str,
    mixed: bool = False
):
    df_orig = pd.read_json(split_fname, lines=True)
    df_gen = pd.read_json(generation_split_fname, lines=True)
    df_gen = df_gen.sort_values(by="dataset_index", ascending=True).reset_index(drop=True)
    df_gen["clean_generations"] = df_gen["generations"].progress_apply(clean_generation)
    def choose(row):
        if row["clean_generations"][0] is None:
            return row
        row["generations"] = row["clean_generations"]
        return row
    df_gen = df_gen.progress_apply(choose, axis=1)

    if mixed:
        to_mix = df_orig.author_id.value_counts().unique()[0] // 2
        df_gen["author_id"] = df_orig["author_id"]
        df_gen["generations"] = df_gen["generations"].apply(lambda x: x[0])
        df_orig = df_orig.groupby("author_id").agg(list).reset_index(drop=False)
        df_gen = df_gen.groupby("author_id").agg(list).reset_index(drop=False)

        for index, row in df_orig.iterrows():
            gens_to_mix = df_gen[df_gen["author_id"] == row["author_id"]]["generations"].iloc[0][:to_mix]
            orig_to_mix = row["syms"][to_mix:]
            row["syms"] = gens_to_mix + orig_to_mix
            df_orig.loc[index] = row
        
        df_orig = df_orig.explode("syms").reset_index(drop=True)
        save_name = split_fname + ".mistral.mixed"
    else:
        df_orig["syms"] = df_gen["generations"]
        df_orig["syms"] = df_orig["syms"].apply(lambda x: x[0])
        save_name = split_fname + ".mistral"

    print(colored(f"Saving to {save_name}", "green"))
    df_orig.to_json(save_name, lines=True, orient="records")

for split_name in ["train", "valid", "test"]:
    split_fname = os.path.join(DATA_PATH, split_name + ".jsonl")
    generation_split_name = glob(os.path.join(DATA_PATH, f"generations/{split_name}*"))[0]
    clean_and_join(split_fname, generation_split_name)
    clean_and_join(split_fname, generation_split_name, mixed=True)