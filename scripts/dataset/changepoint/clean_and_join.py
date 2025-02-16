
import json
import os
import random
import sys
from argparse import ArgumentParser
from glob import glob
from typing import Dict, List, Union

import pandas as pd
from datasets import load_from_disk, Dataset
from termcolor import colored
from tqdm import tqdm

from prompts import PROMPT_NAMES
from utils import clean_generation

random.seed(43)

parser = ArgumentParser()
parser.add_argument("--dirname", type=str,
                    default="/data1/yubnub/changepoint/s2orc_changepoint/unit_128",
                    help="Directory where the dataset is stored.")
parser.add_argument("--split", type=str, default="test",
                    help="Dataset split to clean and join the generations for.")
parser.add_argument("--explore", default=False, action="store_true",
                    help="A debugging mode to explore the generations and the clean_generation function.")
parser.add_argument("--explore_skip", default=False, action="store_true",
                    help="Skips the generations that are equal to their clean versions.")
parser.add_argument("--explore_num", type=int, default=50,
                    help="Number of generations to explore.")
parser.add_argument("--explore_only", type=str, default=None,
                    help="Substring to search for in the filename to explore only those files.")
parser.add_argument("--debug", default=False, action="store_true",
                    help="Just process a small subset of the data.")
args = parser.parse_args()

NUM_STRINGS_REVERTED = 0
NUM_INTERSECTION_FAILED = 0

def clean(row):
    row["generations"] = clean_generation(row["generations"])
    indices_to_remove = [i for i, gen in enumerate(row["generations"]) if gen is None]
    row["generations"] = [gen for i, gen in enumerate(row["generations"]) if i not in indices_to_remove]
    row["changepoint_indices"] = [gen for i, gen in enumerate(row["changepoint_indices"]) if i not in indices_to_remove]
    return row

def explore_generations(dataset: Dataset) -> None:
    """Helper function to explore the generations and the clean_generation function.
    """
    generations_dirname = os.path.join(args.dirname, "generations")
    generation_filenames = list(glob(generations_dirname + f"/*{args.split}*"))
    if len(generation_filenames) == 0:
        generation_filenames = list(glob(generations_dirname + "/*"))
        assert len(generation_filenames) > 0, "No files found in the generations directory."
    generation_filenames = [fname for fname in generation_filenames if "gemma" not in fname]
    if args.explore_only:
        generation_filenames = [filename for filename in generation_filenames if args.explore_only in filename]

    for prompt in PROMPT_NAMES:
        prompt_filenames = [filename for filename in generation_filenames if f"prompt={prompt}_temperature" in filename]

        for filename in prompt_filenames:
            nrows = 10
            df = pd.read_json(filename, lines=True, nrows=nrows)

            # print some generations to get a feel for what needs to be cleaned
            if "changepoint_indices" not in df.columns:
                df["changepoint_indices"] = df["generations"].apply(lambda x: list(range(len(x))))
            to_iterate = df.explode(["generations", "changepoint_indices"]).sample(frac=1., random_state=43)
            to_iterate = to_iterate.reset_index(drop=True).iloc[:args.explore_num]
            for i, row in to_iterate.iterrows():
                original_index = row["changepoint_indices"]
                original_index -= 1 if prompt == "continuation" else 0
                original = dataset[row["dataset_index"]]["units"][original_index]
                generation = row["generations"]

                try:
                    cgeneration = clean_generation(generation)
                except:
                    import pdb; pdb.set_trace()
                    cgeneration = clean_generation(generation)
                    
                if args.explore_skip and generation == cgeneration:
                    continue
                    
                os.system("clear")
                
                print("> PROMPT: " + colored(prompt, "cyan"))
                print("> LLM: " + colored(filename, "cyan"))
                print("{}/{}".format(i+1, len(to_iterate)))
                print()
                print("ORIGINAL > " + colored(original, "blue"))
                print()
                print("GENERATION > " + colored(generation, "yellow"))
                print()
                print("CLEAN GENERATION > " + colored(cgeneration, "green"))

                if len(generation) > len(original):
                    if generation[:len(original)] == original:
                        print(colored(">>> ORIGINAL AND GENERATION ARE EQUAL", "red"))
                elif len(generation) < len(original):
                    if original[:len(generation)] == generation:
                        print(colored(">>> ORIGINAL AND GENERATION ARE EQUAL", "red"))
                elif generation == original:
                    print(colored(">>> ORIGINAL AND GENERATION ARE EQUAL", "red"))

                input("Continue?")

def clean_all_generations() -> List[pd.DataFrame]:
    """Cleans all the generations and returns them in a list of pd.DataFrame, 
       where each DataFrame corresponds to a different LLM and prompt.
    """
    generations_dirname = os.path.join(args.dirname, "generations")
    generation_filenames = list(glob(generations_dirname + f"/*{args.split}*"))
    if len(generation_filenames) == 0:
        generation_filenames = list(glob(generations_dirname + "/*"))
        assert len(generation_filenames) > 0, "No files found in the generations directory."
    generation_filenames = [fname for fname in generation_filenames if "gemma" not in fname]

    tqdm.pandas()
    all_dfs = []
    for i, filename in enumerate(generation_filenames):
        print(colored("Reading: {}".format(os.path.basename(filename)), "cyan"))
        LLM = os.path.basename(filename).split("_")[0]
        nrows = 1000 if args.debug else None
        df = pd.read_json(filename, lines=True, nrows=nrows)
        if "changepoint_indices" not in df.columns:
            df["changepoint_indices"] = df["generations"].apply(lambda x: list(range(len(x))))
        df = df.progress_apply(clean, axis=1)

        for prompt in PROMPT_NAMES:
            s = f"prompt={prompt}_temperature"
            if s in filename:
                df["prompt"] = prompt
        df["LLM"] = LLM
        all_dfs.append(df)
        
    # only keep the dataset indices that are common across all LLMs and prompts
    intersection = set.intersection(*[set(df["dataset_index"].explode().tolist()) for df in all_dfs])
    all_dfs = [df[df["dataset_index"].isin(intersection)] for df in all_dfs]
    return all_dfs

def get_columns_to_add(all_generations: List[pd.DataFrame]) \
    -> Dict[str, Union[List[str], List[dict], List[List[int]]]]:
    """Returns a dictionary where the keys are the columns to add to the dataset,
       and the values are the corresponding values for each row in the dataset.
    """
    global NUM_INTERSECTION_FAILED
    
    columns_to_add = {}
    generation_and_changepoint_keys = []
    for df in all_generations:
        for _, row in df.iterrows():
            LLM = row["LLM"]
            prompt = row["prompt"]
            generations = row["generations"]
            changepoint_indices = row["changepoint_indices"]
            metadata = {
                "temperature": 0.7,
                "top_p": 0.9,
            }
            
            generations_key = f"{LLM}_prompt={prompt}_generations"
            metadata_key = f"{LLM}_prompt={prompt}_metadata"
            changepoints_key = f"{LLM}_prompt={prompt}_changepoint_indices"
            generation_and_changepoint_keys.append((generations_key, changepoints_key))

            if generations_key not in columns_to_add:
                columns_to_add[generations_key] = []
            if metadata_key not in columns_to_add:
                columns_to_add[metadata_key] = []
            if changepoints_key not in columns_to_add:
                columns_to_add[changepoints_key] = []

            columns_to_add[generations_key].append(generations)
            columns_to_add[metadata_key].append(metadata)
            columns_to_add[changepoints_key].append(changepoint_indices)
            
    lengths = [len(v) for v in columns_to_add.values()]
    assert len(set(lengths)) == 1, "All columns should have the same length."
    
    # # this is a tricky bit of code that ensures that all the 
    # # changepoint indices are the same across all LLMs and prompts
    # changepoint_indices = []
    # changepoint_columns = [column for column in columns_to_add.keys() if "changepoint_indices" in column]
    # for i in range(lengths[0]):
    #     all_changepoint_indices = [columns_to_add[column][i] for column in changepoint_columns]
    #     # assert that all changepoint indices are the same
        
    #     intersection_of_indices = list(set.intersection(*[set(indices) for indices in all_changepoint_indices]))
    #     if len(intersection_of_indices) != len(all_changepoint_indices[0]):
    #         NUM_INTERSECTION_FAILED += 1
    #         changepoint_indices.append(intersection_of_indices)

    #         for gkey, cpkey in generation_and_changepoint_keys:
    #             generations = columns_to_add[gkey][i]
    #             changepoints = columns_to_add[cpkey][i]
    #             indices = [changepoints.index(index) for index in intersection_of_indices]
    #             columns_to_add[gkey][i] = [generations[index] for index in indices]
    #             columns_to_add[cpkey][i] = intersection_of_indices
    #     else:
    #         changepoint_indices.append(all_changepoint_indices[0])

    # for column in changepoint_columns:
    #     del columns_to_add[column]
    # columns_to_add["changepoint_indices"] = changepoint_indices

    return columns_to_add

def main():
    split_dirname = os.path.join(args.dirname, args.split)
    dataset = load_from_disk(split_dirname)
    
    if args.explore:
        explore_generations(dataset)
        return 0

    all_generations = clean_all_generations()
    columns_to_add = get_columns_to_add(all_generations)
    dataset = dataset.select(all_generations[0].dataset_index.tolist())
    if "changepoint_indices" in dataset.column_names:
        dataset = dataset.remove_columns("changepoint_indices")
    for column, value in columns_to_add.items():
        dataset = dataset.add_column(column, value)

    # intersection of tokens
    # distance in number of tokens from original
    # probably need to SBERT distance for cleaning?
    # make sure to remove those that are identical to the original
    # calculate #token overlap between generations and original

    save_dirname = split_dirname + "_clean_and_joined"
    save_dirname += "_debug" if args.debug else ""
    dataset.save_to_disk(save_dirname)
    metadata = {
        "NUM_STRINGS_REVERTED": NUM_STRINGS_REVERTED,
        "NUM_INTERSECTION_FAILED": NUM_INTERSECTION_FAILED
    }
    with open(os.path.join(save_dirname, "metadata.json"), "w") as f:
        f.write(json.dumps(metadata, indent=4))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
