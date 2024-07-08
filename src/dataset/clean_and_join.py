
import os
import random; random.seed(43)
import sys
from argparse import ArgumentParser
from glob import glob

import pandas as pd
from datasets import load_from_disk
from termcolor import colored
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--dirname", type=str,
                    default="/data1/yubnub/changepoint/s2orc_changepoint/unit_128",
                    help="Directory where the dataset is stored.")
parser.add_argument("--split", type=str, default="test",
                    help="Dataset split to clean and join the generations for.")
parser.add_argument("--debug", default=False, action="store_true",
                    help="Enable debugging mode.")
args = parser.parse_args()

EXPECTED_NUM_FILES = 9 # 3 generations * 3 LLM = 9 files

def clean(row):
    # versa. \n\nNote: The
    row["generations"] = [generation.split("\n") for generation in row["generations"]]
    return row


def clean_generation(generation):
    # [Note: I rephrased
    if "[Note: I rephrased" in generation:
        generation = generation.split("[Note: I rephrased")[0]

    # # Rephrased passage: The
    if "# Rephrased passage:" in generation:
        generation = generation.split("# Rephrased passage: The")[0]

    # # Passage Preceding
    if "# Passage Preceding" in generation:
        generation = generation.split("# Passage Preceding")[0]
    
    # Rephrase the following passage
    if "Rephrase the following passage" in generation:
        generation = generation.split("Rephrase the following passage")[0]
    
    # Rephrased passage:
    if "Rephrased passage:" in generation:
        generation = generation.split("Rephrased passage:")[0]
    
    # Sure, here is the rephrased passage:
    if "Sure, here is the rephrased passage:\n\n" in generation:
        generation = generation.split("Sure, here is the rephrased passage:\n\n")[1]

    # Sure, here is the continuation:\n\n
    if "Sure, here is the continuation:\n\n" in generation:
        generation = generation.split("Sure, here is the continuation:\n\n")[0]

    # # versa. \n\nNote: The
    generation = generation.split("\n")
    index = [i for i, gen in enumerate(generation) if "note:" in gen.lower() or "rephrased passage:" in gen.lower() or "please note that the rephrased passage" in gen.lower()]
    if len(index) > 0:
        index = index[0]
        generation = "\n".join(generation[:index])
    else:
        generation = "\n".join(generation)

    # remove things that don't end with punctuation, or closed parenthesis
    if generation[-1] not in [".", "?", "!", ")", "]"]:
        generation = generation.rsplit(".", 1)[0] + "."

    return generation

def get_generation_files(dataset):
    generations_dirname = os.path.join(args.dirname, "generations")
    generation_filenames = list(glob(generations_dirname + f"/*{args.split}*"))
    assert len(generation_filenames) == EXPECTED_NUM_FILES

    # Just looking at each LLM for now
    generation_filenames = [filename for filename in generation_filenames if "continuation" in filename]

    for filename in tqdm(generation_filenames):
        nrows = 10 if args.debug else None
        df = pd.read_json(filename, lines=True, nrows=nrows)

        # print some generations to get a feel for what needs to be cleaned
        generations = df["generations"].explode().tolist()
        random.shuffle(generations)
        to_iterate = df.explode(["generations", "changepoint_indices"]).sample(frac=1., random_state=43).reset_index(drop=True)
        for i, row in to_iterate.iterrows():
            original = dataset[row["dataset_index"]]["units"][row["changepoint_indices"]]
            generation = row["generations"]
            try:
                cgeneration = clean_generation(generation)
            except:
                import pdb; pdb.set_trace()
                
            if generation == cgeneration:
                continue
            os.system("clear")
            print("> " + colored(filename, "red"))
            print("{}/{}".format(i, len(to_iterate)))
            print("ORIGINAL > " + colored(original, "blue"))
            print()
            print("GENERATION > " + colored(generation, "yellow"))
            print()
            print("CLEAN GENERATION > " + colored(cgeneration, "green"))

            if len(cgeneration) > len(original):
                if cgeneration == original[:len(cgeneration)]:
                    print(colored(">>> ORIGINAL AND GENERATION ARE EQUAL", "red"))
            elif len(cgeneration) < len(original):
                if original == cgeneration[:len(original)]:
                    print(colored(">>> ORIGINAL AND GENERATION ARE EQUAL", "red"))
            elif cgeneration == original:
                print(colored(">>> ORIGINAL AND GENERATION ARE EQUAL", "red"))

            input("Continue?")
        # now randomly shuffle
        
def main():
    split_dirname = os.path.join(args.dirname, args.split)
    dataset = load_from_disk(split_dirname)
    get_generation_files(dataset)
    
    # d[generations_prompt=<foo>] = generations
    # d[generations_prompt=<foo>_metadata] = metadata
    # metadata = {"top_p": 0.9, "temperature": 0.7, "max_length": 128 + 32}


    print("YOU NEED TO += 1 CHANGEPOINT INDICES FOR GENERATIONS WITH PROMPT=CONTINUATION")
    # intersection of tokens
    # distance in number of tokens from original
    
    # probably need to SBERT distance for cleaning?

    # things that don't belong in the dataset are those that don't intersect
    # across all generations

    # make sure to remove those that are identical to the original
    # calculate #token overlap between generations and original
    
    return 0

if __name__ == "__main__":
    sys.exit(main())