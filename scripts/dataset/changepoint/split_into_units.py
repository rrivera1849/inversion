
import os
import random
import sys
from argparse import ArgumentParser
from functools import partial

import spacy
from datasets import load_from_disk
from transformers import AutoTokenizer

random.seed(43)

parser = ArgumentParser()
parser.add_argument("--num_max_tokens", type=int, default=128)
args = parser.parse_args()

NLP = spacy.load("en_core_web_sm", disable=["tagger", "attribute_ruler", "lemmatizer", "ner"])
TOKENIZER = AutoTokenizer.from_pretrained("roberta-base")

def split_into_units(example, num_max_tokens: int = 128):
    """We make our best effort at splitting text into "units" that are at most 
       `num_max_tokens` long contiguous complete sentences. 

       Sometimes a single sentence is too long to fit into a unit, in which case 
       we include it as its own unit regardless.
    """
    # first paragraph break is always the title, lets just remove it:
    title_index = example["text"].index("\n\n")
    # remove paragraph and section breaks
    text = example["text"][title_index + 2:].replace("\n\n", " ").replace("\n", " ")

    # sentencize, and count the number of tokens in each sentence
    doc = NLP(text)
    sentences = list(doc.sents)
    num_tokens = [count_num_tokens(sent.text) for sent in sentences]

    # split into units
    unit_indices = []
    indices = []
    token_count = 0
    for i, nt in enumerate(num_tokens):
        if token_count + nt > num_max_tokens:
            if len(indices) == 0:
                # if a single sentence is too long, include it as its own unit
                indices.append(i)
                unit_indices.append(indices)
                indices = []
                token_count = 0
            else:
                unit_indices.append(indices)
                indices = [i]
                token_count = nt
        else:
            indices.append(i)
            token_count += nt
    unit_indices.append(indices)
    assert len(set.intersection(*[set(indices) for indices in unit_indices])) == 0
    
    # get metadata and ensure that the units are valid
    indices_larger_than_max, proportion_larger_than_max = get_larger_than_metadata(
        unit_indices, num_tokens, num_max_tokens
    )

    # get the text of each unit
    units = get_unit_text(text, sentences, unit_indices)
    assert len(units) == len(unit_indices)

    example["units"] = units
    example["lens"] = [count_num_tokens(unit) for unit in units]
    example["indices_larger_than_max"] = indices_larger_than_max
    example["proportion_larger_than_max"] = proportion_larger_than_max
    
    # sample the changepoint indices:
    num_changepoints_to_sample = int(len(units) * 0.5)
    changepoint_indices = random.sample(range(1, len(units) - 1), num_changepoints_to_sample)
    example["changepoint_indices"] = sorted(changepoint_indices)

    return example

def count_num_tokens(text: str):
    """Counts the number of tokens in a given text.
    """
    return len(TOKENIZER.tokenize(text))

def get_larger_than_metadata(
    unit_indices: list[list[int]], 
    num_tokens: list[int], 
    num_max_tokens: int
):
    """Get metadata about the units that are larger than `num_max_tokens`.
    """
    indices_larger_than_max = []
    proportion_larger_than_max = 0
    for i, indices in enumerate(unit_indices):
        if len(indices) > 1:
            # sanity check expected token length
            assert sum(num_tokens[i] for i in indices) <= num_max_tokens
        else:
            if num_tokens[indices[0]] > num_max_tokens:
                indices_larger_than_max.append(i)
                proportion_larger_than_max += 1
    proportion_larger_than_max /= len(unit_indices)
    return indices_larger_than_max,proportion_larger_than_max

def get_unit_text(
    text: str, 
    sentences: list[spacy.tokens.span.Span], 
    unit_indices: list[list[int]]
):
    """Return the text for each unit.
    """
    units = []
    for indices in unit_indices:
        unit_text = text[sentences[indices[0]].start_char:sentences[indices[-1]].end_char]
        units.append(unit_text)
    return units

def main():
    print("BE CAREFUL MAN, THERE ARE SOME CRAZY VARS HERE FIX LATER")
    
    # TODO
    save_dir = f"/data1/yubnub/changepoint/s2orc_changepoint/author_unit_{args.num_max_tokens}"
    os.makedirs(save_dir, exist_ok=True)
    # for split in ["validation", "test", "train"]:
    for split in ["validation", "train"]:
        print(f"Processing {split}...")
        path = f"/data1/yubnub/changepoint/s2orc_changepoint/base_author/{split}"
        changepoint = load_from_disk(path)
        split_fn = partial(split_into_units, num_max_tokens=args.num_max_tokens)
        changepoint = changepoint.map(split_fn, num_proc=40)
        changepoint.save_to_disk(f"{save_dir}/{split}")

    return 0

if __name__ == "__main__":
    sys.exit(main())