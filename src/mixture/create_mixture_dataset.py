
import json
import os
import random
import sys
from argparse import ArgumentParser
from collections import Counter
from multiprocessing import Pool

from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm

from utils import get_levenshtein_tag

random.seed(43)

parser = ArgumentParser()
parser.add_argument("--max_num_samples", type=int, default=250_000,
                    help="Maximum number of samples to use for the dataset (Positive / Negative only). "
                         "In the case of the inverse, we derive the data from the positive samples, "
                         "so we have exactly the number of samples.")
parser.add_argument("--perc_s2orc", type=float, default=0.10,
                    help="Percentage of the S2ORC data to use for the dataset.")
parser.add_argument("--stratified", default=False, action="store_true",
                    help="Whether to sample the dataset stratified by domain.")
parser.add_argument("--tokenizer", type=str, default="roberta-large",
                    help="Tokenizer to use for the dataset.")
parser.add_argument("--domain", type=str, default="all",
                    choices=["all", "books", "news", "wiki", "reddit", "recipes", 
                             "poetry", "abstracts", "reviews", "s2orc"],
                    help="Domains to use when creating the dataset.")
parser.add_argument("--is_inverse_data", default=False, action="store_true",
                    help="Whether to create the dataset for the inverse rephrase task.")
parser.add_argument("--debug", default=False, action="store_true")
args = parser.parse_args()


def create_dataset_dir(num_samples: int) -> str:
    """Creates a directory for the dataset.
    """
    tokenizer = os.path.basename(args.tokenizer)
    dataset_dirname = f"./datasets/{args.domain}_{tokenizer}_{num_samples}"
    if args.stratified:
        dataset_dirname += "_stratified"
    if args.is_inverse_data:
        dataset_dirname += "_inverse"
    if args.debug:
        dataset_dirname += "_debug"
        
    os.makedirs(dataset_dirname, exist_ok=True)
    if len(os.listdir(dataset_dirname)) > 0:
        raise ValueError(f"Dataset directory {dataset_dirname} is not empty.")
    
    return dataset_dirname

def save_dataset(
    dataset_mixture: list,
    tokenizer: AutoTokenizer,
    savename: str,
):
    with open(savename, "w+") as fout:
        for sample in tqdm(dataset_mixture):
            if args.is_inverse_data:
                record = {
                    "generation": sample[0][0],
                    "original": sample[0][1],
                    "domain": sample[1],
                    "label": sample[-1][0],
                    "tagger_labels": sample[-1][1],
                }
                if tokenizer is not None:
                    record.update(tokenizer(record["generation"], max_length=512, truncation=True, padding="max_length"))
            else:
                record = {
                    "text": sample[0],
                    "domain": sample[1],
                    "label": sample[-1][0],
                    "tagger_labels": sample[-1][1],
                }
                if tokenizer is not None:
                    record.update(tokenizer(record["text"], max_length=512, truncation=True, padding="max_length"))

            fout.write(json.dumps(record)); fout.write('\n')

def within_length_limit(text: str, tokenizer: AutoTokenizer, max_length: int = 510):
    return len(tokenizer.tokenize(text)) <= max_length

def read_all_domains(tokenizer):
    """Reads data from all domains and returns a mixture of the data.

    TODO - Handle more than just "all" or "s2orc" domains.
    """
    dataset_mixture = []
    if args.domain == "all" or args.domain == "s2orc":
        dataset_mixture += read_dataset(tokenizer, from_s2orc=True, is_inverse_data=args.is_inverse_data)
    if args.domain != "s2orc":
        dataset_mixture += read_dataset(tokenizer, from_s2orc=False, is_inverse_data=args.is_inverse_data)
    assert len(dataset_mixture) > 0
    if not args.is_inverse_data:
        random.shuffle(dataset_mixture)
    return dataset_mixture

def read_dataset(
    tokenizer: AutoTokenizer,
    from_s2orc: bool = False,
    is_inverse_data: bool = False,
) -> list:
    """Reads the dataset from the S2ORC or RAID combined dataset.
    """
    if from_s2orc:
        dirname = "/data1/foobar/changepoint/s2orc_changepoint/unit_128/train_clean_and_joined"
    else:
        dirname = "/data1/foobar/changepoint/RAID_rephrase/train_human_unit_128_clean_and_joined"
    
    dataset = load_from_disk(dirname)
    if args.debug:
        dataset = dataset.select(range(1_000))

    dataset_positive_samples = []
    dataset_negative_samples = []

    # Right now, we're processing the data in chunks of 10,000 samples.
    chunksize = 10_000
    with Pool(40) as pool:
        for i in tqdm(range(0, len(dataset), chunksize)):
            samples = [dataset[j] for j in range(i, min(i + chunksize, len(dataset)))]
            outputs = pool.starmap(process_sample, [(sample, tokenizer, from_s2orc, is_inverse_data) for sample in samples])

            outputs = [output for output in outputs if len(output[0]) > 0 or len(output[1]) > 0]
            for output in outputs:
                dataset_negative_samples.extend(output[0])
                dataset_positive_samples.extend(output[1])

    if not is_inverse_data:
        N = min(len(dataset_negative_samples), len(dataset_positive_samples))
        dataset_negative_samples = random.sample(dataset_negative_samples, k=N)
        dataset_positive_samples = random.sample(dataset_positive_samples, k=N)
        dataset_mixture = dataset_negative_samples + dataset_positive_samples
    else:
        dataset_mixture = dataset_positive_samples

    return dataset_mixture

# is_inverse_data = False -> (text, domain, (label, tagger_labels))
# is_inverse_data = True -> ((text, generation), domain, (label, tagger_labels))
def process_sample(
    sample: dict, 
    tokenizer: AutoTokenizer, 
    from_s2orc: bool, 
    is_inverse_data: bool
):
    """Processes a sample from either the S2ORC or RAID combined dataset.
       Returns a mixture of positive / negative if we're not creating the inverse dataset.
       Otherwise, returns the positive samples only.
    """
    domain = "s2orc" if "domain" not in sample.keys() else sample["domain"]
    
    if not from_s2orc and args.domain != "all" and domain != args.domain:
        return [], []

    dataset_negative_samples = []
    dataset_positive_samples = []
    
    original_text = sample["units"]
        
    # Sample the Negative (human) samples:
    if not is_inverse_data:    
        if from_s2orc:
            K = int(args.perc_s2orc * len(original_text))
            indices_to_sample = random.sample(range(len(original_text)), k=K)
        else:
            indices_to_sample = range(len(original_text))

        for index in indices_to_sample:
            text = original_text[index]
            if not within_length_limit(text, tokenizer):
                continue

            dataset_negative_samples.append((
                    text, domain,
                    (0, [0] * len(tokenizer.tokenize(text)))
                ))
        
    # Sample all the Positive (machine / human) rephrases
    generation_keys = [key for key in sample.keys() if "prompt=rephrase_generations" in key]
    changepoint_indices_keys = [key for key in sample.keys() if "prompt=rephrase_changepoint_indices" in key]

    for gkey, ckey in zip(generation_keys, changepoint_indices_keys):
        generations = sample[gkey]
        changepoint_indices = sample[ckey]
            
        if from_s2orc:
            K = int(args.perc_s2orc * len(generations))
            indices_to_sample = random.sample(range(len(generations)), k=K)
        else:
            indices_to_sample = range(len(generations))

        for index in indices_to_sample:
            generation = generations[index]
            original_index = changepoint_indices[index]
            original = original_text[original_index]

            if not within_length_limit(generation, tokenizer) or \
                not within_length_limit(original, tokenizer):
                continue
            
            tags = get_levenshtein_tags(generation, original, tokenizer.tokenize)
            tag_labels = [int(tag != "KEEP") for tag in tags]
    
            if is_inverse_data:
                dataset_positive_samples.append((
                        (generation, original), domain,
                        (1, tag_labels)
                    ))
            else:
                dataset_positive_samples.append((
                        generation, domain,
                        (1, tag_labels)
                    ))

    return dataset_negative_samples, dataset_positive_samples

def stratify_dataset(dataset_mixture):
    """Stratifies the dataset by domain.
    """
    dataset_mixture_domain_counts = Counter(sample[1] for sample in dataset_mixture)
    min_domain_count = min(dataset_mixture_domain_counts.values())
    domain_counts = {domain: 0 for domain in dataset_mixture_domain_counts.keys()}
    dataset_stratified = []
    for sample in dataset_mixture:
        domain = sample[1]
        if domain_counts[domain] < min_domain_count:
            dataset_stratified.append(sample)
            domain_counts[domain] += 1
                
        if all([v == min_domain_count for v in domain_counts.values()]):
            break
            
    dataset_mixture = dataset_stratified
    return dataset_mixture

def main():
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Reading data...")
    dataset_mixture = read_all_domains(tokenizer)
    
    if args.stratified:
        print("Stratifying dataset...")
        dataset_mixture = stratify_dataset(dataset_mixture)

    if args.max_num_samples != -1:
        num_samples = args.max_num_samples if args.is_inverse_data else 2 * args.max_num_samples
        dataset_mixture = dataset_mixture[:num_samples]

    print("Saving dataset...")
    num_samples = len(dataset_mixture)
    dataset_dirname = create_dataset_dir(num_samples)

    train_size = int(0.70 * len(dataset_mixture))
    val_size = int(0.15 * len(dataset_mixture))
    save_dataset(
        dataset_mixture[:train_size],
        tokenizer,
        f"{dataset_dirname}/train.jsonl",
    )
    save_dataset(
        dataset_mixture[train_size:train_size + val_size],
        tokenizer,
        f"{dataset_dirname}/valid.jsonl",
    )
    save_dataset(
        dataset_mixture[train_size + val_size:],
        tokenizer,
        f"{dataset_dirname}/test.jsonl",
    )

    return 0

if __name__ == "__main__":
    sys.exit(main())