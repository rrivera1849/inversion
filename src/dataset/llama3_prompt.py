
import json
import os
import random
from argparse import ArgumentParser

from datasets import load_from_disk
from llama import Llama
from tqdm import tqdm

from prompts import *

random.seed(43)

parser = ArgumentParser()
parser.add_argument("--split", type=str, default="validation",
                    help="Dataset split to generate prompts for.")
parser.add_argument("--prompt", type=str, default="rephrase",
                    choices=["rephrase", "rephrase_with_context", "continuation"],
                    help="Prompt type to use for generation")
parser.add_argument("--example_batch_size", type=int, default=10,
                    help="Number of examples to generate prompts for at any given tim.")
parser.add_argument("--gen_batch_size", type=int, default=128,
                    help="Number of examples to generate completions for at any given time.")
parser.add_argument("--max_gen_len", type=int, default=128+32,
                    help="Maximum number of tokens to generate.")
parser.add_argument("--temperature", type=float, default=0.7,
                    help="Generation temperature.")
parser.add_argument("--top_p", type=float, default=0.9,
                    help="Generation top-p.")
parser.add_argument("--debug", action="store_true", 
                    help="Enable debugging model.")
args = parser.parse_args()

os.environ["TIKTOKEN_CACHE_DIR"] = "/tmp/riverasoto1"
LLAMA3_PATH = "/home/riverasoto1/repos/llama3/Meta-Llama-3-8B-Instruct"

DIRNAME = "/data1/yubnub/changepoint/s2orc_changepoint/unit_128"
DATASET = load_from_disk(f"{DIRNAME}/{args.split}")
PROMPT = get_prompt(args.prompt)

if args.prompt == "rephrase_with_context":
    max_seq_len = 2048
else:
    max_seq_len = 1024

generator = Llama.build(
    ckpt_dir=LLAMA3_PATH,
    tokenizer_path=os.path.join(LLAMA3_PATH, "tokenizer.model"),
    max_seq_len=max_seq_len,
    max_batch_size=args.gen_batch_size,
)

dataset_name = "{}_prompt={}_temperature={}_top_p={}.jsonl".format(
    args.split, args.prompt, args.temperature, args.top_p
)
savename = f"{DIRNAME}/generations/{dataset_name}"
if os.path.isfile(savename):
    fout = open(savename, "a+")
    last_index = json.loads(open(savename, "r").readlines()[-1])["dataset_index"] + 1
    print(f"Appending to {savename}, last_index={last_index}")
else:
    fout = open(savename, "w+")
    last_index = 0
    print(f"Creating {dataset_name}, last_index={last_index}")

for i in tqdm(range(last_index, len(DATASET), args.example_batch_size)):
    dataset_indices = []
    prompts = []
    lengths = [0]
    subsample_changepoint_indices = []
    for j in range(i, min(i+args.example_batch_size, len(DATASET))):
        example = DATASET[j]
        changepoint_indices = example["changepoint_indices"]
        subsample_indices = random.sample(range(len(changepoint_indices)), len(changepoint_indices) // 2)
        if len(subsample_indices) == 0:
            continue
        changepoint_indices = [changepoint_indices[sidx] for sidx in subsample_indices]
        dataset_indices.append(j)
        subsample_changepoint_indices.append(changepoint_indices)
        lengths.append(len(changepoint_indices))
        
        if args.prompt == "rephrase_with_context":
            prompts.extend(
                [PROMPT.format(example["units"][k-1], example["units"][k+1], example["units"][k]) 
                 for k in changepoint_indices]
            )
        else:
            prompts.extend(
                [PROMPT.format(example["units"][k]) for k in changepoint_indices]
            )

    # Generate Completions
    all_results = []
    for j in range(0, len(prompts), args.gen_batch_size):
        results = generator.text_completion(
            prompts[j:j+args.gen_batch_size],
            max_gen_len=args.max_gen_len,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        all_results.extend(results)
        
        if args.debug:
            for prompt, result in zip(prompts[j:j+args.max_gen_len], results):
                print(prompt)
                print("> {}".format(result["generation"]))
                print("="*50)
                print()
                
    generations = [result["generation"] for result in all_results]
    for j, length in enumerate(lengths[1:]):
        start = sum(lengths[:j+1])
        end = start + length
        item = {
            "generations": generations[start:end],
            "changepoint_indices": subsample_changepoint_indices[j],
            "dataset_index": dataset_indices[j],
        }
        fout.write(json.dumps(item) + "\n")

    if args.debug:
        break