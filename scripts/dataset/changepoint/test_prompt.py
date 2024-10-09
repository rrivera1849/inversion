
import json
import os
from argparse import ArgumentParser

import pandas as pd
from transformers import set_seed
from tqdm import tqdm
from vllm import (
    LLM,
    SamplingParams,
)

from prompts import *
from utils import clean_generation

set_seed(43)

parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default=None, required=True,
                    help="Path to the dataset to generate prompts for.")
parser.add_argument("--key", type=str, default="unit")
parser.add_argument("--prompt_type", type=str, default="rephrase",
                    choices=["rephrase", "respond_reddit"])
parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3",
                    choices=["mistralai/Mistral-7B-Instruct-v0.3", 
                             "meta-llama/Meta-Llama-3-8B-Instruct",
                             "microsoft/Phi-3-mini-4k-instruct"],
                    help="Huggingface model name to use for generation.")
parser.add_argument("--example_batch_size", type=int, default=10,
                    help="Number of examples to generate prompts for at any given time.")
parser.add_argument("--gen_batch_size", type=int, default=128,
                    help="Number of examples to generate completions for at any given time.")
parser.add_argument("--max_gen_len", type=int, default=128+32,
                    help="Maximum number of tokens to generate.")
parser.add_argument("--temperature", type=float, default=0.7,
                    help="Generation temperature.")
parser.add_argument("--top_p", type=float, default=0.9,
                    help="Generation top-p.")
parser.add_argument("--index_start", type=int, default=None)
parser.add_argument("--index_end", type=int, default=None)
parser.add_argument("--debug", action="store_true", 
                    help="Enable debugging mode.")
args = parser.parse_args()

DATASET = pd.read_json(args.dataset_path, lines=True)
DATASET.rename(columns={"syms": "unit"}, inplace=True)
to_expode = [col for col in DATASET.columns if col != "author_id"]
DATASET = DATASET.explode(to_expode).reset_index(drop=True)

PROMPT = get_prompt(args.prompt_type)
model = LLM(args.model_name)

def get_generations(
    prompt_text: list[str],
) -> list[str]:
    sampling_params = SamplingParams(
        n=1,
        max_tokens=args.max_gen_len,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    outputs = model.generate(prompt_text, sampling_params)
    outputs = [[o.text for o in out.outputs][0] for out in outputs]
    return outputs

basename, dirname = os.path.basename(args.dataset_path), os.path.dirname(args.dataset_path)
dataset_name = "{}_{}_{}_temperature={}_top_p={}.jsonl".format(
    basename, args.model_name.split("/")[1], args.prompt_type, args.temperature, args.top_p
)
dirname = os.path.join(dirname, "generations")
os.makedirs(dirname, exist_ok=True)
savename = os.path.join(dirname, dataset_name)
if os.path.isfile(savename):
    try:
        fout = open(savename, "a+")
        last_index = json.loads(open(savename, "r").readlines()[-1])["dataset_index"] + 1
        print(f"Appending to {savename}, last_index={last_index}")
    except:
        fout = open(savename, "w+")
        last_index = 0
        print(f"Creating {dataset_name}, last_index={last_index}")

else:
    fout = open(savename, "w+")
    last_index = 0
    print(f"Creating {dataset_name}, last_index={last_index}")

if args.index_start is not None:
    last_index = args.index_start
    print(f"Overwriting last_index to {args.index_start}")

end_index = len(DATASET) if args.index_end is None else args.index_end
print(f"Generating prompts for {end_index-last_index} examples.")

for i in tqdm(range(last_index, end_index, args.example_batch_size)):
    # Create prompts
    prompts = []
    dataset_indices = []
    examples = []
    for j in range(i, min(i+args.example_batch_size, end_index)):
        example = DATASET.iloc[j][args.key]
        prompts.append(PROMPT.format(example))
        dataset_indices.append(j)
        examples.append(example)
        
    if "Phi-3" in args.model_name:
        # https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        prompts = [
            # "<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\n{}\n<|end|>\n<|assistant|>".format(prompt)
            "<|user|>\n{}\n<|end|>\n<|assistant|>".format(prompt)
            for prompt in prompts
        ]

    # Generate completions
    generations = []
    for j in range(0, len(prompts), args.gen_batch_size):
        batch_generations = get_generations(prompts[j:j+args.gen_batch_size])
        generations.extend(batch_generations)
        if args.debug:
            for prompt, generation in zip(prompts[j:j+args.gen_batch_size], batch_generations):
                print(prompt)
                print("> {}".format(generation))
                print("="*50)
                print()

    for j in range(len(generations)):
        index = dataset_indices[j]
        record = DATASET.iloc[index].to_dict()
        record["model_name"] = args.model_name.split("/")[1]
        record[args.prompt_type] = clean_generation(generations[j], is_reddit=True)
        record[args.prompt_type + "_prompt"] = prompts[j]
        record["dataset_index"] = index
        fout.write(json.dumps(record) + "\n")

    if args.debug:
        import pdb; pdb.set_trace()
        break
