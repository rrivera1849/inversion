"""Very similar to hf_prompt.py, but works on the RAID dataset. 
"""

import json
import os
import random
from argparse import ArgumentParser

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm

from prompts import *
from utils import clean_generation

random.seed(43)

parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default=None, required=True,
                    help="Path to the dataset to generate prompts for.")
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
parser.add_argument("--fall_back", default=False, action="store_true",
                    help="Fall back to smaller batch size if the number of tokens is too large.")
parser.add_argument("--fall_back_num_tokens", type=int, default=450,
                    help="Number of tokens to fall back to.")
parser.add_argument("--debug", action="store_true", 
                    help="Enable debugging mode.")
args = parser.parse_args()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TIKTOKEN_CACHE_DIR"] = "/tmp/riverasoto1"

DATASET = pd.read_json(args.dataset_path, lines=True)
DATASET.rename(columns={"syms": "unit"}, inplace=True)
to_expode = [col for col in DATASET.columns if col != "author_id"]
DATASET = DATASET.explode(to_expode).reset_index(drop=True)

PROMPT = get_prompt("rephrase")

attn_implementation = "flash_attention_2" if "Phi-3" in args.model_name else None
model = AutoModelForCausalLM.from_pretrained(
    args.model_name, 
    device_map="auto", 
    torch_dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation=attn_implementation,
)

if "Mistral" in args.model_name.split('/')[1]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision="pr/51")
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
elif "Mixtral" in args.model_name.split('/')[1]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
elif "Meta-Llama" in args.model_name.split('/')[1]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

generation_config = GenerationConfig(
    temperature=args.temperature,
    do_sample=True,
    top_p=args.top_p,
    max_new_tokens=args.max_gen_len,
)

def count_num_tokens(prompt):
    return len(tokenizer(prompt)["input_ids"])

def get_generations(prompts, sub_batch_size=1):
    """Returns a list of completions for the given prompts.
    """
    with torch.inference_mode():
        generations = []
        num_tokens = [count_num_tokens(prompt) for prompt in prompts]
        
        if args.fall_back and max(num_tokens) > args.fall_back_num_tokens:
            print("Falling to smaller batch size due to large number of tokens...")
            prompts = [prompts[i:i+sub_batch_size] for i in range(0, len(prompts), sub_batch_size)]
        else:
            prompts = [prompts]

        for j, prompt_batch in enumerate(prompts):
            inputs = tokenizer(
                prompt_batch,
                max_length=1024,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to("cuda")
            output = model.generate(**inputs, generation_config=generation_config)

            prompt_lengths = [len(prompt_batch[i]) for i in range(len(prompt_batch))]
            if "Phi-3" in args.model_name:
                prompt_lengths = [length - 32 for length in prompt_lengths]
            generations.extend([
                tokenizer.decode(output[i], skip_special_tokens=True)[prompt_lengths[i]:] 
                for i in range(len(prompt_batch))
            ])

    assert len(generations) == sum(len(prompt_batch) for prompt_batch in prompts)
    return generations

basename, dirname = os.path.basename(args.dataset_path), os.path.dirname(args.dataset_path)
dataset_name = "{}_{}_rephrases_temperature={}_top_p={}.jsonl".format(
    basename, args.model_name.split("/")[1], args.temperature, args.top_p
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

for i in tqdm(range(last_index, len(DATASET), args.example_batch_size)):
    # Create prompts
    prompts = []
    dataset_indices = []
    for j in range(i, min(i+args.example_batch_size, len(DATASET))):
        example = DATASET.iloc[j]["unit"]
        prompts.append(PROMPT.format(example))
        dataset_indices.append(j)

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
        record["rephrase"] = clean_generation(generations[j], is_reddit=True)
        record["rephrase_prompt"] = prompts[j]
        record["dataset_index"] = index
        fout.write(json.dumps(record) + "\n")

    if args.debug:
        break