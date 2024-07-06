
import json
import os
import random
from argparse import ArgumentParser

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm

from prompts import *

random.seed(43)

parser = ArgumentParser()
parser.add_argument("--dirname", type=str,
                    default="/data1/yubnub/changepoint/s2orc_changepoint/unit_128",
                    help="Directory where the dataset is stored.")
parser.add_argument("--model_name", type=str, default="google/gemma-7b-it",
                    choices=["google/gemma-7b-it", "mistralai/Mistral-7B-Instruct-v0.3", 
                             "mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Meta-Llama-3-8B-Instruct"],
                    help="Huggingface model name to use for generation.")
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
parser.add_argument("--fall_back", default=False, action="store_true",
                    help="Fall back to smaller batch size if the number of tokens is too large.")
parser.add_argument("--fall_back_num_tokens", type=int, default=450,
                    help="Number of tokens to fall back to.")
parser.add_argument("--debug", action="store_true", 
                    help="Enable debugging model.")
args = parser.parse_args()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TIKTOKEN_CACHE_DIR"] = "/tmp/riverasoto1"

DIRNAME = args.dirname
DATASET = load_from_disk(f"{DIRNAME}/{args.split}")
PROMPT = get_prompt(args.prompt)

model = AutoModelForCausalLM.from_pretrained(
    args.model_name, 
    device_map="auto", 
    torch_dtype=torch.float16,
    trust_remote_code=True,
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
    max_new_tokens=args.max_gen_len,
    temperature=args.temperature,
    do_sample=True,
    top_p=args.top_p,
)

def count_num_tokens(prompt):
    return len(tokenizer(prompt)["input_ids"])

def get_generations(prompts, sub_batch_size=1):
    """Returns a list of completions for the given prompts.
    """
    with torch.inference_mode():
        max_length = 2048 if args.prompt == "rephrase_with_context" else 1024

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
                max_length=max_length, 
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to("cuda")
            output = model.generate(**inputs, generation_config=generation_config)
            generations.extend([
                tokenizer.decode(output[i], skip_special_tokens=True)[len(prompt_batch[i]):] 
                for i in range(len(prompt_batch))
            ])

    assert len(generations) == sum(len(prompt_batch) for prompt_batch in prompts)
    return generations

dataset_name = "{}_{}_prompt={}_temperature={}_top_p={}.jsonl".format(
    args.model_name.split("/")[1], args.split, args.prompt, args.temperature, args.top_p
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