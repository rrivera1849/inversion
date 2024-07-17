
import json
import os
import sys
import random
from argparse import ArgumentParser

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm

sys.path.append("../changepoint")
from prompts import *

random.seed(43)
parser = ArgumentParser()
parser.add_argument("--text_file", type=str, default=None, required=True,
                    help="File of text samples for which to get rephrased prompts"
                         " to use as hard negatives.")
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

PROMPT = get_prompt("rephrase")

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct", 
    device_map="auto", 
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = tokenizer.pad_token_id

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

            prompt_lengths = [len(prompt_batch[i]) for i in range(len(prompt_batch))]
            if "Phi-3" in args.model_name:
                prompt_lengths = [length - 32 for length in prompt_lengths]
            generations.extend([
                tokenizer.decode(output[i], skip_special_tokens=True)[prompt_lengths[i]:] 
                for i in range(len(prompt_batch))
            ])

    assert len(generations) == sum(len(prompt_batch) for prompt_batch in prompts)
    return generations

