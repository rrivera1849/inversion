
import json
import os
import sys
import random
from argparse import ArgumentParser

import pandas as pd
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm

sys.path.append("../changepoint")
from prompts import *
from clean_and_join import clean_generation   

random.seed(43)
parser = ArgumentParser()
parser.add_argument("--cisr_file", type=str, 
                    default="/home/riverasoto1/repos/Style-Embeddings/Data/train_data/train-210000__subreddits-100-2018_tasks-300000__topic-variable-random.tsv", 
                    help="CISR training file for which to get prompted hard negatives from.")
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

PROMPT = "Paraphrase: {}\nResult: "

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct", 
    device_map="auto", 
    torch_dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

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
        max_length = 1024

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
                tokenizer.decode(output[i], skip_special_tokens=True)
                for i in range(len(prompt_batch))
            ])

    generations = [
        gen[gen.index("Result: ")+len("Result: "):]
        for gen in generations
    ]
    assert len(generations) == sum(len(prompt_batch) for prompt_batch in prompts)
    return generations

nrows = 100 if args.debug else None
df = pd.read_csv(args.cisr_file, nrows=nrows, delimiter="\t")
df = df[(df["Anchor (A)"].isna() | df["Utterance 1 (U1)"] | df["Utterance 2 (U2)"].isna())]

prompts = []
for text in df["Anchor (A)"]:
    prompts.append(PROMPT.format(text))
# https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
prompts = [
    # "<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\n{}\n<|end|>\n<|assistant|>".format(prompt)
    "<|user|>\n{}\n<|end|>\n<|assistant|>".format(prompt)
    for prompt in prompts
]

for i in tqdm(range(0, len(prompts), args.gen_batch_size)):
    generations = get_generations(prompts[i:i+args.gen_batch_size])
    for j, generation in enumerate(generations):
        df.loc[i+j, "Hard Negative"] = clean_generation(generation)

savename = "-".join(args.cisr_file.split("-")[:-1])
savename += "-paraphrase.tsv"
df.to_csv(savename, sep="\t", index=False)