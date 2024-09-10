"""
We want:
- Perplexity
"""

import json
import os
import sys
from argparse import ArgumentParser

import spacy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
from termcolor import colored
from tqdm import tqdm

from utils import build_inverse_prompt, get_mixture_weights, load_mixture_predictor

set_seed(43)

parser = ArgumentParser()
parser.add_argument("--prompt_type", type=str, default="none",
                    choices=["none", "probs", "logprobs", "tokens"])
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

DEBUG = args.debug
NLP = spacy.load("en_core_web_sm")
BATCH_SIZE = 8
USE_MIXTURE_PROBS = args.prompt_type != "none"
MIXTURE_PATH = "./outputs/s2orc_roberta-large_200000_perc=0.5/checkpoints/checkpoint_6/"
PROMPTING_DATA_PATH = "./test_data/generations"
INVERSE_SAVEPATH = "/data1/yubnub/changepoint/models/inverse"
INVERSE_MODELS = {
    "none": "Mistral-7B-v0.3-QLoRA_lr=2e-05_max-steps=3900_num-samples=200000_r=32_alpha=64_dropout=0.1_perc=1.0_prompt=none_perc-gold-labels=0.5_debug=False",
}

def load_inverse_model(
    prompt_type: str = "no_mixture",
):
    """Loads our inversion model, either with or without Mixture Weights in the prompt.
    """
    inverse_model = INVERSE_MODELS[prompt_type]
    print(colored("Loading: ", "green"), inverse_model)
    checkpoint_path = os.path.join(INVERSE_SAVEPATH, inverse_model, "checkpoint-600")

    inverse_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,        
        torch_dtype=torch.float16,
        device_map="auto",
    )
    inverse_model.eval()

    inverse_tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.3", 
        padding_side="left", 
        add_eos_token=False
    )
    inverse_tokenizer.pad_token = inverse_tokenizer.eos_token
    inverse_model.generation_config.pad_token_id = inverse_tokenizer.pad_token_id

    return inverse_model, inverse_tokenizer

def load_prompting_data(
):
    """Loads the Prompting Data.
    """
    fname = "./test_data/s2orc.jsonl"
    data = [json.loads(line) for line in open(fname)]
    return data

def get_best_sentence_substring(
    inverse_text: str,
    original_text: str
):
    """This function extracts the substring that matches the length 
       of the original text the best.
    """
    sentences = NLP(inverse_text).sents
    substrings = [inverse_text[:sent.end_char] for sent in sentences]
    length_difference = [abs(len(original_text) - len(sub)) for sub in substrings]
    min_idx = length_difference.index(min(length_difference))
    return substrings[min_idx].strip()

def main():
    mixture_predictor = load_mixture_predictor(MIXTURE_PATH)
    mixture_token_fn = mixture_predictor.tokenizer.tokenize
    
    inverse_model, inverse_tokenizer = load_inverse_model(args.prompt_type)

    generation_args = {
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    generation_config = GenerationConfig(
        do_sample=True,
        max_new_tokens=128+32,
        **generation_args,
    )
    
    data = load_prompting_data()

    output_fname = os.path.join(PROMPTING_DATA_PATH, f"{args.prompt_type}")
    for name, value in generation_args.items():
        output_fname += f"_{name}={value}"
    output_fname += ".jsonl"
    output_fout = open(output_fname, "w+")
    print(f"Writing to {output_fname}")
    
    for batch_idx in tqdm(range(0, len(data), BATCH_SIZE)):
        batch = data[batch_idx:batch_idx+BATCH_SIZE]
        text = [b["generation"] for b in batch]

        if USE_MIXTURE_PROBS:
            mixture_probs = get_mixture_weights(
                mixture_predictor,
                text,
                batch_size=BATCH_SIZE,
                key=FileNotFoundError,
                progress_bar=False
            )
            mixture_tokens = [mixture_token_fn(t) for t in text]
            
            prompt_text = [
                build_inverse_prompt(t, "", t, w, prompt_type=args.prompt_type, no_stop_tokens=True)
                for t, w in zip(text, mixture_tokens, mixture_probs)
            ]
        else:
            prompt_text = [build_inverse_prompt(t, "", no_stop_tokens=True) for t in text]

        tokenized_text = inverse_tokenizer(
            prompt_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=3072,
        ).to("cuda")
        generations = inverse_model.generate(
            **tokenized_text,
            generation_config=generation_config,
        )
        generations = inverse_tokenizer.batch_decode(generations, skip_special_tokens=True)
        
        outputs = []
        for j, gen in enumerate(generations):
            gen = gen[gen.index("Output:")+len("Output:"):]
            if "\n#####\n" in gen:
                gen = gen[:gen.index("\n#####\n")]
            else:
                gen = get_best_sentence_substring(gen, text[j])
            outputs.append(gen)

        for j in range(len(outputs)):
            elem = batch[j]
            elem["inverse"] = outputs[j]
            elem["inverse_prompt"] = prompt_text[j]
            output_fout.write(json.dumps(elem) + "\n")
            output_fout.flush()
            
        if DEBUG:
            break

    return 0

if __name__ == "__main__":
    sys.exit(main())