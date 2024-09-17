"""Inverse Inference on the Test Split.

TODO: PPL Evaluation
"""

import json
import os
import sys
from argparse import ArgumentParser

import spacy
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    GenerationConfig, 
    set_seed,
)
from tqdm import tqdm

from utils import (
    build_inverse_prompt, 
    get_mixture_weights, 
    load_mixture_predictor,
)

set_seed(43)

parser = ArgumentParser()
parser.add_argument("--dataset_name", type=str, 
                    default="data.jsonl.filtered.cleaned_kmeans_100")
parser.add_argument("--filename", type=str, default="test.jsonl")
parser.add_argument("--prompt_type", type=str, default="none")
parser.add_argument("--mixture_predictor_path", type=str, default=None,
                    help="Path to the mixture predictor model.")
parser.add_argument("--num", type=int, default=320*4)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

DEBUG = args.debug

BATCH_SIZE = 20
DATA_PATH = "/data1/yubnub/changepoint/MUD_inverse/data"
assert os.path.isdir(os.path.join(DATA_PATH, args.dataset_name))

PROMPTING_DATA_PATH = os.path.join(DATA_PATH, args.dataset_name, "inverse_output")
os.makedirs(PROMPTING_DATA_PATH, exist_ok=True)

INVERSE_SAVEPATH = "/data1/yubnub/changepoint/models/inverse"
INVERSE_MODELS = {
    "none": "r=32_alpha=64_dropout=0.1_perc=1.0_perc-gold-labels=0.5",
}
assert args.prompt_type in INVERSE_MODELS
NLP = spacy.load("en_core_web_sm")

def load_inverse_model(
    prompt_type: str = "none",
):
    """Loads our inversion model, either with or without Mixture Weights in the prompt.
    """
    inverse_model = INVERSE_MODELS[prompt_type]
    checkpoint_path = os.path.join(
        INVERSE_SAVEPATH, 
        args.dataset_name, 
        args.prompt_type, 
        inverse_model, 
        f"checkpoint-{args.num}"
    )
    
    inverse_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,        
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    inverse_model = torch.compile(inverse_model)
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
    filename = os.path.join(DATA_PATH, args.dataset_name, args.filename)
    data = [json.loads(line) for line in open(filename)]
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
    if args.prompt_type != "none":
        mixture_predictor = load_mixture_predictor(args.mixture_predictor_path)
        mixture_token_fn = mixture_predictor.tokenizer.tokenize

    data = load_prompting_data()
    inverse_model, inverse_tokenizer = load_inverse_model(args.prompt_type)

    generation_args = {
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    unit_length = 128 + 32
    generation_config = GenerationConfig(
        do_sample=True,
        max_new_tokens=unit_length,
        **generation_args,
    )
    
    output_fname = os.path.join(
        PROMPTING_DATA_PATH, 
        f"{args.prompt_type}_{args.num}"
    )
    for name, value in generation_args.items():
        output_fname += f"_{name}={value}"
    output_fname += ".jsonl"
    output_fout = open(output_fname, "w+")
    print(f"Writing to {output_fname}")
    
    for batch_idx in tqdm(range(0, len(data), BATCH_SIZE)):
        batch = data[batch_idx:batch_idx+BATCH_SIZE]
        text = [b["rephrase"] for b in batch]

        if args.prompt_type != "none":
            mixture_probs = get_mixture_weights(
                mixture_predictor,
                text,
                key=None,
                batch_size=BATCH_SIZE,
                progress_bar=False,
            )
            mixture_tokens = [mixture_token_fn(t) for t in text]
            
            prompt_text = [
                build_inverse_prompt(sample, "", tokens, probs, prompt_type=args.prompt_type, no_stop_tokens=True)
                for sample, tokens, probs in zip(text, mixture_tokens, mixture_probs)
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
            gen = gen[gen.index("Output: ")+len("Output: "):]
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

        if DEBUG and batch_idx >= BATCH_SIZE * 10:
            break

    return 0

if __name__ == "__main__":
    sys.exit(main())