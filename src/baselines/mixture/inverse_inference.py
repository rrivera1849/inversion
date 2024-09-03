
import json
import os
import sys
from argparse import ArgumentParser

import spacy
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from termcolor import colored
from tqdm import tqdm

from utils import build_inverse_prompt, get_mixture_weights, load_mixture_predictor, get_levenshtein_tags

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="mixture_simple")
parser.add_argument("--oracle", default=False, action="store_true")
parser.add_argument("--oracle_tok", type=str, default="roberta-large")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

DEBUG = args.debug
NLP = spacy.load("en_core_web_sm")
BATCH_SIZE = 8
USE_MIXTURE_WEIGHTS = args.model_name != "no_mixture"
PROMPTING_DATA_PATH = "./prompting_data/inverse_trained"
INVERSE_SAVEPATH = "/data1/yubnub/changepoint/models/inverse"
INVERSE_MODELS = {
    "no_mixture": "Mistral-7B-v0.3-QLoRA_lr=2e-05_max_steps=1000_num_samples=25900_r=32_alpha=64_dropout=0.1_debug=False",
    "mixture": "Mistral-7B-v0.3-QLoRA_lr=2e-05_max-steps=1000_num-samples=25900_r=32_alpha=64_dropout=0.1_use-mixture-weights=True_perc-gold-labels=0.5_debug=False",
    "mixture_simple": "Mistral-7B-v0.3-QLoRA_lr=2e-05_max-steps=1000_num-samples=25900_r=32_alpha=64_dropout=0.1_simple-prompt=True_perc-gold-labels=0.5_debug=False",
}

def get_oracle_tokens_and_probs(
    rephrase: str,
    original: str,
):
    tokenizer = AutoTokenizer.from_pretrained(args.oracle_tok)
    tags = get_levenshtein_tags(rephrase, original, tokenizer.tokenize)
    tokens = tokenizer.tokenize(rephrase)
    probs = [[1.0, 0.0] if tag == "KEEP" else [0.0, 1.0] for tag in tags]
    return tokens, probs

def load_inverse_model(
    model_name: str = "no_mixture",
):
    """Loads our inversion model, either with or without Mixture Weights in the prompt.
    """
    inverse_model = INVERSE_MODELS[model_name]
    print(colored("Loading: ", "green"), inverse_model)
        
    checkpoint_path = os.path.join(INVERSE_SAVEPATH, inverse_model, "checkpoint-1000")
    
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
    fname = "./prompting_data/rephrases_gpt-4o.jsonl"
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
    
    if not args.oracle:
        mixture_predictor = load_mixture_predictor()
        mixture_token_fn = mixture_predictor.tokenizer.tokenize
    else:
        mixture_token_fn = AutoTokenizer.from_pretrained(args.oracle_tok).tokenize
    
    inverse_model, inverse_tokenizer = load_inverse_model(
        args.model_name,
    )

    generation_args = {
        "temperature": 0.7,
        "top_p": 0.9,
    }
    generation_config = GenerationConfig(
        do_sample=True,
        max_new_tokens=128+32,
        **generation_args,
    )
    
    data = load_prompting_data()
    output_fname = os.path.join(PROMPTING_DATA_PATH, f"{args.model_name}")
    args_to_save = generation_args.copy()
    if args.oracle:
        args_to_save["oracle"] = True
        args_to_save["oracle_tok"] = args.oracle_tok
    for name, value in args_to_save.items():
        output_fname += f"_{name}={value}"
    output_fname += ".jsonl"
    output_fout = open(output_fname, "w+")
    print(f"Writing to {output_fname}")
    
    for batch_idx in tqdm(range(0, len(data), BATCH_SIZE)):
        batch = data[batch_idx:batch_idx+BATCH_SIZE]
        text = [b["unit"] for b in batch]

        if not args.oracle:
            mixture_out = get_mixture_weights(
                mixture_predictor,
                batch,
                batch_size=BATCH_SIZE,
                key="unit",
                return_sequence_probs=True,
                progress_bar=False
            )
            token_probs = mixture_out[1]
        else:
            _, token_probs = zip(*[get_oracle_tokens_and_probs(b["rephrase"], b["unit"]) for b in batch])
            
        if USE_MIXTURE_WEIGHTS:
            simple_prompt = True if "simple" in args.model_name else False
            prompt_text = [
                build_inverse_prompt(t, "", mixture_token_fn(t), w, simple_prompt=simple_prompt)
                for t, w in zip(text, token_probs)
            ]
        else:
            prompt_text = [build_inverse_prompt(t, "") for t in text]

        tokenized_text = inverse_tokenizer(
            prompt_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=2048,
        ).to("cuda")
        outputs = inverse_model.generate(
            **tokenized_text,
            generation_config=generation_config,
        )

        outputs = inverse_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs = [t[t.index("Output: ")+len("Output: "):] for t in outputs]
        outputs = [get_best_sentence_substring(o, t) for o, t in zip(outputs, text)]

        for j in range(len(outputs)):
            elem = batch[j]
            elem["inverse"] = outputs[j]
            elem["inverse_prompt"] = prompt_text[j]
            output_fout.write(json.dumps(elem) + "\n")
            
            print("Original:", colored(elem["unit"], "blue"))
            print()
            print("Inverse:", colored(outputs[j], "red"))
            print("=====================================")

        if DEBUG:
            break

    return 0

if __name__ == "__main__":
    sys.exit(main())