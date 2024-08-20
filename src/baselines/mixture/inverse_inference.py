"""Quick inference script for the inverse rephrase models, right now it is hard-coded to the Author 100 task.
"""

import json
import os
import sys
from argparse import ArgumentParser

import spacy
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm

from utils import build_inverse_prompt, get_mixture_weights, load_mixture_predictor

parser = ArgumentParser()
parser.add_argument("--use_mixture_weights", action="store_true")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

DEBUG = args.debug
NLP = spacy.load("en_core_web_sm")
BATCH_SIZE = 8
USE_MIXTURE_WEIGHTS = args.use_mixture_weights
AUTHOR_DATA_PATH = "/data1/yubnub/data/iur_dataset/author_100.politics"
INVERSE_SAVEPATH = "/data1/yubnub/changepoint/models/inverse"
INVERSE_MODELS = {
    "no_mixture": "Mistral-7B-v0.3-QLoRA_lr=2e-05_max_steps=1000_num_samples=25900_r=32_alpha=64_dropout=0.1_debug=False",
    "mixture": "Mistral-7B-v0.3-QLoRA_lr=2e-05_max-steps=1000_num-samples=25900_r=32_alpha=64_dropout=0.1_use-mixture-weights=True_perc-gold-labels=0.5_debug=False",
}

def load_inverse_model(
    use_mixture_weights: bool = True
):
    """Loads our inversion model, either with or without Mixture Weights in the prompt.
    """
    if use_mixture_weights:
        inverse_model = INVERSE_MODELS["mixture"]
    else:
        inverse_model = INVERSE_MODELS["no_mixture"]
        
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

def load_author_data(
    split_name: str = "train"
):
    """Loads a split from the Author 100 Politics dataset.
    """
    assert split_name in ["train", "valid", "test"]
    split_name += ".jsonl.mistral"
    with open(os.path.join(AUTHOR_DATA_PATH, split_name)) as fin:
        data = [json.loads(line) for line in fin]
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
    return substrings[min_idx]

def main():
    
    mixture_predictor = load_mixture_predictor()
    mixture_token_fn = mixture_predictor.tokenizer.tokenize
    
    inverse_model, inverse_tokenizer = load_inverse_model(
        use_mixture_weights=USE_MIXTURE_WEIGHTS,
    )
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=128+32,
    )
    
    for split_name in ["train", "valid", "test"]:
        data = load_author_data(split_name)
        output_fname = os.path.join(AUTHOR_DATA_PATH, f"{split_name}.jsonl.mistral.inverse")
        if USE_MIXTURE_WEIGHTS:
            output_fname += "-mixture"
        output_fout = open(output_fname, "w+")
        print(f"Writing to {output_fname}")
        
        mixture_seq_preds_orig = []
        mixture_seq_preds_inverse = []

        for batch_idx in tqdm(range(0, len(data), BATCH_SIZE)):
            batch = data[batch_idx:batch_idx+BATCH_SIZE]
            text = [b["syms"] for b in batch]

            mixture_out = get_mixture_weights(
                mixture_predictor,
                batch,
                batch_size=BATCH_SIZE,
                key="syms",
                return_sequence_probs=True,
                progress_bar=False
            )
            mixture_seq_preds_orig.extend([m[1] for m in mixture_out[0]])

            if USE_MIXTURE_WEIGHTS:
                prompt_text = [
                    build_inverse_prompt(t, "", mixture_token_fn(t), w)
                    for t, w in zip(text, mixture_out[1])
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

            mixture_out_inverse = get_mixture_weights(
                mixture_predictor,
                outputs,
                batch_size=BATCH_SIZE,
                key=None,
                return_sequence_probs=True,
                progress_bar=False
            )
            mixture_seq_preds_inverse.extend([m[1] for m in mixture_out_inverse[0]])

            for j in range(len(outputs)):
                elem = batch[j]
                elem["syms"] = outputs[j]
                output_fout.write(json.dumps(elem) + "\n")

            if DEBUG:
                break
            
        acc_orig = accuracy_score([1]*len(mixture_seq_preds_orig), [m > 0.5 for m in mixture_seq_preds_orig])
        acc_inverse = accuracy_score([1]*len(mixture_seq_preds_inverse), [m > 0.5 for m in mixture_seq_preds_inverse])
        print(f"Split: {split_name}")
        print(f"Original Acc: {acc_orig:.4f}")
        print(f"Inverse Acc: {acc_inverse:.4f}")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())