
import json
import os
import sys
from argparse import ArgumentParser
from typing import Union

import spacy
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig, 
    set_seed,
)
from tqdm import tqdm
from vllm import (
    LLM, 
    SamplingParams,
)

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
parser.add_argument("--num", type=int, default=6400)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--vllm", default=False, action="store_true")
parser.add_argument("--num_generations", type=int, default=1)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

DEBUG = args.debug

BATCH_SIZE = args.batch_size
DATA_PATH = "/data1/yubnub/changepoint/MUD_inverse/data"
assert os.path.isdir(os.path.join(DATA_PATH, args.dataset_name))

PROMPTING_DATA_PATH = os.path.join(DATA_PATH, args.dataset_name, "inverse_output")
os.makedirs(PROMPTING_DATA_PATH, exist_ok=True)

INVERSE_SAVEPATH = "/data1/yubnub/changepoint/models/inverse"
INVERSE_MODELS = {
    "none": "r=32_alpha=64_dropout=0.1_perc=1.0_perc-gold-labels=0.5",
    "tokens": "r=32_alpha=64_dropout=0.1_perc=1.0_perc-gold-labels=0.0",
    "probs": "r=32_alpha=64_dropout=0.1_perc=1.0_perc-gold-labels=0.0",
}
assert args.prompt_type in INVERSE_MODELS
NLP = spacy.load("en_core_web_sm")

def create_output_file(
    generation_args: dict,
):
    output_fname = os.path.join(
        PROMPTING_DATA_PATH, 
        f"{args.prompt_type}_{args.num}"
    )
    for name, value in generation_args.items():
        output_fname += f"_{name}={value}"
    output_fname += ".jsonl" 
    output_fname += f".vllm_n={args.num_generations}" if args.vllm else f"n={args.num_generations}"
    output_fname += ".debug" if DEBUG else ""
    output_fout = open(output_fname, "w+")
    print(f"Writing to {output_fname}")
    return output_fout

def load_inverse_model(
    prompt_type: str = "none",
    vllm: bool = False,
) -> Union[tuple[AutoModelForCausalLM, AutoTokenizer], tuple[LLM, None]]:
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
    
    if vllm:
        checkpoint_path += "_merged"
        inverse_model = LLM(checkpoint_path)
        inverse_tokenizer = None # VLLM integrates tokenizer in LLM class
        return inverse_model, inverse_tokenizer
    else:
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

def get_prompt_text(
    text: list[str],
    mixture_predictor,
) -> list[str]:
    """Generates the prompt text for the inverse model.
    """
    if args.prompt_type != "none":
        mixture_probs = get_mixture_weights(
                mixture_predictor,
                text,
                key=None,
                batch_size=BATCH_SIZE,
                progress_bar=False,
            )
        mixture_tokens = [mixture_predictor.tokenizer.tokenize(t) for t in text]
        prompt_text = [
                build_inverse_prompt(sample, "", tokens, probs, prompt_type=args.prompt_type, no_stop_tokens=True)
                for sample, tokens, probs in zip(text, mixture_tokens, mixture_probs)
            ]
    else:
        prompt_text = [
            build_inverse_prompt(t, "", no_stop_tokens=True) for t in text
        ]

    return prompt_text

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

def get_HF_generations(
    prompt_text: list[str],
    text: list[str],
    inverse_model: AutoModelForCausalLM,
    inverse_tokenizer: AutoTokenizer,
    generation_args: dict,
) -> list[str]:
    """Generates the inverse text using a HuggingFace model.
    """
    generation_config = GenerationConfig(
        do_sample=True,
        max_new_tokens=128+32,
        **generation_args,
    )

    tokenized_text = inverse_tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

    all_generations = []
    for _ in range(args.num_generations):
        generations = inverse_model.generate(
                **tokenized_text,
                generation_config=generation_config,
            )
        generations = inverse_tokenizer.batch_decode(generations, skip_special_tokens=True)
        all_generations.append(generations)

    outputs = [[] for _ in range(len(prompt_text))]
    for j in range(len(prompt_text)):
        sample_generations = [all_generations[k][j] for k in range(args.num_generations)]
        for gen in sample_generations:
            gen = gen[gen.index("Output: ")+len("Output: "):]
            if "\n#####\n" in gen:
                gen = gen[:gen.index("\n#####\n")]
            else:
                gen = get_best_sentence_substring(gen, text[j])
            outputs[j].append(gen)

    outputs = [list(set(o)) for o in outputs]
    return outputs

def get_VLLM_generations(
    prompt_text: list[str],
    inverse_model: LLM,
    generation_args: dict,
) -> list[str]:
    sampling_params = SamplingParams(
        n=args.num_generations,
        max_tokens=128+32,
        stop="\n#####\n",
        seed=43,
        **generation_args,
    )
    
    outputs = inverse_model.generate(prompt_text, sampling_params)
    outputs = [list(set([o.text for o in out.outputs])) for out in outputs]
    return outputs

def main():
    mixture_predictor = None
    if args.prompt_type != "none":
        mixture_predictor = load_mixture_predictor(args.mixture_predictor_path)

    data = load_prompting_data()
    inverse_model, inverse_tokenizer = load_inverse_model(args.prompt_type, args.vllm)

    generation_args = {
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    output_fout = create_output_file(generation_args)

    for batch_idx in tqdm(range(0, len(data), BATCH_SIZE)):
        batch = data[batch_idx:batch_idx+BATCH_SIZE]
        text = [b["rephrase"] for b in batch]
        prompt_text = get_prompt_text(text, mixture_predictor)

        if args.vllm:
            outputs = get_VLLM_generations(
                prompt_text,
                inverse_model,
                generation_args,
            )
        else:
            outputs = get_HF_generations(
                prompt_text,
                text,
                inverse_model, 
                inverse_tokenizer, 
                generation_args, 
            )

        for j in range(len(outputs)):
            elem = batch[j]
            elem["inverse"] = outputs[j]
            elem["inverse_prompt"] = prompt_text[j]
            output_fout.write(json.dumps(elem) + "\n")
            output_fout.flush()

        if DEBUG:
            # import pdb; pdb.set_trace()
            break

    return 0

if __name__ == "__main__":
    sys.exit(main())