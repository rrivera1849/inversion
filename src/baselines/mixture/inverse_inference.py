
import json
import os
import random
import sys
from argparse import ArgumentParser
from typing import Union

import pandas as pd
import spacy
import torch
import torch.nn as nn
from termcolor import colored
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

from embedding_utils import load_luar_model_and_tokenizer, get_luar_author_embeddings
from utils import (
    build_inverse_prompt,
    get_mixture_weights, 
    load_mixture_predictor,
)

set_seed(43)

parser = ArgumentParser()
parser.add_argument("--dataset_name", type=str, 
                    default="data.jsonl.filtered.cleaned_kmeans_100")
parser.add_argument("--filename", type=str, default="test.small.jsonl")
parser.add_argument("--prompt_type", type=str, default="none")
parser.add_argument("--mixture_predictor_path", type=str, default=None,
                    help="Path to the mixture predictor model.")
parser.add_argument("--num", type=int, default=6400)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--vllm", default=False, action="store_true")
parser.add_argument("--num_generations", type=int, default=1)
parser.add_argument("--with_style_embeddings", default=False, action="store_true")
parser.add_argument("--with_examples", default=False, action="store_true")
parser.add_argument("--num_examples", type=int, default=None)
parser.add_argument("--targetted_mode", type=str, default=None,
                    choices=["author", "plagiarism"])
parser.add_argument("--debug", default=False, action="store_true")
args = parser.parse_args()

if args.with_style_embeddings:
    assert not args.vllm, "VLLM does not support `inputs_embeds`"

DEBUG = args.debug

BATCH_SIZE = args.batch_size
DATA_PATH = "/data1/yubnub/changepoint/MUD_inverse/data"
assert os.path.isdir(os.path.join(DATA_PATH, args.dataset_name))

PROMPTING_DATA_PATH = os.path.join(DATA_PATH, args.dataset_name, "inverse_output")
os.makedirs(PROMPTING_DATA_PATH, exist_ok=True)

INVERSE_SAVEPATH = "/data1/yubnub/changepoint/models/inverse"
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
    if "targetted" in args.prompt_type:
        output_fname += f".targetted_mode={args.targetted_mode}"
    if args.num_examples is not None:
        output_fname += f"_num_examples={args.num_examples}"
    output_fname += ".debug" if DEBUG else ""
    
    if os.path.exists(output_fname):
        output_fout = open(output_fname, "a+")
        last_line_num = sum(1 for line in open(output_fname))
    else:
        output_fout = open(output_fname, "w+")
        last_line_num = 0
        
    print(f"Writing to {output_fname}, last line number: {last_line_num}")
    return output_fout, last_line_num

def load_inverse_model(
    prompt_type: str = "none",
    vllm: bool = False,
) -> Union[tuple[AutoModelForCausalLM, AutoTokenizer], tuple[LLM, None]]:
    """Loads our inversion model, either with or without Mixture Weights in the prompt.
    """
    checkpoint_path = os.path.join(
        INVERSE_SAVEPATH, 
        args.dataset_name, 
        prompt_type, 
        "r=32_alpha=64_dropout=0.1_perc=1.0_perc-gold-labels=0.5", 
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

def load_style_embedding_projection(
    prompt_type: str = "none",
) -> Union[tuple[AutoModelForCausalLM, AutoTokenizer], tuple[LLM, None]]:
    """Loads our inversion model, either with or without Mixture Weights in the prompt.
    """
    checkpoint_path = os.path.join(
        INVERSE_SAVEPATH, 
        args.dataset_name, 
        prompt_type, 
        "r=32_alpha=64_dropout=0.1_perc=1.0_perc-gold-labels=0.5", 
        f"checkpoint-{args.num}",
        "style_embedding_proj.pt"
    )
    state_dict = torch.load(checkpoint_path)
    style_embedding_proj = nn.Linear(512, 4096)
    style_embedding_proj.load_state_dict(state_dict, strict=True)
    style_embedding_proj.to("cuda")
    style_embedding_proj.eval()
    return style_embedding_proj

def load_prompting_data(
):
    """Loads the Prompting Data.
    """
    filename = os.path.join(DATA_PATH, args.dataset_name, args.filename)
    data = [json.loads(line) for line in open(filename)]
    return data

def get_prompt_text(
    text: list[str],
    mixture_predictor = None,
    examples: list[list[str]] = None,
) -> list[str]:
    """Generates the prompt text for the inverse model.
    """
    if "none" not in args.prompt_type:
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
    elif examples is not None:
        prompt_text = [
            build_inverse_prompt(t, "", examples=e, no_stop_tokens=True) for t, e in zip(text, examples)
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

def get_input_embeds(
    prompt_text: list[str], 
    inverse_model: AutoModelForCausalLM, 
    style_embeddings: torch.Tensor, 
    embedding_indices: list[int], 
    tokenized_text: dict,
):
    """Embeds the input using Mistral, and concatenates the target style embedding.
    """
    # 1. Get the Mistral embeddings for the input:
    emb = inverse_model.get_input_embeddings()
    inputs_embeds = emb(tokenized_text["input_ids"])

    # 2. Find the indices of the <s> tokens:
    indices = torch.where(tokenized_text["input_ids"] == 1)[1]
    new_inputs_embeds = []
    for j, idx in enumerate(indices):
        idx = idx.item()
        # New input concatenates the style embedding before the <s> token:
        new_inputs_embeds.append(torch.cat(
                (inputs_embeds[j:j+1, :idx], 
                 style_embeddings[embedding_indices[j]:embedding_indices[j]+1], 
                 inputs_embeds[j:j+1, idx:]),
                dim=1,
            ))
    new_inputs_embeds = torch.cat(new_inputs_embeds, dim=0)
    tokenized_text["inputs_embeds"] = new_inputs_embeds

    # 3. Add a "1" to the attention mask for the style embedding, to account for the new token:
    tokenized_text["attention_mask"] = torch.cat(
            (tokenized_text["attention_mask"],
             torch.ones(len(prompt_text), 1, dtype=torch.long).to(inverse_model.device)), 
            dim=1,
        )
    
    # 4. Remove the input_ids, as we are using inputs_embeds:
    tokenized_text.pop("input_ids")
    return tokenized_text

@torch.no_grad()
def get_HF_generations(
    prompt_text: list[str],
    text: list[str],
    inverse_model: AutoModelForCausalLM,
    inverse_tokenizer: AutoTokenizer,
    generation_args: dict,
    style_embeddings: torch.Tensor = None,
    embedding_indices: list[int] = None,
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
    
    if style_embeddings is not None:
        assert embedding_indices is not None
        tokenized_text = get_input_embeds(
            prompt_text, 
            inverse_model, 
            style_embeddings, 
            embedding_indices, 
            tokenized_text
        )

    all_generations = []
    for _ in tqdm(range(args.num_generations)):
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
            if "targetted" not in args.prompt_type:
                # When providing input embeddings, we only get the output
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

def convert_to_targetted_mode(data: list[dict], targetted_mode: str):
    """Converts the data to targetted mode for Plagiarism or Authorship ID.
    """
    data = pd.merge(data, data, how="cross")

    if targetted_mode == "author":
        for col in data.columns:
            if "author_id" in col:
                continue
            if col.endswith("_x"):
                data[col] = data[col].apply(lambda x: x[:len(x)//2])
            else:
                data[col] = data[col].apply(lambda x: x[len(x)//2:])

    to_explode = [col for col in data.columns if col.endswith("_x") and "author_id" not in col]
    data = data.explode(to_explode)
    return data

def get_num_expected_rows(data: pd.DataFrame, targetted_mode: str):
    if targetted_mode == "author":
        num_expected = data["rephrase"].apply(lambda x: len(x) // 2).sum() * 100
    else:
        num_expected = data["rephrase"].apply(lambda x: len(x)).sum() * 100
    return num_expected

def clean_duplicate_units(data: pd.DataFrame):
    """Have to be careful when running authorship ID, as we need to remove the 
       repeated units. 
    """
    def get_indices_of_repeated(lst: list):
        seen = set()
        indices = []
        for j, elem in enumerate(lst):
            if elem in seen:
                indices.append(j)
            seen.add(elem)
        return indices
    
    def remove_repeated(row):
        indices_to_remove = get_indices_of_repeated(row["unit"])
        for key, value in row.items():
            if isinstance(value, list):
                row[key] = [v for j, v in enumerate(value) if j not in indices_to_remove]
        return row
    
    data = data.apply(remove_repeated, axis=1)
    return data

def main():
    if args.targetted_mode == "plagiarism":
        raise NotImplementedError("Plagiarism mode is not supported yet.")
    
    mixture_predictor = None
    if "none" not in args.prompt_type:
        mixture_predictor = load_mixture_predictor(args.mixture_predictor_path)

    data = load_prompting_data()
    
    if args.targetted_mode is not None:
        assert args.with_style_embeddings or args.with_examples, "Targetted mode requires either style embeddings or examples!"
        print(colored("Targetted Mode", "yellow"))
        random.shuffle(data)
        data = pd.DataFrame(data)
        data = data.groupby("author_id").agg(list).reset_index()
        if args.targetted_mode == "author":
            data = clean_duplicate_units(data)
        num_expected = get_num_expected_rows(data, args.targetted_mode)
        data = convert_to_targetted_mode(data, args.targetted_mode)
        assert len(data) == num_expected
    
    style_embeddings = None
    if args.with_style_embeddings:
        print(colored("Calculating Style Embeddings", "green"))
        
        # 1. There are only `num_authors` unique targets:
        mapping = {}
        target_text = []
        authors = data.author_id_y.unique()
        for j, author in enumerate(authors):
            mapping[author] = j
            target = data[data.author_id_y == author].iloc[0]["unit_y"]
            target_text.append(target)

        # 2. Embed the target text using the LUAR model, and project it to the
        #    Mistral embedding space
        style_embedding_projector = load_style_embedding_projection(args.prompt_type)
        luar, luar_tok = load_luar_model_and_tokenizer()
        luar.to("cuda")
        style_embeddings = [get_luar_author_embeddings(target, luar, luar_tok) for target in target_text]
        style_embeddings = torch.cat(style_embeddings, dim=0)
        style_embeddings = style_embedding_projector(style_embeddings)
        style_embeddings = style_embeddings.unsqueeze(1)
        style_embeddings = style_embeddings.to(torch.bfloat16)

        # 3. Map the targetted samples to their respective style embeddings:
        data["embedding_idx"] = data.author_id_y.map(mapping)
        data = data.to_dict(orient="records")
    elif args.with_examples:
        data = data.to_dict(orient="records")

    inverse_model, inverse_tokenizer = load_inverse_model(args.prompt_type, args.vllm)

    generation_args = {
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    output_fout, last_line_num = create_output_file(generation_args)

    for batch_idx in tqdm(range(last_line_num, len(data), BATCH_SIZE)):
        batch = data[batch_idx:batch_idx+BATCH_SIZE]
        key = "rephrase_x" if args.with_style_embeddings or args.with_examples else "rephrase"
        text = [b[key] for b in batch]
        if args.with_examples:
            if args.num_examples is not None:
                examples = [b["unit_y"][:args.num_examples] for b in batch]
            else:
                examples = [b["unit_y"] for b in batch]
            prompt_text = get_prompt_text(text, examples=examples)
        else:
            prompt_text = get_prompt_text(text, mixture_predictor)
        
        if args.with_style_embeddings:
            embedding_indices = [b["embedding_idx"] for b in batch]
        else:
            embedding_indices = None

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
                style_embeddings,
                embedding_indices,
            )

        for j in range(len(outputs)):
            elem = batch[j]
            elem["inverse"] = outputs[j]
            elem["inverse_prompt"] = prompt_text[j]
            output_fout.write(json.dumps(elem) + "\n")
            output_fout.flush()

        if DEBUG:
            import pdb; pdb.set_trace()
            break

    return 0

if __name__ == "__main__":
    sys.exit(main())