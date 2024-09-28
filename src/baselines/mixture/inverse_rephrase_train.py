
import json
import os
import random
import sys
from math import ceil
from argparse import ArgumentParser
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
from accelerate import PartialState
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from termcolor import colored
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
)
from trl import (
    SFTConfig, 
    SFTTrainer, 
    DataCollatorForCompletionOnlyLM,
    set_seed
)

from embedding_utils import get_luar_instance_embeddings, load_luar_model_and_tokenizer
from utils import build_inverse_prompt

torch.autograd.set_detect_anomaly(True)

set_seed(43)
OUTPUT_DIR = "/data1/yubnub/changepoint/models/inverse"

parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str,
                    default="./datasets/s2orc_roberta-large_200000_inverse",
                    help="Directory where the dataset is stored.")
parser.add_argument("--lr", type=float, default=2e-5,
                    help="Learning rate.")
parser.add_argument("--max_steps", type=int, default=1000,
                    help="Number of training steps.")
parser.add_argument("--resume_from_checkpoint", default=False,
                    action="store_true",
                    help="Whether to resume from a checkpoint.")
parser.add_argument("--save_steps", type=int, default=200,
                    help="Number of steps to save the model.")
parser.add_argument("--logging_steps", type=int, default=100,
                    help="Number of steps to log the training metrics.")
parser.add_argument("--per_device_train_batch_size", type=int, default=16,
                    help="Batch size per device.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Number of gradient accumulation steps.")
parser.add_argument("--perc", type=float, default=1.0,
                    help="Percentage of the training dataset to use.")

# LoRA Parameters:
parser.add_argument("--lora_r", type=float, default=32,
                    help="Dimensionality of low-rank matrices used for \delta W.")
parser.add_argument("--lora_alpha", type=float, default=64,
                    help="Weight used on update \lora_alpha * \deltaW.")
parser.add_argument("--lora_dropout", type=float, default=0.1,
                    help="Dropout parameter for LoRA adapter layers.")

parser.add_argument("--prompt_type", type=str, default="none",
                    choices=["none", "tokens", "probs", "logprobs"],
                    help="Type of prompt to use for conditioning the generation.")
parser.add_argument("--targetted_mode", type=str, default=None,
                    choices=["embeddings", "examples"],
                    help="Will use a Style Embedding to condition the generation.")

parser.add_argument("--perc_gold_labels", type=float, default=0.5, 
                    help="Percentage of gold labels to use for conditioning the generation.")
 
parser.add_argument("--debug", default=False, action="store_true")
args = parser.parse_args()

MODEL_NAME = "mistralai/Mistral-7B-v0.3"

def get_max_seq_length():
    if args.prompt_type == "none":
        return 512
    elif args.prompt_type == "none" and args.targetted_mode == "examples":
        return 2048
    elif args.prompt_type == "tokens":
        return 1024
    else:
        return 2048
    
def get_device_string():
    device_string = PartialState().process_index 
    return device_string

def load_model_and_tokenizer():
    device_string = get_device_string()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map={'':device_string},
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    # https://github.com/huggingface/peft/issues/137
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias="none", # keeps the original model perform equally when the adapter is not used
    )
    model = get_peft_model(model, peft_config)

    # If we do this before, PEFT will add a LORA layer to the style_embedding_proj
    if args.targetted_mode == "embeddings":
        model.style_embedding_proj = nn.Linear(512, model.config.hidden_size)
        model.original_save_pretrained = model.save_pretrained
        model.save_pretrained = custom_save_pretrained.__get__(model)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side="right" if args.targetted_mode == "embeddings" else "left",
        add_eos_token=True,
    )

    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_dataset() -> tuple[list[str, list[str]]]:
    N = 100 if args.debug else None
    train_data = []
    valid_data = []
    i = 0
    fname = "train.jsonl"
    fname += ".mixture" if args.prompt_type != "none" else ""
    train_author_to_idx = defaultdict(list)
    with open(os.path.join(args.dataset_path, fname), "r") as fin:
        for line in fin:
            data = json.loads(line)
            train_data.append(data)
            train_author_to_idx[data["author_id"]].append(i)
            i += 1
            if i % 10_000 == 0:
                print(colored(f"Loaded {i} training samples", "yellow"))
            if N is not None and i >= N:
                break
    if args.perc < 1.0:
        random.shuffle(train_data)
        train_data = train_data[:int(args.perc * len(train_data))]
    i = 0
    fname = "valid.jsonl"
    fname += ".mixture" if args.prompt_type != "none" else ""
    valid_author_to_idx = defaultdict(list)
    with open(os.path.join(args.dataset_path, fname), "r") as fin:
        for line in fin:
            data = json.loads(line)
            valid_data.append(data)
            valid_author_to_idx[data["author_id"]].append(i)
            i += 1
            if i % 10_000 == 0:
                print(colored(f"Loaded {i} validation samples", "yellow"))
            if N is not None and i >= N:
                break
            
    if args.targetted_mode == "examples":
        for author, indices in train_author_to_idx.items():
            # multiple Rephrase -> Unit mappings:
            all_author_examples = set([train_data[idx]["unit"] for idx in indices])
            for i in range(len(indices)):
                current_unit = train_data[indices[i]]["unit"]
                train_data[indices[i]]["examples"] = [example for example in all_author_examples if example != current_unit]
        for author, indices in valid_author_to_idx.items():
            # multiple Rephrase -> Unit mappings:
            all_author_examples = set([valid_data[idx]["unit"] for idx in indices])
            for i in range(len(indices)):
                current_unit = valid_data[indices[i]]["unit"]
                valid_data[indices[i]]["examples"] = [example for example in all_author_examples if example != current_unit]
            
    train_data = Dataset.from_list(train_data)
    valid_data = Dataset.from_list(valid_data)

    if args.targetted_mode == "embeddings":
        # get the style embeddings for the training and validation data
        luar, luar_tok = load_luar_model_and_tokenizer()
        key = "unit"
        if torch.cuda.is_available():
            device = torch.device("cuda:{}".format(get_device_string()))
            luar.to(device)

        print(colored("Computing style embeddings for training data", "green"))
        train_embeddings = get_luar_instance_embeddings(
            train_data[key], luar, luar_tok, progress_bar=True, batch_size=1024,
        ).cpu().tolist()
        train_data = train_data.add_column("style_embedding", train_embeddings)

        print(colored("Computing style embeddings for validation data", "green"))
        valid_embeddings = get_luar_instance_embeddings(
            valid_data[key], luar, luar_tok, progress_bar=True, batch_size=1024,
        ).cpu().tolist()
        valid_data = valid_data.add_column("style_embedding", valid_embeddings)
    
    return train_data, valid_data

def formatting_func(example: Dataset) -> list[str]:
    """Builds the inverse prompt for the generation task.
    """
    genkey = "rephrase"
    origkey = "unit"

    texts = []
    for i in range(len(example[genkey])):
        if args.prompt_type != "none":
            text = build_inverse_prompt(
                example[genkey][i], 
                example[origkey][i], 
                example["mixture_tokens"][i], 
                example["mixture_probs"][i], 
                prompt_type=args.prompt_type,
            )
        elif args.targetted_mode == "examples":
            examples = example["examples"][i]
            num_examples = ceil(np.random.beta(2, 1) * 10)
            num_examples = max(1, min(num_examples, len(examples)))
            examples = random.sample(examples, num_examples)
            
            text = build_inverse_prompt(
                example[genkey][i], 
                example[origkey][i],
                examples=examples,
            )
        else:
            text = build_inverse_prompt(
                example[genkey][i], 
                example[origkey][i],
            )
        texts.append(text)
        
    return texts

def mix_gold_labels(
    weights: list[list[tuple[int, int]]], 
    data: dict
):
    """Mixes the gold token probabilities in our training data.
    """
    for i in range(len(data)):
        if random.random() > args.perc_gold_labels:
            gold_labels = [[abs(1 - label), label] for label in data[i]["tagger_labels"]]
            weights[i] = gold_labels
    return weights

def get_experiment_dir() -> str:
    dataset_name = os.path.basename(args.dataset_path)
    experiment_dir = os.path.join(dataset_name, args.prompt_type)
    if args.targetted:
        experiment_dir += "_targetted={}".format(args.targetted_mode)
    if args.debug:
        experiment_dir += "_debug"
    experiment_dir = os.path.join(OUTPUT_DIR, experiment_dir)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def get_run_name() -> str:
    run_name = f"r={args.lora_r}_alpha={args.lora_alpha}_dropout={args.lora_dropout}_perc={args.perc}_perc-gold-labels={args.perc_gold_labels}"
    return run_name

def save_hparams(experiment_dir: str, run_name: str) -> None:
    hparams = OrderedDict(
        dataset_path=args.dataset_path,
        lr=args.lr,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        perc=args.perc,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        prompt_type=args.prompt_type,
        perc_gold_labels=args.perc_gold_labels,
    )
    os.makedirs(os.path.join(experiment_dir, run_name), exist_ok=True)
    with open(os.path.join(experiment_dir, run_name, "hparams.json"), "w") as fout:
        json.dump(hparams, fout, indent=4)
        
class DataCollatorWithStyleEmbeddings(DataCollatorForCompletionOnlyLM):
    def __init__(self, emb, style_emb_proj, response_template, tokenizer):
        super().__init__(response_template=response_template, tokenizer=tokenizer)
        self.emb = emb
        self.style_emb_proj = style_emb_proj
        process_index = get_device_string()
        self.device = torch.device("cuda:{}".format(process_index))
        
    def __call__(self, examples):
        examples = Dataset.from_list(examples)

        # format -> tokenize -> collate with DataCollatorForCompletionOnlyLM
        outputs = formatting_func(examples)
        outputs = [self.tokenizer(ex) for ex in outputs]
        outputs = super().__call__(outputs)
        outputs = outputs.to(self.device)

        B = outputs["labels"].size(0)
        # 1. Get the Mistral embeddings:
        outputs["inputs_embeds"] = self.emb(outputs["input_ids"]).detach()
        outputs["inputs_embeds"].requires_grad = True
        # 2. Project style embeddings to Mistral space:
        style_embeddings = torch.tensor(examples["style_embedding"]).to(self.device)
        style_embeddings = self.style_emb_proj(style_embeddings).unsqueeze(1)
        # 3. Input = [style_embedding || embeds]
        outputs["inputs_embeds"] = torch.cat((style_embeddings, outputs["inputs_embeds"]), dim=1)

        # 5. Shift the labels to the right to account for the style embedding:
        new_label = torch.zeros(B, 1, dtype=torch.long).fill_(-100).to(self.device)
        outputs["labels"] = torch.cat((new_label, outputs["labels"]), dim=1)

        # 6. Add one extra element to the attention mask:
        outputs["attention_mask"] = torch.cat(
            (torch.ones(B, 1, dtype=torch.long).to(self.device), outputs["attention_mask"]),
            dim=1,
        )

        outputs.pop("input_ids")
        return outputs
    
def custom_save_pretrained(self, *args, **kwargs):
    """Saves the style_embedding_proj layer, along with the rest of the model.
    """
    path = args[0]
    torch.save(self.style_embedding_proj.state_dict(), os.path.join(path, "style_embedding_proj.pt"))
    self.original_save_pretrained(*args, **kwargs)
    
def main():
    if args.debug:
        print(colored("Running in DEBUG mode", "green"))
        
    for key, value in vars(args).items():
        print(colored(f"\t{key} = {value}", "yellow"))
    
    train_samples, test_samples = load_dataset()
    response_template = "[/INST]\n###Output:"
    
    print(colored(f"len(train_samples)={len(train_samples)}", "yellow"))
    print(colored(f"len(test_samples)={len(test_samples)}", "yellow"))

    experiment_dir = get_experiment_dir()
    run_name = get_run_name()
    save_hparams(experiment_dir, run_name)
    
    model, tokenizer = load_model_and_tokenizer()

    max_steps = 20 if args.debug else args.max_steps
    save_steps = 10 if args.debug else args.save_steps
    logging_steps = 1 if args.debug else args.logging_steps

    config = SFTConfig(
        max_seq_length=get_max_seq_length(),
        report_to="tensorboard",
        output_dir=os.path.join(experiment_dir, run_name),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="adamw_hf",
        bf16=True,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        max_steps=max_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        warmup_steps=int(0.01 * args.max_steps),
        # https://discuss.huggingface.co/t/training-llama-with-lora-on-multiple-gpus-may-exist-bug/47005/3
        ddp_find_unused_parameters=False,
        gradient_checkpointing_kwargs={"use_reentrant" : False},
    )
    
    if args.targetted_mode == "embeddings":
        config.dataset_kwargs = {"skip_prepare_dataset": True}
        config.remove_unused_columns = False
        config.dataset_text_field = ""
        config.dataloader_pin_memory = False
        form_fn = None
        
        collator = DataCollatorWithStyleEmbeddings(
            emb=model.get_input_embeddings(),
            style_emb_proj=model.style_embedding_proj,
            response_template=response_template, 
            tokenizer=tokenizer
        )
    else:
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template, 
            tokenizer=tokenizer
        )
        form_fn = formatting_func
    
    trainer = SFTTrainer(
        args=config,
        model=model,
        train_dataset=train_samples,
        eval_dataset=test_samples,
        data_collator=collator,
        formatting_func=form_fn,
        
    )
    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

if __name__ == "__main__":
    assert args.perc > 0.0 and args.perc <= 1.0
    sys.exit(main())