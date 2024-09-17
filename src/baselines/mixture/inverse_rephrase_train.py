
import json
import os
import random
import sys
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Dict, Union

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

from utils import build_inverse_prompt

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
parser.add_argument("--save_steps", type=int, default=200,
                    help="Number of steps to save the model.")
parser.add_argument("--logging_steps", type=int, default=100,
                    help="Number of steps to log the training metrics.")
parser.add_argument("--per_device_train_batch_size", type=int, default=64,
                    help="Batch size per device.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
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
parser.add_argument("--with_cluster_id", default=False, action="store_true",
                    help="Whether to use the cluster_id for conditioning the generation.")

parser.add_argument("--perc_gold_labels", type=float, default=0.5, 
                    help="Percentage of gold labels to use for conditioning the generation.")
 
parser.add_argument("--debug", default=False, action="store_true")
args = parser.parse_args()

MODEL_NAME = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side="left",
    add_eos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto",
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
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    return peft_model

def load_dataset() -> Union[list[Dict[str, list[int]]]]:
    N = 100 if args.debug else None
    train_data = []
    valid_data = []
    i = 0
    fname = "train.jsonl"
    fname += ".mixture" if args.prompt_type != "none" else ""
    with open(os.path.join(args.dataset_path, fname), "r") as fin:
        for line in fin:
            data = json.loads(line)
            train_data.append(data)
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
    with open(os.path.join(args.dataset_path, fname), "r") as fin:
        for line in fin:
            data = json.loads(line)
            valid_data.append(data)
            i += 1
            if i % 10_000 == 0:
                print(colored(f"Loaded {i} validation samples", "yellow"))
            if N is not None and i >= N:
                break
            
    if args.with_cluster_id:
        mapping = json.loads(open(os.path.join(args.dataset_path, "author_id_to_cluster_center.json")).read())
        for data in train_data:
            data["cluster_id"] = mapping[data["author_id"]]
        for data in valid_data:
            data["cluster_id"] = mapping[data["author_id"]]

    train_data = Dataset.from_list(train_data)
    valid_data = Dataset.from_list(valid_data)
    return train_data, valid_data

def get_valid_key(keys: list[str], valid_names: list[str]) -> str:
    for key in keys:
        if key in valid_names:
            return key
    assert False
    
def formatting_func(example: Dataset) -> list[str]:
    """Builds the inverse prompt for the generation task.
    """
    genkey = get_valid_key(example, ["generation", "rephrase"])
    origkey = get_valid_key(example, ["original", "unit"])

    texts = []
    for i in range(len(example[genkey])):
        if args.prompt_type != "none":
            text = build_inverse_prompt(
                example[genkey][i], 
                example[origkey][i], 
                example["mixture_tokens"][i], 
                example["mixture_probs"][i], 
                prompt_type=args.prompt_type,
                cluster_id=example["cluster_id"][i] if args.with_cluster_id else None,
            )
        else:
            text = build_inverse_prompt(
                example[genkey][i], 
                example[origkey][i],
                cluster_id=example["cluster_id"][i] if args.with_cluster_id else None,
            )
        texts.append(text)
        
    return texts

def mix_gold_labels(
    weights: list[list[tuple[int, int]]], 
    data: dict
):
    for i in range(len(data)):
        if random.random() > args.perc_gold_labels:
            gold_labels = [[abs(1 - label), label] for label in data[i]["tagger_labels"]]
            weights[i] = gold_labels
    return weights

def get_experiment_dir() -> str:
    dataset_name = os.path.basename(args.dataset_path)
    experiment_dir = os.path.join(dataset_name, args.prompt_type)
    if args.with_cluster_id:
        experiment_dir += "_cluster_id"
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
    
    peft_model = load_model()

    max_steps = 20 if args.debug else args.max_steps
    save_steps = 10 if args.debug else args.save_steps
    logging_steps = 1 if args.debug else args.logging_steps

    config = SFTConfig(
        max_seq_length=4096,
        report_to="tensorboard",
        output_dir=os.path.join(experiment_dir, run_name),
        run_name=run_name,
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
    )
    trainer = SFTTrainer(
        args=config,
        model=peft_model,
        train_dataset=train_samples,
        data_collator=DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer),
        formatting_func=formatting_func,
    )
    
    trainer.train()
    test_metrics = trainer.evaluate(test_samples)
    print(test_metrics)

if __name__ == "__main__":
    assert args.perc > 0.0 and args.perc <= 1.0
    sys.exit(main())