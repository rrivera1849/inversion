
import json
import os
import random
import sys
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Dict, Union

from peft import LoraConfig, get_peft_model
from termcolor import colored
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm

from utils import build_inverse_prompt

random.seed(43)

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
parser.add_argument("--loss_on_completions_only", default=False, action="store_true",
                    help="Whether to compute the loss only on completions.")

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

parser.add_argument("--perc_gold_labels", type=float, default=0.5, 
                    help="Percentage of gold labels to use for conditioning the generation.")
 
parser.add_argument("--debug", default=False, action="store_true")
args = parser.parse_args()

if args.prompt_type in ["probs", "logprobs"]:
    MAX_LENGTH = 2048
elif args.prompt_type == "tokens":
    MAX_LENGTH = 1024
else:
    MAX_LENGTH = (128 + 32) * 2 # 320

MODEL_NAME = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    model_max_lenght=MAX_LENGTH,
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

def tokenize_and_pad_to_fixed_length(
    sample: str
) -> Dict[str, list[int]]:
    result = tokenizer(
        sample,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    if args.loss_on_completions_only:
        output_ids = tokenizer("[/INST]\nOutput:", add_special_tokens=False)["input_ids"]
        _, end = [(start, start+len(output_ids)) for start in range(len(result["input_ids"])) if result["input_ids"][start:start+len(output_ids)] == output_ids][0]
        start = result["input_ids"].index(tokenizer("<s>", add_special_tokens=False)["input_ids"][0])
        result["labels"][start:end] = [-100] * (end-start+1)

    return result

def get_valid_key(keys: list[str], valid_names: list[str]) -> str:
    for key in keys:
        if key in valid_names:
            return key
    assert False
    
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

    genkey = get_valid_key(train_data[0].keys(), ["generation", "rephrase"])
    origkey = get_valid_key(train_data[0].keys(), ["original", "unit"])
    if args.prompt_type != "none":
        train_text = [build_inverse_prompt(data[genkey], data[origkey], data["mixture_tokens"], data["mixture_probs"], prompt_type=args.prompt_type) for data in train_data]
        valid_text = [build_inverse_prompt(data[genkey], data[origkey], data["mixture_tokens"], data["mixture_probs"], prompt_type=args.prompt_type) for data in valid_data]
    else:
        train_text = [build_inverse_prompt(data[genkey], data[origkey]) for data in train_data]
        valid_text = [build_inverse_prompt(data[genkey], data[origkey]) for data in valid_data]

    train_samples = [tokenize_and_pad_to_fixed_length(sample) for sample in tqdm(train_text)]
    valid_samples = [tokenize_and_pad_to_fixed_length(sample) for sample in tqdm(valid_text)]

    return train_samples, valid_samples

def mix_gold_labels(
    weights: list[list[tuple[int, int]]], 
    data: dict
):
    for i in range(len(data)):
        if random.random() > args.perc_gold_labels:
            gold_labels = [[abs(1 - label), label] for label in data[i]["tagger_labels"]]
            weights[i] = gold_labels
    return weights

def get_run_name(train_samples):
    run_name_items = OrderedDict({
        "dataset_name": os.path.basename(args.dataset_path),
        "lr": args.lr,
        "max-steps": args.max_steps,
        "num-samples": len(train_samples),
        "r": args.lora_r,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "perc": args.perc,
        "prompt": args.prompt_type,
    })
    if args.prompt_type in ["probs", "logprobs", "tokens"]:
        run_name_items["perc-gold-labels"] = args.perc_gold_labels
    run_name_items["debug"] = args.debug
    run_name = "Mistral-7B-v0.3-QLoRA"
    for key, value in run_name_items.items():
        run_name += f"_{key}={value}"
    return run_name

def main():
    if args.debug:
        print(colored("Running in DEBUG mode", "green"))
        
    for key, value in vars(args).items():
        print(colored(f"\t{key} = {value}", "yellow"))
    
    train_samples, test_samples = load_dataset()
    print(colored(f"len(train_samples)={len(train_samples)}", "yellow"))
    print(colored(f"len(test_samples)={len(test_samples)}", "yellow"))
    
    run_name = get_run_name(train_samples)
    print(colored(f"run_name={run_name}", "cyan"))
    
    output_dir = os.path.join("/data1/yubnub/changepoint/models/inverse", run_name)
    os.makedirs(output_dir, exist_ok=True)

    peft_model = load_model()

    max_steps = 20 if args.debug else args.max_steps
    save_steps = 10 if args.debug else args.save_steps
    logging_steps = 1 if args.debug else args.logging_steps
    training_args = TrainingArguments(
        report_to="tensorboard",
        run_name=run_name,
        output_dir=output_dir,
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

    trainer = Trainer(
        model=peft_model,
        train_dataset=train_samples,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        args=training_args,
    )
    # use_cache=True is incompatible with gradient checkpointing.
    peft_model.config.use_cache = False
    
    trainer.train()
    
    test_metrics = trainer.evaluate(test_samples)
    print(test_metrics)

if __name__ == "__main__":
    assert args.perc > 0.0 and args.perc <= 1.0
    sys.exit(main())