"""Use PEFT to train fine-tune an LLM to invert the rephrases.

1. Rephrase -> Original
2. Prompt(MixtureWeights) + Rephrase -> Original
    - MixtureWeights sampled from the gold labels, or mixture predictor with probability x%.
"""

import json
import os
import random
import sys
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Dict, Union

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from termcolor import colored
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm

from utils import build_inverse_prompt, get_mixture_weights, load_mixture_predictor

random.seed(43)

parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str,
                    default="./datasets/all_roberta-large_250000_stratified_inverse",
                    help="Directory where the dataset is stored.")
parser.add_argument("--lr", type=float, default=2e-5,
                    help="Learning rate.")
parser.add_argument("--max_steps", type=int, default=1000,
                    help="Number of training steps.")
parser.add_argument("--per_device_train_batch_size", type=int, default=64,
                    help="Batch size per device.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                    help="Number of gradient accumulation steps.")

# LoRA Parameters:
parser.add_argument("--lora_r", type=float, default=32,
                    help="Dimensionality of low-rank matrices used for \delta W.")
parser.add_argument("--lora_alpha", type=float, default=64,
                    help="Weight used on update \lora_alpha * \deltaW.")
parser.add_argument("--lora_dropout", type=float, default=0.1,
                    help="Dropout parameter for LoRA adapter layers.")

parser.add_argument("--use_mixture_weights", default=False, action="store_true",
                    help="Use mixture weights to condition the generation.")         
parser.add_argument("--use_mixture_weights_no_probs", default=False, action="store_true",
                    help="If True, will set up a simple prompt with the human tokens to keep, but nothing else.")
parser.add_argument("--perc_gold_labels", type=float, default=0.5, 
                    help="Percentage of gold labels to use for conditioning the generation.")
 
parser.add_argument("--debug", default=False, action="store_true")
args = parser.parse_args()

if args.use_mixture_weights:
    MAX_LENGTH = 2048
elif args.use_mixture_weights_no_probs:
    MAX_LENGTH = 1024
else:
    MAX_LENGTH = (128 + 32) * 2

MODEL_NAME = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    model_max_lenght=MAX_LENGTH,
    padding_side="left", # left padding solves the following problem: the quick brown <PAD> --> prediction uses PAD token 
    add_eos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

def load_model():
    quantization_config = BitsAndBytesConfig(
        # Load the model with 4-bit quantization
        load_in_4bit=True,
        # Use double quantization
        bnb_4bit_use_double_quant=True,
        # Use 4-bit Normal Float for storing the base model weights in GPU memory
        bnb_4bit_quant_type="nf4",
        # De-quantize the weights to 16-bit (Brain) float before the forward/backward pass
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto",
        quantization_config=quantization_config,
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        # Bias parameters to train. 'none' is recommended to keep the original model performing equally when turning off the adapter.
        bias="none",
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
    return result

def load_dataset() -> Union[list[Dict[str, list[int]]]]:

    N = 100 if args.debug else None
    train_data = []
    valid_data = []
    i = 0
    with open(os.path.join(args.dataset_path, "train.jsonl"), "r") as fin:
        for line in fin:
            data = json.loads(line)
            train_data.append(data)
            i += 1
            if N is not None and i >= N:
                break
    i = 0
    with open(os.path.join(args.dataset_path, "valid.jsonl"), "r") as fin:
        for line in fin:
            data = json.loads(line)
            valid_data.append(data)
            i += 1
            if N is not None and i >= N:
                break
    
    if args.use_mixture_weights or args.use_mixture_weights_no_probs:
        mixture_predictor = load_mixture_predictor()

        train_weights = get_mixture_weights(mixture_predictor, train_data)
        valid_weights = get_mixture_weights(mixture_predictor, valid_data)
        train_weights = mix_gold_labels(train_weights, train_data)
        # RRS - shouldn't mix gold labels in validation data!
        # valid_weights = mix_gold_labels(valid_weights, valid_data)

        train_data_tokens = [mixture_predictor.tokenizer.tokenize(data["generation"]) for data in tqdm(train_data)]
        valid_data_tokens = [mixture_predictor.tokenizer.tokenize(data["generation"]) for data in tqdm(valid_data)]

        train_text = [build_inverse_prompt(data["generation"], data["original"], tokens, weights, simple_prompt=args.use_mixture_weights_no_probs) for data, tokens, weights in zip(train_data, train_data_tokens, train_weights)]
        valid_text = [build_inverse_prompt(data["generation"], data["original"], tokens, weights, simple_prompt=args.use_mixture_weights_no_probs) for data, tokens, weights in zip(valid_data, valid_data_tokens, valid_weights)]
    else:
        train_text = [build_inverse_prompt(data["generation"], data["original"]) for data in train_data]
        valid_text = [build_inverse_prompt(data["generation"], data["original"]) for data in valid_data]

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
        "lr": args.lr,
        "max-steps": args.max_steps,
        "num-samples": len(train_samples),
        "r": args.lora_r,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
    })
    if args.use_mixture_weights:
        run_name_items["use-mixture-weights"] = args.use_mixture_weights
        run_name_items["perc-gold-labels"] = args.perc_gold_labels
    if args.use_mixture_weights_no_probs:
        run_name_items["simple-prompt"] = args.use_mixture_weights_no_probs
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
    save_steps = 10 if args.debug else 200
    logging_steps = 1 if args.debug else 100
    training_args = TrainingArguments(
        report_to="tensorboard",
        run_name=run_name,
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
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
    sys.exit(main())