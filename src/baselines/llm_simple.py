"""Use PEFT to train an LLM on human text, continuations, rephrases, or rephrases with context.

Good Tutorial on QLoRA for Fine-Tuning LLMs: https://mlflow.org/docs/latest/llms/transformers/tutorials/fine-tuning/transformers-peft.html
"""

import os
import random
import sys
from argparse import ArgumentParser
from typing import Dict, Union

import mlflow
import torch
from datasets import load_from_disk
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

sys.path.append("../../scripts/dataset/changepoint")
from prompts import PROMPT_NAMES

random.seed(43)

parser = ArgumentParser()
parser.add_argument("--dirname", type=str,
                    default="/data1/yubnub/changepoint/s2orc_changepoint/unit_128",
                    help="Directory where the dataset is stored.")
parser.add_argument("--prompt", type=str, default="none",
                    choices=PROMPT_NAMES + ["none", "all"],
                    help="Which prompt-type to train on. If 'all', will train a single model on all types.")

parser.add_argument("--num_samples", type=int, default=100_000,
                    help="Number of samples to draw from the dataset in total.")
parser.add_argument("--units_perc_per_sample", type=float, default=0.10,
                    help="Percentage of units to take from each paper when creating the dataset.")

# LoRA Parameters:
parser.add_argument("--lora_r", type=float, default=32,
                    help="Dimensionality of low-rank matrices used for \delta W.")
parser.add_argument("--lora_alpha", type=float, default=64,
                    help="Weight used on update \lora_alpha * \deltaW.")
parser.add_argument("--lora_dropout", type=float, default=0.1,
                    help="Dropout parameter for LoRA adapter layers.")
                    
parser.add_argument("--debug", default=False, action="store_true")
args = parser.parse_args()

# each unit should be at most 128 tokens
MAX_LENGTH = 128 + 32
MODEL_NAME = "mistralai/Mistral-7B-v0.3"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    model_max_lenght=MAX_LENGTH,
    padding_side="left", # left padding solves the following problem: the quick brown <PAD> --> prediction uses PAD token 
    add_eos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

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
    dataset = load_from_disk(os.path.join(args.dirname, "train_clean_and_joined"))
    if args.debug:
        dataset = dataset.select(range(1_000))

    if args.prompt == "none":
        keys = ["units"]
    else:
        keys = [key for key in dataset[0].keys() if f"{args.prompt}_generations" in key]
    
    text = []
    for i in tqdm(range(len(dataset))):
        for key in keys:
            N = int(args.units_perc_per_sample * len(dataset[i][key]))
            text.extend(random.sample(dataset[i][key], k=N))
            
    text = text[:args.num_samples]
    train_size = int(len(text) * 0.9)
    train_text = text[:train_size]
    test_text = text[train_size:]
    
    train_samples = [tokenize_and_pad_to_fixed_length(sample) for sample in tqdm(train_text)]
    test_samples = [tokenize_and_pad_to_fixed_length(sample) for sample in tqdm(test_text)]

    return train_samples, test_samples

def main():
    if args.prompt != "all":
        NotImplementedError("--prompt \"all\" is not implemented yet.")

    if args.debug:
        print(colored("Running in DEBUG mode", "green"))
        
    for key, value in vars(args).items():
        print(colored(f"\t{key} = {value}", "yellow"))

    train_samples, test_samples = load_dataset()
    
    print(colored(f"len(train_samples)={len(train_samples)}", "blue"))
    print(colored(f"len(test_samples)={len(test_samples)}", "blue"))
    
    run_name = f"Mistral-7B-v0.3-QLoRA-prompt={args.prompt}-perc={args.units_perc_per_sample}-ns={args.num_samples}-debug={args.debug}"
    output_dir = os.path.join("/scratch1/yubnub/changepoint/output", run_name)
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        report_to="mlflow",
        run_name=run_name,
        output_dir=output_dir,
        per_device_train_batch_size=128,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        bf16=True,
        learning_rate=2e-5,
        lr_scheduler_type="constant",
        max_steps=1000,
        save_steps=100,
        logging_steps=100,
        warmup_steps=10,
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