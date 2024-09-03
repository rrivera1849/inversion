"""Use PEFT to train an LLM on human text, continuations, rephrases, or rephrases with context.

Good Tutorial on QLoRA for Fine-Tuning LLMs: https://mlflow.org/docs/latest/llms/transformers/tutorials/fine-tuning/transformers-peft.html
"""

# I want Train on Rephrases -> Test on Human
# I want Train on Human -> Test on Human 
# I want Train on 50% Rephrases + 50% Human -> Test on Human

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
MODEL_NAME = "mistralai/Mistral-7B-v0.3"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
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

def group_texts(examples, block_size=128):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def get_prompt_type(key: str) -> str:
    if "rephrase_with_context" in key:
        return "rephrase_with_context"
    elif "rephrase" in key:
        return "rephrase"
    elif "continuation" in key:
        return "continuation"
    else:
        return "none"

def load_dataset() -> Union[list[Dict[str, list[int]]]]:
    dataset = load_from_disk(os.path.join(args.dirname, "train_clean_and_joined"))
    if args.debug:
        dataset = dataset.select(range(1_000))

    if args.prompt == "none":
        keys = ["units"]
    elif "+" in args.prompt:
        # Format: <PROMPT>-<Perc>+<PROMPT>-<Perc>
        prompt_type_1, prompt_type_2 = args.prompt.split("+")
        prompt_type_1, percentage_1 = prompt_type_1.split("-")
        prompt_type_2, percentage_2 = prompt_type_2.split("-")
        percentage_1 = float(percentage_1)
        percentage_2 = float(percentage_2)
        assert percentage_1 + percentage_2 == 1.0, "Percentages must sum to 1.0."
        keys = []
        keys.extend([key for key in dataset[0].keys() if f"{prompt_type_1}_generations" in key])
        keys.extend([key for key in dataset[0].keys() if f"{prompt_type_2}_generations" in key])
        if prompt_type_1 == "none" or prompt_type_2 == "none":
            keys.append("units")
    else:
        keys = [key for key in dataset[0].keys() if f"{args.prompt}_generations" in key]
    
    text = []
    prompt_types = []
    for i in tqdm(range(len(dataset))):
        for key in keys:
            N = int(args.units_perc_per_sample * len(dataset[i][key]))
            text.extend(random.sample(dataset[i][key], k=N))
            prompt_types.extend([get_prompt_type(key)] * N)
            
    unique_prompt_types = list(set(prompt_types))
    if len(unique_prompt_types) > 1:
        assert len(unique_prompt_types) == 2, "Only two prompt types are supported."
        sample_size = args.num_samples // 2
        sampled_text = []
        sampled_text.extend(random.sample([text for i, text in enumerate(text) if prompt_types[i] == unique_prompt_types[0]], k=sample_size))
        sampled_text.extend(random.sample([text for i, text in enumerate(text) if prompt_types[i] == unique_prompt_types[1]], k=sample_size))
        random.shuffle(sampled_text)
        text = sampled_text
    else:
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
    
    # run_name = f"Mistral-7B-v0.3-QLoRA-prompt={args.prompt}-perc={args.units_perc_per_sample}-ns={args.num_samples}-debug={args.debug}"
    # output_dir = os.path.join("/scratch1/yubnub/changepoint/output", run_name)
    # os.makedirs(output_dir, exist_ok=True)

    peft_model.config.use_cache = False
    
    data_loader = torch.utils.data.DataLoader(
        train_samples,
        batch_size=8,
        shuffle=True,
        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    for batch in data_loader:
        labels = batch.pop("labels")
        out = model(**batch)
        import pdb; pdb.set_trace()
        break

    # training_args = TrainingArguments(
    #     report_to="mlflow",
    #     run_name=run_name,
    #     output_dir=output_dir,
    #     per_device_train_batch_size=128,
    #     gradient_accumulation_steps=2,
    #     gradient_checkpointing=True,
    #     optim="paged_adamw_8bit",
    #     bf16=True,
    #     learning_rate=2e-5,
    #     lr_scheduler_type="constant",
    #     max_steps=3000,
    #     save_steps=200,
    #     logging_steps=100,
    #     warmup_steps=10,
    #     # https://discuss.huggingface.co/t/training-llama-with-lora-on-multiple-gpus-may-exist-bug/47005/3
    #     ddp_find_unused_parameters=False,
    # )

    # trainer = Trainer(
    #     model=peft_model,
    #     train_dataset=train_samples,
    #     data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    #     args=training_args,
    # )
    # # use_cache=True is incompatible with gradient checkpointing
    
    # trainer.train()
    
    # test_metrics = trainer.evaluate(test_samples)
    # print(test_metrics)

if __name__ == "__main__":
    sys.exit(main())