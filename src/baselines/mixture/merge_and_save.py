# Merges our trained PEFT model and saves it with .save_pretrained().
# Necessary before running inference with VLLM.
# https://github.com/huggingface/peft/issues/692

import os
import sys

import torch
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

peft_model_id = sys.argv[1]
new_model_id = sys.argv[1] + "_merged"

print("Loading: ", peft_model_id)
print("Saving to: ", new_model_id)

config = PeftConfig.from_pretrained(peft_model_id)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, peft_model_id)

merged_model = model.merge_and_unload()
merged_model.save_pretrained(
    new_model_id,
)
tokenizer.save_pretrained(
    new_model_id,
)