
import os
import sys
import json
from glob import glob

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from termcolor import colored
from tqdm import tqdm

from model import MixturePredictor

def load_model():
    model = MixturePredictor()
    state_path = "./outputs/roberta-large_stratified/checkpoints/checkpoint_9/"

    accelerator = Accelerator()
    model = accelerator.prepare(model)
    model.eval()
    accelerator.load_state(state_path)
    return model

DATA_PATH = "/data1/foobar/data/iur_dataset/author_100.politics/"
model = load_model()

# for fname in glob(os.path.join(DATA_PATH, "*mistral*")):
for fname in ["train", "valid", "test"]:
    fname = os.path.join(DATA_PATH, f"{fname}.jsonl")
    fname = fname[:-1] if fname.endswith("/") else fname
    fname_out = fname + ".token_mixture_preds"
    print(colored(f"input file: {fname}", "green"))
    print(colored(f"output file: {fname_out}", "green"))
    print()

    dataset = [json.loads(line) for line in open(fname, "r").readlines()]
    batch_size = 256

    fout = open(fname_out, "w+")
    with torch.inference_mode():
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i+batch_size]

            batch_text = [x["syms"] for x in batch]
            sequence_preds, token_mixture_preds = model.predict(batch_text)
            sequence_preds = sequence_preds.argmax(1).detach().cpu().tolist()
            token_mixture_preds = [F.softmax(pred, dim=-1) for pred in token_mixture_preds]
            token_mixture_preds = [pred.detach().cpu().tolist() for pred in token_mixture_preds]

            for j, x in enumerate(batch):
                record = x
                record["is_machine"] = sequence_preds[j]
                record["token_mixture_preds"] = token_mixture_preds[j]
                fout.write(json.dumps(record) + "\n")

    fout.close()