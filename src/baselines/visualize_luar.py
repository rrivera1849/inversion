
import math
import os
import sys
from argparse import ArgumentParser
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import umap
import umap.plot
from termcolor import colored
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from utils import load_s2orc_MTD_data

parser = ArgumentParser()
parser.add_argument("--dirname", type=str,
                    default="/data1/foobar/changepoint/s2orc_changepoint/unit_128",
                    help="Directory where the dataset is stored.")
args = parser.parse_args()

@torch.no_grad()
def get_uar_embedding(
    samples: list[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    batch_size: int = 512,
):
    device = model.device
        
    embeddings = []
    for i in tqdm(range(0, len(samples), batch_size)):
        batch = samples[i:i + batch_size]
        tok = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        tok = {k:v.unsqueeze(1) for k,v in tok.items()}
        out = model(**tok)
        out = F.normalize(out, p=2.0)
        embeddings.append(out.cpu())
        
    embeddings = torch.cat(embeddings, dim=0).numpy()
    return embeddings


def main():
    os.makedirs("LUAR_viz", exist_ok=True)
    
    texts, model_names, prompt_names = load_s2orc_MTD_data(args.dirname, debug=True, debug_N=100)
    for i, prompt in enumerate(prompt_names):
        if prompt is None:
            prompt_names[i] = "human"
        if model_names[i] is None:
            model_names[i] = "human"
        if "Llama-3" in model_names[i]:
            model_names[i] = "Llama-3"
        if "Mistral-7B" in model_names[i]:
            model_names[i] = "Mistral-7B"
        if "Phi-3" in model_names[i]:
            model_names[i] = "Phi-3"
        
    print(colored("Dataset Statistics: ", "green"))
    for k, v in Counter(prompt_names).items():
        print(colored(f"\t{k}: {v}", "yellow"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)

    embeddings = get_uar_embedding(texts, model, tokenizer)

    print("UMAP...")
    mapper = umap.UMAP(metric="cosine").fit(embeddings)

    _ = plt.figure()
    umap.plot.points(mapper, labels=np.array(prompt_names))
    plt.savefig("./LUAR_viz/embeddings_label=prompt.png")
    plt.close()
    
    _ = plt.figure()
    umap.plot.points(mapper, labels=np.array(model_names))
    plt.savefig("./LUAR_viz/embeddings_label=model.png")
    plt.close()
    
    joint_labels = [f"{model_names[i]}_{prompt_names[i]}" for i in range(len(model_names))]
    _ = plt.figure()
    umap.plot.points(mapper, labels=np.array(joint_labels))
    plt.savefig("./LUAR_viz/embeddings_label=model_prompt.png")
    plt.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())