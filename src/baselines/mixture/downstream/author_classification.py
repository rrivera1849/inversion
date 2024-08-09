
import json
import os
import sys
from argparse import ArgumentParser

import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import LoggerType, ProjectConfiguration, set_seed
from torch.utils.data import Dataset, DataLoader
from termcolor import colored
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

sys.path.append("../")

parser = ArgumentParser()
parser.add_argument("--train_suffix", type=str, default="")
parser.add_argument("--test_suffix", type=str, default="")
parser.add_argument("--num_epoch", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--token_mixture_multiplier", type=float, default=1.0)
args = parser.parse_args()

DATA_PATH = "/data1/yubnub/data/iur_dataset/author_100.politics"

def mean_pooling(model_output, attention_mask, weights=None):
    token_embeddings = model_output.last_hidden_state
    if weights is not None:
        token_embeddings = token_embeddings * weights.unsqueeze(-1).expand(token_embeddings.size())
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    return sum_embeddings / sum_mask

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("roberta-base", add_pooling_layer=False)
        self.classifier = nn.Linear(self.model.config.hidden_size, 100)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(
        self, 
        inputs: dict
    ) -> dict:
        
        out = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        if "token_mixture_preds" in inputs:
            token_mixture_preds = inputs["token_mixture_preds"]
            out = mean_pooling(out, inputs["attention_mask"], weights=token_mixture_preds)
        else:
            out = mean_pooling(out, inputs["attention_mask"])
        out = self.classifier(out)
        loss = self.loss(out, inputs["label"].squeeze())

        accuracy = (out.argmax(1) == inputs["label"].squeeze()).float()
        assert accuracy.size(0) == inputs["input_ids"].size(0)
        accuracy = accuracy.mean()
        
        return {
            "accuracy": accuracy,
            "logits": out,
            "loss": loss,
        }

class JSONLDataset(Dataset):
    def __init__(self, fname):
        print(colored(f"Reading data from {fname}", "green"))
        self.df = pd.read_json(fname, lines=True)
        # NOTE - this only works because the same author_ids are used in all datasets
        author_ids = sorted(self.df["author_id"].unique())
        author2idx = {author_id: idx for idx, author_id in enumerate(author_ids)}
        self.df["author_id"] = self.df["author_id"].apply(lambda x: author2idx[x])
        self.df.rename(columns={"author_id": "label", "syms": "text"}, inplace=True)
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inputs = self.tokenizer(
            row["text"],
            padding="max_length",
            max_length=512,
            return_tensors="pt", 
            truncation=True
        )
        inputs["input_ids"] = inputs["input_ids"].squeeze()
        inputs["attention_mask"] = inputs["attention_mask"].squeeze()
        inputs["label"] = torch.LongTensor([row["label"]])

        if "token_mixture_preds" in row:
            token_mixture_preds = row["token_mixture_preds"]
            token_mixture_preds = [1. + pred[0] * args.token_mixture_multiplier for pred in token_mixture_preds]
            token_mixture_preds = [0.] + token_mixture_preds
            token_mixture_preds += [0.] * (512 - len(token_mixture_preds))
            token_mixture_preds = torch.FloatTensor(token_mixture_preds)
            inputs["token_mixture_preds"] = token_mixture_preds

        return inputs
    
def get_dataloader(fname, batch_size=16, shuffle=True):
    dataset = JSONLDataset(fname)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle
    )

def gather_single_value(value: float, accelerator: Accelerator) -> float:
    return accelerator.gather(torch.FloatTensor([value]).to(accelerator.device)).mean().item()

def get_dataset_distribution_name(name: str):
    if name.endswith(".mistral") or name.endswith(".mistral.token_mixture_preds"):
        return "mistral"
    elif name.endswith(".mistral.mixed") or name.endswith(".mistral.mixed.token_mixture_preds"):
        return "mixed"
    elif name.endswith("mistral.oracle"):
        return "mistral_oracle"
    elif name.endswith("mistral.mixed.oracle"):
        return "mixed_oracle"
    elif name.endswith("mistral.uniform"):
        return "mistral_uniform"
    elif name.endswith("mistral.mixed.uniform"):
        return "mixed_uniform"
    elif name.endswith(".oracle"):
        return "human_oracle"
    elif name.endswith(".uniform"):
        return "human_uniform"
    elif name.endswith(".token_mixture_preds") or name == "":
        return "human" 
    else:
        raise ValueError(f"Invalid dataset distribution name: {name}")

def main():
    os.makedirs("./outputs/author_classification_100", exist_ok=True)

    if args.train_suffix == "" and args.test_suffix.endswith("token_mixture_preds"):
        raise ValueError(f"Invalid combination of train_suffix and test_suffix: {args.train_suffix}, {args.test_suffix}")
    train_fname = os.path.join(DATA_PATH, f"train.jsonl{args.train_suffix}")
    valid_fname = os.path.join(DATA_PATH, f"valid.jsonl{args.train_suffix}")
    test_fname = os.path.join(DATA_PATH, f"test.jsonl{args.test_suffix}")
    print(colored(f"train_fname: {train_fname}", "yellow"))
    print(colored(f"valid_fname: {valid_fname}", "yellow"))
    print(colored(f"test_fname: {test_fname}", "yellow"))

    num_epoch = args.num_epoch
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    
    fr = get_dataset_distribution_name(args.train_suffix)
    to = get_dataset_distribution_name(args.test_suffix)
    run_id = f"{fr}_{to}"
    if "token_mixture_preds" in args.train_suffix or "token_mixture_preds" in args.test_suffix:
        run_id += f"_weight={args.token_mixture_multiplier}"

    run_name = f"{run_id}_{num_epoch}_{batch_size}_{learning_rate}"
    print(colored(f"run_name: {run_name}", "cyan"))
    experiment_dir = f"./outputs/author_classification_100/{run_name}"

    project_config = ProjectConfiguration(
        project_dir=experiment_dir,
        automatic_checkpoint_naming=True,
        total_limit=1,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=8,
        log_with=LoggerType.TENSORBOARD,
        project_config=project_config,
    )

    if accelerator.is_main_process:
        print(colored(f"num_epoch: {num_epoch}", "yellow"))
        print(colored(f"batch_size: {batch_size}", "yellow"))
        print(colored(f"learning_rate: {learning_rate}", "yellow"))

    hparams = {
        "num_epoch": num_epoch,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    accelerator.init_trackers("logs", hparams)

    model = Classifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_dataloader = get_dataloader(
        os.path.join(DATA_PATH, train_fname),
        batch_size=batch_size,
        shuffle=True,
    )
    valid_dataloader = get_dataloader(
        os.path.join(DATA_PATH, valid_fname),
        batch_size=batch_size,
        shuffle=False,
    )

    model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader
    )
    
    best_accuracy = 0
    best_model = None
    for epoch in range(num_epoch):
        if accelerator.is_main_process:
            print(colored(f"Epoch {epoch}", "green"))

        pbar = tqdm(total=len(train_dataloader), desc="Training", disable=not accelerator.is_main_process)
        model.train()
        average_training_loss = 0.
        for batch in train_dataloader:
            with accelerator.accumulate():
                optimizer.zero_grad()
                outputs = model(batch)
                accelerator.backward(outputs["loss"])
                optimizer.step()
                average_training_loss += outputs["loss"].item()
            pbar.set_description(f"Training: loss: {outputs['loss'].item():.2f}), accuracy: {outputs['accuracy'].item():.2f}")
            pbar.update(1)
        average_training_loss /= len(train_dataloader)
        accelerator.log({"train/loss": average_training_loss}, step=epoch)

        pbar = tqdm(total=len(valid_dataloader), desc="Validation", disable=not accelerator.is_main_process)
        model.eval()
        average_accuracy = 0
        for batch in valid_dataloader:
            with torch.no_grad():
                outputs = model(batch)
                average_accuracy += outputs["accuracy"].item()
            pbar.update(1)
        average_accuracy /= len(valid_dataloader)
        accelerator.log({"valid/accuracy": average_accuracy}, step=epoch)

        average_accuracy = gather_single_value(average_accuracy, accelerator)
        if average_accuracy > best_accuracy:
            if accelerator.is_main_process:
                print(colored(f"New best accuracy: {average_accuracy}", "green"))
            best_accuracy = average_accuracy
            best_model = model.state_dict()
            accelerator.save_state()
    
    model.load_state_dict(best_model)
    model.eval()
    test_dataloader = get_dataloader(
        os.path.join(DATA_PATH, test_fname),
        batch_size=batch_size,
        shuffle=False,
    )
    test_dataloader = accelerator.prepare(test_dataloader)
    average_accuracy = 0
    for batch in test_dataloader:
        with torch.no_grad():
            outputs = model(batch)
            average_accuracy += outputs["accuracy"].item()
    average_accuracy /= len(test_dataloader)
    average_accuracy = gather_single_value(average_accuracy, accelerator)
    
    if accelerator.is_main_process:
        print(colored(f"Average accuracy on test set: {average_accuracy}", "green"))
    with open(os.path.join(experiment_dir, "results.json"), "w") as f:
        json.dump({"test_accuracy": average_accuracy}, f, indent=4)
            
    return 0

if __name__ == "__main__":
    set_seed(43)
    sys.exit(main())