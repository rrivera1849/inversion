
import pickle
import os
import random
import sys
from argparse import ArgumentParser
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import LoggerType, ProjectConfiguration, set_seed
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from termcolor import colored
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from model import MixturePredictor
from utils import get_levenshtein_tags

parser = ArgumentParser()
parser.add_argument("--dataset_dirname", type=str,
                    default="./datasets/all_roberta-large_250000_stratified/",
                    help="Directory containing the dataset files.")
parser.add_argument("--experiment_id", type=str, default="debug",
                    help="Experiment ID, used for logging and saving checkpoints.")
parser.add_argument("--num_epochs", type=int, default=10,
                    help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=16,
                    help="Batch size to use during training.")
parser.add_argument("--learning_rate", type=float, default=2e-5,
                    help="Learning rate to use during training.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                    help="Number of gradient accumulation steps.")
parser.add_argument("--perc", type=float, default=1.0,
                    help="Percentage of the training dataset to use.")
parser.add_argument("--checkpoint_path", type=str, default=None,
                    help="Path to a checkpoint to resume training from.")
parser.add_argument("--evaluate_only", default=False, action="store_true",
                    help="If set, only evaluate the model on the validation set.")
parser.add_argument("--seed", type=int, default=43)
parser.add_argument("--debug", default=False, action="store_true")
args = parser.parse_args()

METRICS = {
    "accuracy": accuracy_score,
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
}

def is_inverse_data(df):
    return "original" in df.columns and "generation" in df.columns

def is_author_data(df):
    return "author_id" in df.columns

def get_tagger_labels(generation: str, original: str, tokenizer_fn: callable):
    tags = get_levenshtein_tags(generation, original, tokenizer_fn)
    # 1 in case it's machine, 0 otherwise
    tagger_labels = [int(tag != "KEEP") for tag in tags]
    return tagger_labels

class JSONLDataset(Dataset):
    def __init__(self, path):
        self.dataset = pd.read_json(path, lines=True)
        if args.perc < 1.0:
            self.dataset = self.dataset.sample(frac=args.perc, random_state=args.seed)
        print("Total samples:", len(self.dataset))

        # In case we're reading from the inverse dataset:
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")

        self.is_inverse_data = is_inverse_data(self.dataset)
        self.is_author_data = is_author_data(self.dataset)

    def __len__(self):
        return len(self.dataset)
    
    def process_inverse_sample(self, sample):
        # In case we're reading from the inverse dataset:
        if random.random() < 0.5:
            text = sample["original"]
            sample["tagger_labels"] = [0] * len(self.tokenizer.tokenize(text))
            sample["label"] = 0
        else:
            text = sample["generation"]
        sample.update(self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=512, truncation=True))
        return sample

    def process_author_sample(self, sample):
        # In case we're reading from the author dataset:
        if random.random() < 0.5:
            text = sample["unit"]
            sample["tagger_labels"] = [0] * len(self.tokenizer.tokenize(text))
            sample["label"] = 0
        else:
            text = sample["rephrase"]
            sample["tagger_labels"] = get_tagger_labels(sample["rephrase"], sample["unit"], self.tokenizer.tokenize)
            sample["label"] = 1
        sample.update(self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=512, truncation=True))
        return sample

    def __getitem__(self, idx):
        sample =  self.dataset.iloc[idx].to_dict()

        if self.is_inverse_data:
            sample = self.process_inverse_sample(sample)
        elif self.is_author_data:
            sample = self.process_author_sample(sample)

        return sample
    
def get_dataloader(path, shuffle=True, max_samples=None):
    dataset = JSONLDataset(path)
    if max_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(max_samples))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )

def collate_fn(batch):
    return {
        "input_ids": torch.stack([torch.LongTensor(b["input_ids"]) for b in batch]),
        "attention_mask": torch.stack([torch.LongTensor(b["attention_mask"]) for b in batch]),
        "label": torch.concatenate([torch.LongTensor([b["label"]]) for b in batch], dim=0),
        "tagger_labels": [torch.LongTensor(b["tagger_labels"]) for b in batch],
    }
        
def compute_metrics(
    label: torch.Tensor, 
    tagger_labels: list[torch.Tensor], 
    sequence_mixture_preds: torch.Tensor, 
    token_mixture_preds: list[torch.Tensor],
):
    metric_results = {}
    label = label.cpu()
    tagger_labels = [tagger_label.cpu() for tagger_label in tagger_labels]
    for metric_name, metric_fn in METRICS.items():
        if metric_name != "accuracy":
            kwargs = {"zero_division": 0.}
        else:
            kwargs = {}

        # Sequence Level Metrics:
        sequence_metric = metric_fn(label, sequence_mixture_preds.argmax(axis=1), **kwargs)

        # Token Level Metrics:
        token_mixture_metric = 0.0
        for j, tagger_label in enumerate(tagger_labels):
            if label[j] == 0:
                # To avoid errors in the metric calculation, we invert the tagger label.
                tagger_label = abs(1 - tagger_label)
                pred = abs(1 - token_mixture_preds[j].argmax(axis=1))
            else:
                pred = token_mixture_preds[j].argmax(axis=1)

            metric = metric_fn(tagger_label, pred, **kwargs)
            token_mixture_metric += metric
        token_mixture_metric /= len(tagger_labels)
        metric_results[metric_name] = (sequence_metric, token_mixture_metric)

    return metric_results
    
def train_step(
    data_loader: DataLoader,
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    accelerator: Accelerator, 
    n_train_iter: int,
):
    num_training_batches = 50 if args.debug else len(data_loader)
    pbar = tqdm(total=num_training_batches, unit="batch", desc="Training", disable=not accelerator.is_local_main_process)
    model.train()
    for i, batch in enumerate(data_loader):
        with accelerator.accumulate(model):
            optimizer.zero_grad()

            if batch["input_ids"].size(1) == 1 and batch["attention_mask"].size(1) == 1:
                batch["input_ids"] = batch["input_ids"].squeeze(1)
                batch["attention_mask"] = batch["attention_mask"].squeeze(1)

            output = model(batch)
            loss = output["loss"]
            accelerator.backward(loss)
            optimizer.step()

        logging_dict = {
            "train/loss": loss.item(),
            "train/sequence_loss": output["sequence_loss"].item(),
            "train/token_mixture_loss": output["token_mixture_loss"].item(),
        }
        
        metric_results = compute_metrics(
            batch["label"],
            batch["tagger_labels"],
            output["sequence_mixture_preds"],
            output["token_mixture_preds"],
        )
        for (metric_name, (sequence_metric, token_mixture_metric)) in metric_results.items():
            logging_dict[f"train/sequence_{metric_name}"] = sequence_metric
            logging_dict[f"train/token_mixture_{metric_name}"] = token_mixture_metric

        accelerator.log(logging_dict, step=n_train_iter)

        n_train_iter += 1
        sequence_accuracy = metric_results["accuracy"][0]
        token_mixture_f1 = metric_results["f1"][1]
        pbar.set_description(f"Training Loss: {loss.item():.4f} | Seq Acc: {sequence_accuracy:.4f} | Tok F1: {token_mixture_f1:.4f}")
        pbar.update(1)

        if args.debug and i >= 49:
            break
        
    pbar.close()
    return n_train_iter

@torch.no_grad()
def validation_step(
    validation_dataloader: DataLoader, 
    model: nn.Module, 
    accelerator: Accelerator, 
    epoch: int,
    write_tensorboard_logs: bool = True,
):
    num_validation_batches = 50 if args.debug else len(validation_dataloader)
    pbar = tqdm(total=num_validation_batches, desc="Validation", unit="batch", disable=not accelerator.is_local_main_process)
    model.eval()

    average_loss = 0.
    average_sequence_loss = 0.
    average_token_mixture_loss = 0.

    metric_accumulators = {metric_name: [] for metric_name in METRICS.keys()}
    
    for i, batch in enumerate(validation_dataloader):
        if batch["input_ids"].size(1) == 1 and batch["attention_mask"].size(1) == 1:
            batch["input_ids"] = batch["input_ids"].squeeze(1)
            batch["attention_mask"] = batch["attention_mask"].squeeze(1)
            
        output = model(batch)

        average_loss += output["loss"].item()
        average_sequence_loss += output["sequence_loss"].item()
        average_token_mixture_loss += output["token_mixture_loss"].item()

        metric_results = compute_metrics(
            batch["label"],
            batch["tagger_labels"],
            output["sequence_mixture_preds"],
            output["token_mixture_preds"],
        )
        for metric_name, (sequence_metric, token_mixture_metric) in metric_results.items():
            metric_accumulators[metric_name].append((sequence_metric, token_mixture_metric))

        sequence_accuracy = metric_results["accuracy"][0]
        token_mixture_f1 = metric_results["f1"][1]
        pbar.set_description(f"Validation Loss: {output['loss'].item():.4f} | Seq Acc: {sequence_accuracy:.4f} | Tok F1: {token_mixture_f1:.4f}")
        pbar.update(1)

        if args.debug and i >= 49:
            break

    pbar.close()

    average_loss /= num_validation_batches
    average_sequence_loss /= num_validation_batches
    average_token_mixture_loss /= num_validation_batches
    
    if write_tensorboard_logs:
        logging_dict = {
            "valid/loss": average_loss,
            "valid/sequence_loss": average_sequence_loss,
            "valid/token_mixture_loss": average_token_mixture_loss,
        }
        for metric_name, metric_values in metric_accumulators.items():
            sequence_metric_values, token_mixture_metric_values = zip(*metric_values)
            logging_dict[f"valid/sequence_{metric_name}"] = np.mean(sequence_metric_values)
            logging_dict[f"valid/token_mixture_{metric_name}"] = np.mean(token_mixture_metric_values)
        accelerator.log(logging_dict, step=epoch)

    if accelerator.is_main_process:
        print(colored(f"{'Validation Epoch':<20} {epoch+1}", "blue"))
        print(colored(f"{'Loss':<20} {average_loss:.4f}", "cyan"))
        print(colored(f"{'Seq Loss':<20} {average_sequence_loss:.4f}", "cyan"))
        print(colored(f"{'Tok Loss':<20} {average_token_mixture_loss:.4f}", "cyan"))
        for metric_name, metric_values in metric_accumulators.items():
            sequence_metric_values, token_mixture_metric_values = zip(*metric_values)
            print(colored(f"{'Seq '+metric_name:<20} {np.mean(sequence_metric_values):.4f}", "cyan"))
            print(colored(f"{'Tok '+metric_name:<20} {np.mean(token_mixture_metric_values):.4f}", "cyan"))

    return average_loss

def main():
    if args.evaluate_only and args.checkpoint_path is None:
        raise ValueError("Must provide a checkpoint path to evaluate the model.")

    if args.perc < 1.0:
        experiment_id = f"{args.experiment_id}_perc={args.perc}"
    else:
        experiment_id = args.experiment_id

    experiment_dir = os.path.join(f"./outputs/{experiment_id}")
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    model = MixturePredictor()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    project_config = ProjectConfiguration(
        project_dir=experiment_dir,
        automatic_checkpoint_naming=True,
    )
    accelerator = Accelerator(
        log_with=LoggerType.TENSORBOARD,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=project_config,
    )
    accelerator.init_trackers("logs", vars(args))
    model, optimizer = accelerator.prepare(model, optimizer)
    
    best_loss = float("inf")
    if args.checkpoint_path is not None:
        accelerator.load_state(args.checkpoint_path)
        print(colored(f"Loaded checkpoint from {args.checkpoint_path}", "green"))
        epoch = int(os.path.basename(args.checkpoint_path).split("_")[1])
        metadata_fname = list(glob(os.path.join(checkpoints_dir, f"metadata_epoch={epoch:03d}*")))[0]
        checkpoint = pickle.load(open(metadata_fname, "rb"))
        best_loss = checkpoint["loss"]

    if args.evaluate_only and accelerator.is_main_process:
        test_dataloader = get_dataloader(
            os.path.join(args.dataset_dirname, "test.jsonl"),
            shuffle=False,
        )
        test_dataloader = accelerator.prepare(test_dataloader)
        _ = validation_step(
            test_dataloader,
            model,
            accelerator,
            -1,
            write_tensorboard_logs=False,
        )
        return 0
    
    train_dataloader = get_dataloader(
        os.path.join(args.dataset_dirname, "train.jsonl"),
        shuffle=True,
    )
    valid_dataloader = get_dataloader(
        os.path.join(args.dataset_dirname, "valid.jsonl"),
        shuffle=False,
    )
    train_dataloader, valid_dataloader = accelerator.prepare(train_dataloader, valid_dataloader)
    
    num_epochs = 2 if args.debug else args.num_epochs
    n_train_iter = 0
    for epoch in range(num_epochs):
        if accelerator.is_main_process:
            print(f"Epoch: {epoch+1}/{num_epochs}")
        
        n_train_iter = train_step(
            train_dataloader,
            model,
            optimizer,
            accelerator,
            n_train_iter
        )
        average_val_loss = validation_step(
            valid_dataloader,
            model,
            accelerator,
            epoch
        )
        average_val_loss = accelerator.gather(torch.FloatTensor([average_val_loss]).to(accelerator.device)).mean().item()

        if accelerator.is_main_process:
            metadata = {
                "epoch": epoch,
                "loss": average_val_loss,
            }
            if average_val_loss < best_loss:
                best_loss = average_val_loss
                metadata_fname = f"metadata_epoch={epoch:03d}_best.pkl"
                print(colored(f"Saving best model at epoch {epoch+1} with loss {best_loss:.4f}", "green"))
            else:
                metadata_fname = f"metadata_epoch={epoch:03d}.pkl"
            pickle.dump(metadata, open(os.path.join(checkpoints_dir, metadata_fname), "wb"))

        accelerator.save_state()

    accelerator.end_training()
    return 0

if __name__ == "__main__":
    set_seed(args.seed)
    assert args.perc > 0. and args.perc <= 1.
    sys.exit(main())