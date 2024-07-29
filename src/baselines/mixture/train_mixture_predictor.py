"""TODO
1. HuggingFace accelerate for multi-processing.
2. Continuations (full machine) for training.
3. Logging to file.
4. Sequence prediction dependence on token prediction.
5. Dataset should have LLM column and be split into validation and test beforehand.
"""

import os
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import MixturePredictor

parser = ArgumentParser()
parser.add_argument("--experiment_id", type=str, default="debug",
                    help="Experiment ID, used for logging and saving checkpoints.")
parser.add_argument("--num_epochs", type=int, default=25,
                    help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=16,
                    help="Batch size to use during training.")
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

def collate_fn(batch):
    return {
        "input_ids": torch.stack([torch.LongTensor(b["input_ids"]) for b in batch]),
        "attention_mask": torch.stack([torch.LongTensor(b["attention_mask"]) for b in batch]),
        "label": torch.concatenate([torch.LongTensor([b["label"]]) for b in batch], dim=0),
        "tagger_labels": [torch.LongTensor(b["tagger_labels"]) for b in batch],
    }

def to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, list):
            batch[k] = [vv.to(device) for vv in v]
        else:
            batch[k] = v.to(device)
    return batch

def batch_generator(
    dataset: pd.DataFrame,
    batch_size: int,
    device: torch.device,
    num_batches: int = None
):
    dataset = dataset.sample(frac=1., random_state=args.seed).reset_index(drop=True)
    ibatch = 0
    for i in range(0, len(dataset), batch_size):
        batch = dataset.iloc[i:i+batch_size]
        batch = [batch.iloc[j].to_dict() for j in range(len(batch))]
        batch = collate_fn(batch)
        batch = to_device(batch, device)
        yield batch

        ibatch += 1
        if num_batches is not None and ibatch > num_batches:
            break
   
def save_checkpoint(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    epoch: int,
    loss: float,
    path: str
) -> None:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }, path)
    
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
    dataset: pd.DataFrame,
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    writer: SummaryWriter, 
    device: torch.device, 
    n_train_iter: int,
):
    num_training_batches = len(range(0, len(dataset), args.batch_size))
    num_training_batches = 10 if args.debug else num_training_batches
    pbar = tqdm(total=num_training_batches, unit="batch")
    model.train()
    for batch in batch_generator(
        dataset, 
        args.batch_size, 
        device, 
        num_batches=10 if args.debug else None
    ):
        output = model(batch)

        loss = output["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("train/loss", loss.item(), n_train_iter)
        writer.add_scalar("train/sequence_loss", output["sequence_loss"].item(), n_train_iter)
        writer.add_scalar("train/token_mixture_loss", output["token_mixture_loss"].item(), n_train_iter)

        metric_results = compute_metrics(
            batch["label"],
            batch["tagger_labels"],
            output["sequence_mixture_preds"],
            output["token_mixture_preds"],
        )
        for (metric_name, (sequence_metric, token_mixture_metric)) in metric_results.items():
            writer.add_scalar(f"train/sequence_{metric_name}", sequence_metric, n_train_iter)
            writer.add_scalar(f"train/token_mixture_{metric_name}", token_mixture_metric, n_train_iter)

        n_train_iter += 1

        sequence_accuracy = metric_results["accuracy"][0]
        token_mixture_f1 = metric_results["f1"][1]
        pbar.set_description(f"Training Loss: {loss.item():.4f} | Seq Acc: {sequence_accuracy:.4f} | Tok F1: {token_mixture_f1:.4f}")
        pbar.update(1)
    pbar.close()
    
    return n_train_iter

@torch.no_grad()
def validation_step(
    validation_dataset: pd.DataFrame, 
    model: nn.Module, 
    writer: SummaryWriter, 
    device: torch.device, 
    epoch: int,
    write_tensorboard_logs: bool = True,
):
    num_validation_batches = len(range(0, len(validation_dataset), args.batch_size))
    num_validation_batches = 10 if args.debug else num_validation_batches
    pbar = tqdm(total=num_validation_batches, desc="Validation", unit="batch")
    model.eval()

    average_loss = 0.
    average_sequence_loss = 0.
    average_token_mixture_loss = 0.

    metric_accumulators = {metric_name: [] for metric_name in METRICS.keys()}
    
    for batch in batch_generator(
        validation_dataset, 
        args.batch_size, 
        device, 
        num_batches=10 if args.debug else None
    ):
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

    average_loss /= num_validation_batches
    average_sequence_loss /= num_validation_batches
    average_token_mixture_loss /= num_validation_batches
    
    if write_tensorboard_logs:
        writer.add_scalar("validation_loss/total", average_loss, epoch)
        writer.add_scalar("validation_loss/sequence", average_sequence_loss, epoch)
        writer.add_scalar("validation_loss/token_mixture", average_token_mixture_loss, epoch)

        for metric_name, metric_values in metric_accumulators.items():
            sequence_metric_values, token_mixture_metric_values = zip(*metric_values)
            writer.add_scalar(f"validation/sequence_{metric_name}", np.mean(sequence_metric_values), epoch)
            writer.add_scalar(f"validation/token_mixture_{metric_name}", np.mean(token_mixture_metric_values), epoch)

    pbar.close()
    
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

    experiment_dir = os.path.join(f"./outputs/{args.experiment_id}")
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    writer = SummaryWriter(experiment_dir)

    dataset = pd.read_json("./mixture_dataset.jsonl", lines=True)
    dataset = dataset.sample(frac=1., random_state=args.seed).reset_index(drop=True)
    validation_dataset = dataset.iloc[:len(dataset)//10]
    dataset = dataset.iloc[len(dataset)//10:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MixturePredictor()
    model.to(device)

    best_loss = float("inf")
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(colored(f"Loaded checkpoint from {args.checkpoint_path}", "green"))

        if "loss" not in checkpoint and not args.evaluate_only:
            print(colored("Checkpoint does not contain loss information, runing validation to compute loss.", "yellow"))

            best_loss = validation_step(
                validation_dataset,
                model,
                writer,
                device,
                -1,
                write_tensorboard_logs=False
            )
        elif "loss" in checkpoint:
            best_loss = checkpoint["loss"]

    if args.evaluate_only:
        average_val_loss = validation_step(
            validation_dataset,
            model,
            writer,
            device,
            -1,
            write_tensorboard_logs=False,
        )
        return 0
    
    num_epochs = 1 if args.debug else args.num_epochs
    n_train_iter = 0
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}/{num_epochs}")
        
        n_train_iter = train_step(
            dataset,
            model,
            optimizer,
            writer,
            device,
            n_train_iter
        )
        average_val_loss = validation_step(
            validation_dataset,
            model,
            writer,
            device,
            epoch
        )

        if average_val_loss < best_loss:
            best_loss = average_val_loss
            checkpoint_fname = os.path.join(checkpoints_dir, "best.pth")
            save_checkpoint(model, optimizer, epoch, best_loss, checkpoint_fname)
            print(colored(f"Saving best model at epoch {epoch+1} with loss {best_loss:.4f}", "green"))

        checkpoint_fname = os.path.join(checkpoints_dir, f"checkpoint_{epoch:04d}.pth")
        save_checkpoint(model, optimizer, epoch, average_val_loss, checkpoint_fname)

    return 0

if __name__ == "__main__":
    sys.exit(main())