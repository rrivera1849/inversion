"""TODO
1. HuggingFace accelerate for multi-processing.
2. Continuations (full machine) for training.
3. Logging to file.
4. More metrics than just accuracy.
5. Sequence prediction dependence on token prediction.
"""

import os
import sys
from argparse import ArgumentParser

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
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
        writer.add_scalar("train_loss/total", loss.item(), n_train_iter)
        writer.add_scalar("train_loss/sequence", output["sequence_loss"].item(), n_train_iter)
        writer.add_scalar("train_loss/token_mixture", output["token_mixture_loss"].item(), n_train_iter)
        writer.add_scalar("train_accuracy/sequence", output["sequence_accuracy"].item(), n_train_iter)
        writer.add_scalar("train_accuracy/token_mixture", output["token_mixture_accuracy"].item(), n_train_iter)
        n_train_iter += 1

        pbar.set_description(f"Training Loss: {loss.item():.4f} | Seq Acc: {output['sequence_accuracy'].item():.4f} | Tok Acc: {output['token_mixture_accuracy'].item():.4f}")
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
    average_sequence_accuracy = 0.
    average_token_mixture_accuracy = 0.
    
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
        average_sequence_accuracy += output["sequence_accuracy"].item()
        average_token_mixture_accuracy += output["token_mixture_accuracy"].item()
        pbar.set_description(f"Validation Loss: {output['loss'].item():.4f} | Seq Acc: {output['sequence_accuracy'].item():.4f} | Tok Acc: {output['token_mixture_accuracy'].item():.4f}")
        pbar.update(1)
        
    average_loss /= num_validation_batches
    average_sequence_loss /= num_validation_batches
    average_token_mixture_loss /= num_validation_batches
    average_sequence_accuracy /= num_validation_batches
    average_token_mixture_accuracy /= num_validation_batches
    
    if write_tensorboard_logs:
        writer.add_scalar("validation_loss/total", average_loss, epoch)
        writer.add_scalar("validation_loss/sequence", average_sequence_loss, epoch)
        writer.add_scalar("validation_loss/token_mixture", average_token_mixture_loss, epoch)
        writer.add_scalar("validation_accuracy/sequence", average_sequence_accuracy, epoch)
        writer.add_scalar("validation_accuracy/token_mixture", average_token_mixture_accuracy, epoch)
    pbar.close()
    
    # the same but nice colors:
    print(colored(f"{'Validation Epoch':<20} {epoch+1}", "blue"))
    print(colored(f"{'\tLoss':<20} {average_loss:.4f}", "cyan"))
    print(colored(f"{'\tSeq Loss':<20} {average_sequence_loss:.4f}", "cyan"))
    print(colored(f"{'\tTok Loss':<20} {average_token_mixture_loss:.4f}", "cyan"))
    print(colored(f"{'\tSeq Acc':<20} {average_sequence_accuracy:.4f}", "cyan"))
    print(colored(f"{'\tTok Acc':<20} {average_token_mixture_accuracy:.4f}", "cyan"))
    
    return average_loss

def main():
    if args.evalute_only and args.checkpoint_path is None:
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

        if "loss" not in checkpoint:
            print(colored("Checkpoint does not contain loss information, runing validation to compute loss.", "yellow"))

            best_loss = validation_step(
                validation_dataset,
                model,
                writer,
                device,
                None,
                write_tensorboard_logs=False
            )
        else:
            best_loss = checkpoint["loss"]

    if args.evaluate_only:
        average_val_loss = validation_step(
            validation_dataset,
            model,
            writer,
            device,
            None
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