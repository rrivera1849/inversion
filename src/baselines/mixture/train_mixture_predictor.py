"""TODO
1. HuggingFace accelerate for multi-processing.
2. Continuations (full machine) for training.
3. Logging to file.
4. More metrics than just accuracy.
"""

import os
import sys
from argparse import ArgumentParser
from shutil import copy

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score, f1_score

parser = ArgumentParser()
parser.add_argument("--experiment_id", type=str, default="debug",
                    help="Experiment ID, used for logging and saving checkpoints.")
parser.add_argument("--num_epochs", type=int, default=100,
                    help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=16,
                    help="Batch size to use during training.")
parser.add_argument("--seed", type=int, default=43)
parser.add_argument("--debug", default=False, action="store_true")
args = parser.parse_args()

class MixturePredictor(nn.Module):
    """Simple wrapper around RoBERTa Large for predicting mixtures.
    
    Tasks:
    1. Is it a mixture or not?
    2. For each token, which mixture does it come from?
    """
    def __init__(self):
        super().__init__()
        HF = "roberta-large"
        self.model = AutoModel.from_pretrained(HF)
        self.tokenizer = AutoTokenizer.from_pretrained(HF)

        self.hdim = self.model.config.hidden_size
        self.sequence_mixture_cls = nn.Linear(self.hdim, 2)
        self.token_mixture_cls = nn.Linear(self.hdim, 2)

        self.loss = nn.CrossEntropyLoss()

    def forward(
        self, 
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        batch_size = inputs["input_ids"].size(0)
        
        out = self.model(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
        )

        # Sequence Level Predictions:
        sequence_mixture_preds = self.sequence_mixture_cls(out["pooler_output"])
        sequence_loss = self.loss(sequence_mixture_preds, inputs["label"])
        sequence_accuracy = (sequence_mixture_preds.argmax(dim=1) == inputs["label"]).float().mean()

        # Token Level Predictions:
        token_mixture_loss = 0.
        token_mixture_accuracy = 0.
        for i, tagger_label in enumerate(inputs["tagger_labels"]):
            token_mixture_preds = self.token_mixture_cls(out["last_hidden_state"][i, 1:len(tagger_label)+1])
            token_mixture_loss += self.loss(token_mixture_preds, tagger_label)
            token_mixture_accuracy += (token_mixture_preds.argmax(dim=1) == tagger_label).float().mean()
        token_mixture_loss /= batch_size
        token_mixture_accuracy /= batch_size

        return {
            "loss": sequence_loss + token_mixture_loss,
            "sequence_loss": sequence_loss,
            "token_mixture_loss": token_mixture_loss,
            "sequence_accuracy": sequence_accuracy,
            "token_mixture_accuracy": token_mixture_accuracy
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
    path: str
) -> None:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }, path)

def main():
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
    
    num_epochs = 1 if args.debug else args.num_epochs
    n_train_iter = 0
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}/{num_epochs}")
        
        num_training_batches = len(range(0, len(dataset), args.batch_size))
        num_training_batches = 10 if args.debug else num_training_batches
        pbar = tqdm(total=num_training_batches, unit="batch")
        model.train()
        for batch in batch_generator(dataset, args.batch_size, device, num_batches=10 if args.debug else None):
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

        with torch.no_grad():
            num_validation_batches = len(range(0, len(validation_dataset), args.batch_size))
            num_validation_batches = 10 if args.debug else num_validation_batches
            pbar = tqdm(total=num_validation_batches, desc="Validation", unit="batch")
            model.eval()

            average_loss = 0.
            average_sequence_loss = 0.
            average_token_mixture_loss = 0.
            average_sequence_accuracy = 0.
            average_token_mixture_accuracy = 0.
            
            for batch in batch_generator(validation_dataset, args.batch_size, device, num_batches=10 if args.debug else None):
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
            
            writer.add_scalar("validation_loss/total", average_loss, epoch)
            writer.add_scalar("validation_loss/sequence", average_sequence_loss, epoch)
            writer.add_scalar("validation_loss/token_mixture", average_token_mixture_loss, epoch)
            writer.add_scalar("validation_accuracy/sequence", average_sequence_accuracy, epoch)
            writer.add_scalar("validation_accuracy/token_mixture", average_token_mixture_accuracy, epoch)
            pbar.close()

        if average_loss < best_loss:
            best_loss = average_loss
            checkpoint_fname = os.path.join(checkpoints_dir, "best.pth")
            save_checkpoint(model, optimizer, epoch, checkpoint_fname)
            print(colored(f"Saving best model at epoch {epoch+1} with loss {best_loss:.4f}", "green"))

        checkpoint_fname = os.path.join(checkpoints_dir, f"checkpoint_{epoch:04d}.pth")
        save_checkpoint(model, optimizer, epoch, checkpoint_fname)

    return 0

if __name__ == "__main__":
    sys.exit(main())