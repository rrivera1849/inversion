
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

class ScorePredictor(pl.LightningModule):
    def __init__(
        self, 
        emb_dim: int = 512, 
        num_layers: int = 3,
        dropout_p: float = 0.1
    ):
        super(ScorePredictor, self).__init__()
        self.num_layers = num_layers
        
        layers = []
        for _ in range(num_layers - 1):
            block = nn.Sequential(
                nn.Linear(emb_dim, emb_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p)
            )
            layers.extend(block)
        layers.append(nn.Linear(emb_dim, 10, bias=True))
        self.score_predictor = nn.Sequential(*layers)
        
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.score_predictor(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)
    
    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y.argmax(dim=1)).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y.argmax(dim=1)).float().mean()
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

def create_binned_labels(labels, num_bins=10):
    bins = np.linspace(0, 1., num_bins)
    binned_labels = np.digitize(labels, bins)
    return binned_labels - 1

def read_best_score_dataset(dataset_path):
    embeddings, labels = [], []
    with open(dataset_path, "r") as f:
        for line in tqdm(f.readlines()):
            sample = json.loads(line)
            emb = torch.from_numpy(np.array(sample["embeddings"])).float().transpose(0, 1)
            emb = torch.mean(emb, dim=1)
            
            embeddings.append(emb)
            labels.append(sample["f1_score"])

    embeddings = torch.cat(embeddings, dim=0)
    labels = create_binned_labels(labels)
    labels = torch.LongTensor(labels)
    labels = F.one_hot(labels, num_classes=10).float()
    return embeddings, labels

def main():
    train_path = "/data1/foobar/changepoint/pan23/pan23-multi-author-analysis-dataset3/train_best_performance_scores.jsonl"
    validation_path = "/data1/foobar/changepoint/pan23/pan23-multi-author-analysis-dataset3/validation_best_performance_scores.jsonl"
    
    # I want to bin the labels into 10 bins and make this a classification task
    train_embeddings, train_labels = read_best_score_dataset(train_path)
    
    # validation_embeddings, validation_labels = read_best_score_dataset(validation_path)

    train_embeddings = train_embeddings[:int(0.8*len(train_embeddings))]
    train_labels = train_labels[:int(0.8*len(train_labels))]
    validation_embeddings = train_embeddings[int(0.8*len(train_embeddings)):]
    validation_labels = train_labels[int(0.8*len(train_labels)):]

    train_dataset = TensorDataset(train_embeddings, train_labels)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=2048, 
        shuffle=True, 
        num_workers=4,
    )
    validation_dataset = TensorDataset(validation_embeddings, validation_labels)
    validation_dataloader = DataLoader(
        validation_dataset, 
        batch_size=1024, 
        shuffle=False, 
        num_workers=4,
    )
    
    model = ScorePredictor()
    trainer = pl.Trainer(
        max_epochs=200,
        devices=1,
        accelerator="gpu",
        enable_checkpointing=True,
        # accumulate_grad_batches=10,
        callbacks=[pl.callbacks.StochasticWeightAveraging(swa_lrs=[1e-2])]
    )
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)
    # TODO to add validation data loader

    # # get numbers on validation set
    # best_checkpoint = trainer.checkpoint_callback.best_model_path
    # model = ScorePredictor.load_from_checkpoint(best_checkpoint)
    # model.eval()
    # out = model(validation_embeddings.to(model.device))
    # import pdb; pdb.set_trace()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())