
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

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
        
    @torch.no_grad()
    def predict(
        self, 
        text: list[str],
    ):
        if isinstance(text, str):
            text = [text]
            
        inputs = self.tokenizer(
            text,
            max_length = 512,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt",
        )
        inputs = {k:v.to(self.model.device) for k, v in inputs.items()}
        
        out = self.model(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
        )
        sequence_mixture_preds = self.sequence_mixture_cls(out["pooler_output"])
        token_mixture_preds = []
        for i in range(len(text)):
            hidden_state = out["last_hidden_state"][i, :]
            end_token = torch.where(inputs["input_ids"][i] == 2)[0][0]
            hidden_state = out["last_hidden_state"][i, 1:end_token]
            token_mixture_preds.append(self.token_mixture_cls(hidden_state))
        return sequence_mixture_preds, token_mixture_preds
