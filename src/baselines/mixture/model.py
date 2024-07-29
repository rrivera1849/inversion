
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
        is_train: bool = True,
    ) -> dict[str, float]:
        result = {}
        batch_size = inputs["input_ids"].size(0)
        
        out = self.model(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
        )
        
        # Sequence Level Predictions:
        sequence_mixture_preds = self.sequence_mixture_cls(out["pooler_output"])
        result["sequence_loss"] = self.loss(sequence_mixture_preds, inputs["label"])
        result["sequence_mixture_preds"] = sequence_mixture_preds.detach().cpu()

        # Token Level Predictions:
        result["token_mixture_loss"] = 0.
        result["token_mixture_preds"] = []
        for i, tagger_label in enumerate(inputs["tagger_labels"]):
            token_mixture_preds = self.token_mixture_cls(out["last_hidden_state"][i, 1:len(tagger_label)+1])
            result["token_mixture_loss"] += self.loss(token_mixture_preds, tagger_label)
            result["token_mixture_preds"].append(token_mixture_preds.detach().cpu())
        result["token_mixture_loss"] /= batch_size

        result["loss"] = result["sequence_loss"] + result["token_mixture_loss"]
        return result
        
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
