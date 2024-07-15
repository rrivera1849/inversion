
import math
from typing import List, Union

import torch
import torch.nn.functional as F

def mean_pooling(model_output, attention_mask):
    """Mean pooling of the token embeddings, ignoring tokens where no attention is paid.
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_uar_embedding(sample: Union[List[str], str], model, tokenizer, device):
    """Returns a **single** UAR embedding, either for a single sample or a list of samples.
    """
    if not isinstance(sample, list):
        sample = [sample]

    all_input_ids = []
    all_attention_masks = []
    
    for paragraph in sample:
        tok = tokenizer(
            paragraph,
            truncation=False,
            padding=True,
            return_tensors="pt"
        )

        # UAR's backbone can handle up to 512 tokens
        # Here we're padding the sample to the nearest multiple of 512:
        _, NT = tok["input_ids"].size()
        nearest = 512 * int(math.ceil(NT / 512))
        tok["input_ids"] = F.pad(tok["input_ids"], (1, nearest - NT - 1), value=tokenizer.pad_token_id)
        tok["attention_mask"] = F.pad(tok["attention_mask"], (1, nearest - NT - 1), value=0)

        # Reshape into (batch_size=1, history_size=N, num_tokens=512)
        tok["input_ids"] = tok["input_ids"].reshape(1, -1, 512).to(device)
        tok["attention_mask"] = tok["attention_mask"].reshape(1, -1, 512).to(device)

        all_input_ids.append(tok["input_ids"])
        all_attention_masks.append(tok["attention_mask"])
        
    tok = {
        "input_ids": torch.cat(all_input_ids, dim=1),
        "attention_mask": torch.cat(all_attention_masks, dim=1)
    }
    with torch.inference_mode():
        out = model(**tok)
        out = F.normalize(out, p=2.0)
    return out

def get_cisr_embedding(sample: Union[List[str], str], model, tokenizer, device):
    """Returns a **single** CISR embedding, either for a single sample or a list of samples.
    """
    if not isinstance(sample, list):
        sample = [sample]
        
    tok = tokenizer(
        sample,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    tok = {k: v.to(device) for k, v in tok.items()}
    
    with torch.inference_mode():
        out = model(**tok)
        out = mean_pooling(out, tok["attention_mask"])
        out = F.normalize(out, p=2.0)
    
    out = out.mean(dim=0, keepdim=True)
    return out

def get_embedding(sample: Union[List[str], str], model, tokenizer, device, model_name):
    """Helper function to get the embedding of a sample using the specified model.
    """
    if model_name == "uar":
        return get_uar_embedding(sample, model, tokenizer, device)
    elif model_name == "cisr":
        return get_cisr_embedding(sample, model, tokenizer, device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def PPL(text: str, model, tokenizer, device):
    """Returns the perplexity of the given text under the LLM provided.
    """
    tok = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    tok = {k: v.to(device) for k, v in tok.items()}
    
    with torch.inference_mode():
        out = model(**tok, labels=tok["input_ids"].clone())
        NLL = out.loss
        PPL = torch.exp(NLL)
        
    return PPL.item()