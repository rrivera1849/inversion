
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

def load_luar_model_and_tokenizer(HF_id: str = "rrivera1849/LUAR-MUD"):
    luar = AutoModel.from_pretrained(HF_id, trust_remote_code=True)
    luar.eval()
    # RRS - Avoid weird library issue when loading tokenizer
    # luar_tok = AutoTokenizer.from_pretrained(HF_id, trust_remote_code=True)
    luar_tok = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-distilroberta-base-v1")
    return luar, luar_tok

@torch.no_grad()
def get_luar_author_embeddings(
    text: list[str],
    luar: AutoModel,
    luar_tok: AutoTokenizer,
):
    # output: tensor of shape (1, 512)
    assert isinstance(text, list)
    inputs = luar_tok(
        text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(luar.device)
    inputs["input_ids"] = inputs["input_ids"].view(1, len(text), 512)
    inputs["attention_mask"] = inputs["attention_mask"].view(1, len(text), 512)
    outputs = luar(**inputs)
    outputs = F.normalize(outputs, p=2, dim=-1)
    return outputs

@torch.no_grad()
def get_luar_instance_embeddings(
    text: list[str],
    luar: AutoModel,
    luar_tok: AutoTokenizer,
    batch_size: int = 32,
    progress_bar: bool = False,
):
    # output: tensor of shape (len(text), 512)
    all_outputs = []
    if progress_bar:
        iter = tqdm(range(0, len(text), batch_size))
    else:
        iter = range(0, len(text), batch_size)

    for i in iter:
        batch = text[i:i+batch_size]
        inputs = luar_tok(
            batch,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(luar.device)
        inputs["input_ids"] = inputs["input_ids"].view(len(batch), 1, 512)
        inputs["attention_mask"] = inputs["attention_mask"].view(len(batch), 1, 512)
        outputs = luar(**inputs)
        all_outputs.append(outputs)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_outputs = F.normalize(all_outputs, p=2, dim=-1)
    return all_outputs
