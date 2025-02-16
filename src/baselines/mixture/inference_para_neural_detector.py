
import os
import sys

import pandas as pd
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm

from model import MixturePredictor

DATA_PATH = "/data1/foobar/changepoint/MUD_inverse/data/"

def main():
    experiment_id = sys.argv[1]
    checkpoint_name = sys.argv[2]
    domain_name = sys.argv[3]
    test_fname = sys.argv[4]
    
    accelerator = Accelerator()
    mpmodel = MixturePredictor()
    mpmodel = accelerator.prepare(mpmodel)
    accelerator.load_state(os.path.join("./outputs", experiment_id, "checkpoints", checkpoint_name))

    fname = os.path.join(DATA_PATH, domain_name, "inverse_output", test_fname)
    test_df = pd.read_json(fname, lines=True)
    texts = test_df.rephrase.tolist()
    
    batch_size = 32
    use_inverse = []
    probs = []

    for batch_idx in tqdm(range(0, len(texts), batch_size)):
        batch = texts[batch_idx:batch_idx+batch_size]
        sequence_logits, _ = mpmodel.predict(batch)
        sequence_probs = F.softmax(sequence_logits, dim=-1)
        preds = sequence_probs.argmax(dim=1).tolist()
        use_inverse.extend([True if p == 1 else False for p in preds])
        probs.extend(sequence_probs[:, 1].tolist())

    test_df["use_inverse"] = use_inverse
    test_df["probs"] = probs
    savename = fname + "-neural-pred-" + experiment_id
    test_df.to_json(savename, lines=True, orient="records")

    return 0

main()