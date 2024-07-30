
import os
import sys
from functools import partial

import pandas as pd
from datasets import Dataset
from utils import split_into_units

if len(sys.argv) > 1:
    num_max_tokens = int(sys.argv[1])
else:
    num_max_tokens = 128

path = "/data1/yubnub/changepoint/RAID_rephrase/train_human.jsonl"
dataset = pd.read_json(path, lines=True)
dataset = Dataset.from_pandas(dataset)
split_fn = partial(
    split_into_units,
    num_max_tokens=num_max_tokens,
    remove_title=False,
    calculate_metadata=False,
    choose_changepoint_indices=False,
    text_key="generation",
)
dataset = dataset.map(split_fn, num_proc=40)
savename = os.path.join(os.path.dirname(path), f"train_human_unit_{num_max_tokens}")
dataset.save_to_disk(savename)