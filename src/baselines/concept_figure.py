
import os
import pandas as pd

import evaluate

BLEU = evaluate.load("bleu")

DATA_DIR = "/data1/yubnub/changepoint/MUD_inverse/data/abstracts/inverse_output"
fname = "test_final_none_3000_temperature=0.7_top_p=0.9.jsonl.vllm_n=100"
path = os.path.join(DATA_DIR, fname)

df = pd.read_json(path, lines=True)

for index, row in df.iterrows():
    if not row["is_machine"]:
        continue
    
    scores = [
        BLEU.compute(predictions=[row["inverse"][i]], references=[row["unit"]])["bleu"]
        for i in range(len(row["inverse"]))
    ]
    max_score = max(scores)
    index = scores.index(max_score)
    
    print("BLEU: {}".format(max_score))
    print("UNIT")
    print(row["unit"])
    print("REPHRASE")
    print(row["rephrase"])
    print("INVERSE")
    print(row["inverse"][index])
    input("Continue?")