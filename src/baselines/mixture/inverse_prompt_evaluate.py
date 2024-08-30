
import os
import random
random.seed(43)

import evaluate
import pandas as pd

rouge = evaluate.load('rouge')
metrics = {}
it = 0
for filename in os.listdir("./prompting_data/"):
    if "inverse" not in filename and "rephrase" not in filename:
        continue
    
    print("Evaluating", filename)
    filename = os.path.join("./prompting_data/", filename)
    df = pd.read_json(filename, lines=True)

    if "inverse" in filename:
        candidates = df.inverse.tolist()
        references = df.unit.tolist()
    else:
        candidates = df.rephrase.tolist()
        references = df.unit.tolist()

    results = rouge.compute(predictions=candidates, references=references)
    name = os.path.basename(filename).split(".")[0]
    metrics[name] = {}
    metrics[name]["rouge1"] = results["rouge1"]
    metrics[name]["rouge2"] = results["rouge2"]
    metrics[name]["rougeL"] = results["rougeL"]

    if it == 0:
        random.shuffle(candidates)
        metrics["random"] = {}
        results = rouge.compute(predictions=candidates, references=references)
        metrics["random"]["rouge1"] = results["rouge1"]
        metrics["random"]["rouge2"] = results["rouge2"]
        metrics["random"]["rougeL"] = results["rougeL"]
        
    it += 1
    
df = pd.DataFrame.from_dict(metrics, orient="index")
print(df.to_markdown())
df.to_json("inverse_prompt_results.json", orient="index")