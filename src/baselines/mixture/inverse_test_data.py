
import random; random.seed(43)
import sys

import pandas as pd
from datasets import load_from_disk

N = 100
TEST_DATASET_PATH = "/data1/foobar/changepoint/s2orc_changepoint/unit_128/test_clean_and_joined"

def main():
    dataset = load_from_disk(TEST_DATASET_PATH)

    test_data = []
    for i in range(N):
        sample = dataset[i]
        units = sample["units"]
        generation_keys = [key for key in sample.keys() if "prompt=rephrase_generations" in key]
        changepoint_indices_keys = [key for key in sample.keys() if "prompt=rephrase_changepoint_indices" in key]
        
        for gkey, cpkey in zip(generation_keys, changepoint_indices_keys):
            model_name = gkey.split("_prompt")[0]
            generations = sample[gkey]
            changepoint_indices = sample[cpkey]
        
            index = random.choice(range(len(changepoint_indices)))
            unit = units[changepoint_indices[index]]
            gen = generations[index]
            test_data.append((sample["id"], unit, gen, model_name))

    df = pd.DataFrame(data={
        "id": [data[0] for data in test_data],
        "unit": [data[1] for data in test_data],
        "generation": [data[2] for data in test_data],
        "model_name": [data[3] for data in test_data],
    })
    df.to_json("./test_data/s2orc.jsonl", orient="records", lines=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())