
import json
import random
import sys
from multiprocessing import Pool
from typing import Union

from termcolor import colored
from transformers import AutoTokenizer
from transformers import logging as hf_logging
from tqdm.auto import tqdm

random.seed(43)
hf_logging.set_verbosity_error()
MUD_PATH = "/data1/yubnub/data/raw_all/data.jsonl.crud.filtered"
TOKENIZER = AutoTokenizer.from_pretrained("roberta-large")
CHUNKSIZE = 10_000
DEBUG = False

def count_num_tokens(text: str) -> int:
    return len(TOKENIZER.tokenize(text))

def filter(data: dict) -> Union[dict, None]:
    min_num_tokens = 64
    max_num_tokens = 128 

    to_delete = [key for key in data.keys() if key not in ["author_id", "syms", "action_type"]]
    for key in to_delete:
        del data[key]

    data["syms"] = list(set(data["syms"]))
    indices_to_keep = [
        i for i, text in enumerate(data["syms"]) if min_num_tokens <= count_num_tokens(text) <= max_num_tokens
    ]

    N = 10
    if len(indices_to_keep) < N:
        return None
    
    indices_to_keep = random.sample(indices_to_keep, N)
    data["syms"] = [data["syms"][i] for i in indices_to_keep]
    data["action_type"] = [data["action_type"][i] for i in indices_to_keep]
    
    return data

def main():
    
    fout = open("/data1/yubnub/changepoint/MUD_inverse/raw/data.jsonl.crud.filtered", "w+")
    with open(MUD_PATH, "r") as fin:
        done = False
        pool = Pool(40)
        num_lines = 0
        while True:
            lines = []
            for _ in range(CHUNKSIZE):
                line = fin.readline()
                if not line:
                    done = True
                    break
                lines.append(line)
            
            if len(lines) > 0:
                filtered_data = list(tqdm(pool.imap(filter, [json.loads(line) for line in lines]), total=len(lines)))
                filtered_data = [data for data in filtered_data if data is not None]
                for data in filtered_data:
                    fout.write(json.dumps(data) + "\n")
            
            num_lines += len(lines)
            print(colored("Processed {} lines".format(num_lines), "green"))
            
            if done:
                break
            
            if DEBUG:
                import pdb; pdb.set_trace()
                break

    fout.close()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())