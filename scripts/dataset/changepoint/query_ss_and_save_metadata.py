
import json
import os
import sys
from argparse import ArgumentParser
from time import sleep
from typing import Dict, List, Union

import requests
from datasets import load_from_disk
from termcolor import colored
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--dirname", type=str,
                    default="/data1/yubnub/changepoint/s2orc_changepoint/unit_128",
                    help="Directory where the dataset is stored.")
parser.add_argument("--split", type=str, default="test",
                    help="Dataset split to clean and join the generations for.")
parser.add_argument("--debug", default=False, action="store_true",
                    help="Debug mode, runs in a small subset of data.")
args = parser.parse_args()

def get_paper_data(paper_id, tries=0, max_tries=10):
    """https://www.semanticscholar.org/product/api/tutorial#searching-and-retrieving-paper-details
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/CorpusID:" + paper_id

    headers = {'x-api-key': '9CL8w1KzzO56a5OakVSrI1SWQ8JaZ6WK7h2zLws0'}
    paper_data_query_params = {"fields": "title,year,authors"}

    # Send the API request and store the response in a variable
    response = requests.get(url, params=paper_data_query_params, headers=headers)
    if response.status_code == 200:
        return response.json()
    elif tries < max_tries:
        sleep(2)
        return get_paper_data(paper_id, tries=tries+1)
    else:
        return None

def main():
    dataset = load_from_disk(os.path.join(args.dirname, args.split))
    N = 100 if args.debug else len(dataset)

    savename = os.path.join(args.dirname, f"metadata_ss_{args.split}.jsonl")
    if os.path.isfile(savename):
        try:
            fout = open(savename, "a+")
            last_index = json.loads(open(savename, "r").readlines()[-1])["index"] + 1
        except:
            os.remove(savename)
            fout = open(savename, "w+")
            last_index = 0
    else:
        fout = open(savename, "w+")
        last_index = 0

    print(colored("last_index=", "green"), last_index)
    for i in tqdm(range(last_index, last_index+N)):
        corpus_id = dataset[i]["id"]
        paper_metadata = get_paper_data(corpus_id)

        record = {}
        record["index"] = i
        record["corpus_id"] = corpus_id
        record["metadata"] = paper_metadata
        fout.write(json.dumps(record)); fout.write('\n')

    return 0


if __name__ == "__main__":
    sys.exit(main())