
import json
import os
import sys

import editdistance
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "/data1/foobar/changepoint/MUD_inverse/data"

def main():
    dataset_name = "data.jsonl.filtered.respond_reddit.cleaned"
    filename = "valid_with_all_none_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=1"
    full_path = os.path.join(DATA_PATH, dataset_name, "inverse_output", filename)

    edit_distances_units = []
    edit_distances_rephrases = []
    edit_distances_human_paraphrase = []
    with open(full_path, "r") as fin:
        for line in fin:
            d = json.loads(line)
            EDs = [editdistance.eval(inv, d["rephrase"]) for inv in d["inverse"]]
            if d["is_machine"]:
                edit_distances_rephrases.extend(EDs)
            elif d["is_human_paraphrase"]:
                edit_distances_human_paraphrase.extend(EDs)
            else:
                edit_distances_units.extend(EDs)
    
    _ = plt.figure()
    plt.hist(edit_distances_units, label="Human Text", alpha=0.5)
    # plt.hist(edit_distances_rephrases+edit_distances_human_paraphrase, label="Paraphrases of Human and Machine Text", alpha=0.5)
    plt.hist(edit_distances_rephrases, label="Paraphrases of Machine Text", alpha=0.5)
    plt.hist(edit_distances_human_paraphrase, label="Paraphrases of Human Text", alpha=0.5)
    plt.legend()
    plt.title("Edit Distance")
    plt.savefig("./editdist.pdf")
    plt.close()

    print("Edit Distance for Units: {:.2f}".format(np.mean(edit_distances_units)))
    # print("Edit Distance for Machine Paraphrases: {:.2f}".format(np.mean(edit_distances_rephrases)))            
    # print("Edit Distance for Human Paraphrases: {:.2f}".format(np.mean(edit_distances_human_paraphrase)))            
    print("Edit Distance for Paraphrases: {:.2f}".format(np.mean(edit_distances_rephrases+edit_distances_human_paraphrase)))            

    return 0

if __name__ == "__main__":
    sys.exit(main())