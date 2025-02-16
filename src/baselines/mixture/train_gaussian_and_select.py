
import json
import os
import sys

import editdistance
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
tqdm.pandas()

DATA_PATH = "/data1/yubnub/changepoint/MUD_inverse/data/"

def train_gmm(
    human_scores: list[int],
    machine_scores: list[int],
) -> tuple[GaussianMixture, GaussianMixture]:
    gmm_human = GaussianMixture()
    gmm_human.fit(np.array(human_scores).reshape(-1, 1))

    gmm_machine = GaussianMixture()
    gmm_machine.fit(np.array(machine_scores).reshape(-1, 1))

    return gmm_human, gmm_machine

def main():
    domain_name = sys.argv[1]
    valid_fname = sys.argv[2]
    test_fname = sys.argv[3]

    fname = os.path.join(DATA_PATH, domain_name, "inverse_output", valid_fname)
    human_scores, machine_scores = [], []
    with open(os.path.join(DATA_PATH, fname)) as fin:
        for line in fin:
            data = json.loads(line)
            edit_distance = editdistance.eval(data["rephrase"], data["inverse"][0])
            if data["is_machine"] or data["is_human_paraphrase"]:
                machine_scores.append(edit_distance)
            elif not data["is_human_paraphrase"] and not data["is_machine"]:
                human_scores.append(edit_distance)

    gmm_human, gmm_machine = train_gmm(human_scores, machine_scores)
    print(np.mean(machine_scores), np.mean(human_scores))
    fname = os.path.join(DATA_PATH, domain_name, "inverse_output", test_fname)
    MTD_df = pd.read_json(os.path.join(DATA_PATH, fname), lines=True)

    def process(row):
        # majority voting: check whether it is more probable that it's an inversion of a human-text:
        edit_distances = [editdistance.eval(row["rephrase"], inv) for inv in row["inverse"]]
        edit_distances = np.array(edit_distances).reshape(-1, 1)
        vote = (gmm_human.score_samples(edit_distances) > gmm_machine.score_samples(edit_distances)).sum()
        vote = vote > (len(row["inverse"]) // 2)
        if vote:
            row["use_inverse"] = False
        else:
            row["use_inverse"] = True

        scores_human = gmm_human.score_samples(edit_distances).flatten().tolist()
        scores_human = np.mean(scores_human)
        scores_machine = gmm_machine.score_samples(edit_distances).flatten().tolist()
        scores_machine = np.mean(scores_machine)
        row["gmm_scores_human"] = scores_human
        row["gmm_scores_machine"] = scores_machine
        
        return row

    MTD_df = MTD_df.progress_apply(process, axis=1)
    savename = fname + "-edit-detector"
    MTD_df.to_json(
        os.path.join(DATA_PATH, savename),
        lines=True,
        orient="records",
    )

    return 0

main()
