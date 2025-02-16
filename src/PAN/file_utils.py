
import json
import os

from tqdm import tqdm

# paths to the PAN23 datasets TODO
PAN23_train_paths = {
    "easy": "/data1/foobar/changepoint/pan23/pan23-multi-author-analysis-dataset1/pan23-multi-author-analysis-dataset1-train",
    "medium": "/data1/foobar/changepoint/pan23/pan23-multi-author-analysis-dataset2/pan23-multi-author-analysis-dataset2-train",
    "hard": "/data1/foobar/changepoint/pan23/pan23-multi-author-analysis-dataset3/pan23-multi-author-analysis-dataset3-train",
}

PAN23_validation_paths = {
    "easy": "/data1/foobar/changepoint/pan23/pan23-multi-author-analysis-dataset1/pan23-multi-author-analysis-dataset1-validation",
    "medium": "/data1/foobar/changepoint/pan23/pan23-multi-author-analysis-dataset2/pan23-multi-author-analysis-dataset2-validation",
    "hard": "/data1/foobar/changepoint/pan23/pan23-multi-author-analysis-dataset3/pan23-multi-author-analysis-dataset3-validation",
}

def read_PAN_dataset(path):
    """Reads a PAN dataset from a given path and returns a list of samples.
    """
    N = len(os.listdir(path)) // 2
    samples = []
    for i in tqdm(range(1, N+1)):
        # newline="" as specified by PAN organizers
        problem = open(os.path.join(path, f"problem-{i}.txt"), "r", newline="").readlines()
        truth = json.load(open(os.path.join(path, f"truth-problem-{i}.json"), "r"))
        sample = {"text": [paragraph.strip() for paragraph in problem]}
        sample.update(truth)
        samples.append(sample)
    return samples
