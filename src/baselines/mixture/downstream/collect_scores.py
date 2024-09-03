
import json
import os
import sys
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

def get_plot_name(dirname):
    if "oracle_weight" in dirname and "fixed_weight" in dirname:
        return "oracle"
    elif "uniform_weight" in dirname and "fixed_weight" in dirname:
        return "uniform"
    elif "method=learned_weight_uniform-weight" in dirname:
        return "learned_weight_uniform-weight"
    elif "method=fixed_weight_uniform-weight" in dirname:
        return "fixed_weight_uniform-weight"
    elif "method=learned_weight_no_bias_uniform-weight" in dirname:
        return "learned_weight_no_bias_uniform-weight"
    elif "method=learned_weight_no_bias" in dirname:
        return "learned_weight_no_bias"
    elif "method=learned_weight" in dirname:
        return "learned_weight"
    elif "method=fixed_weight" in dirname:
        return "fixed_weight"
    else:
        return dirname.split("_")[0]

def main():
    output_dir = "./outputs/author_classification_100"

    boxplot_scores = []
    boxplot_names = []
    for dirname in os.listdir(output_dir):
        if "human" not in dirname and not dirname == "human_20_16_5e-05":
            continue
        # if "human" not in dirname or "mixture_embeddings" in dirname:
            # continue
        # print(dirname)

        path = os.path.join(output_dir, dirname)
        filenames = glob(os.path.join(path, "*/*human*"))

        scores = []
        for fname in filenames:
            d = json.loads(open(fname, "r").read())
            scores.append(d["test_accuracy"])

        print(fname, scores[-1])

        # boxplot_scores.append(scores)
        # boxplot_names.append(get_plot_name(os.path.basename(path)))
    
    for name, scores in zip(boxplot_names, boxplot_scores):
        print(name + " {:.2f} ± {:.2f}".format(np.mean(scores), np.std(scores)))
        
    # _ = plt.figure()
    # plt.boxplot(boxplot_scores, labels=boxplot_names)
    # plt.legend()
    # plt.savefig("./scores.png")
    # plt.close()

    return 0

if __name__ == "__main__":
    sys.exit(main())