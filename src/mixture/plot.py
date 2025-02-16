
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import umap.plot

from metric_utils import calculate_metrics
from embedding_utils import *

def targeted_similarity_plot(key="inverse_max"):
    untargeted_fname = "./metrics/new/data.jsonl.filtered.cleaned_kmeans_100/plagiarism/crud/none_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=100"
    untargeted = json.loads(open(untargeted_fname).read())
    similarities = untargeted[key]["similarities"]
    labels = untargeted[key]["labels"]
    untargeted_true = [sim for sim, label in zip(similarities, labels) if label == 1]
    untargeted_other = [sim for sim, label in zip(similarities, labels) if label == 0]
    
    filenames = [
        "none_targetted=examples_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=5.targetted_mode=author_num_examples=1",
        "none_targetted=examples_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=5.targetted_mode=author_num_examples=2",
        "none_targetted=examples_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=5.targetted_mode=author_num_examples=3",
        "none_targetted=examples_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=5.targetted_mode=author",
        "none_targetted_6400_temperature=0.7_top_p=0.9.jsonln=5.targetted_mode=author",
    ]
    
    Y_target = []
    Y_non_target = []
    Y_target_10 = []
    Y_non_target_10 = []
    Y_target_90 = []
    Y_non_target_90 = []
    
    Y_target.append(np.median(untargeted_true))
    Y_non_target.append(np.median(untargeted_other))
    Y_target_10.append(np.percentile(untargeted_true, 10))
    Y_non_target_10.append(np.percentile(untargeted_other, 10))
    Y_target_90.append(np.percentile(untargeted_true, 90))
    Y_non_target_90.append(np.percentile(untargeted_other, 90))
    
    for fname in filenames:
        path = os.path.join("metrics/new/data.jsonl.filtered.cleaned_kmeans_100/author/crud", fname)
        data = json.loads(open(path).read())
        similarities = data[key]["similarities"]
        labels = data[key]["labels"]

        target_sims = [sim for sim, label in zip(similarities, labels) if label == 1]
        non_target_sims = [sim for sim, label in zip(similarities, labels) if label == 0]
        
        Y_target.append(np.median(target_sims))
        Y_non_target.append(np.median(non_target_sims))
        Y_target_10.append(np.percentile(target_sims, 10))
        Y_target_90.append(np.percentile(target_sims, 90))
        Y_non_target_10.append(np.percentile(non_target_sims, 10))
        Y_non_target_90.append(np.percentile(non_target_sims, 90))

    Y_target = np.array(Y_target)
    Y_non_target = np.array(Y_non_target)
    Y_target_10 = np.array(Y_target_10)
    Y_non_target_10 = np.array(Y_non_target_10)
    Y_target_90 = np.array(Y_target_90)
    Y_non_target_90 = np.array(Y_non_target_90)
        
    N = len(Y_target)
    _ = plt.figure()
    plt.errorbar(range(N), Y_target, yerr=[Y_target - Y_target_10, Y_target_90 - Y_target], fmt="o", label="True Match")
    plt.errorbar(range(N), Y_non_target, yerr=[Y_non_target - Y_non_target_10, Y_non_target_90 - Y_non_target], fmt="o", label="Other")
    plt.xticks(range(N), ["Untargeted (Single)", "Targeted (N=1)", "Targeted (N=2)", "Targeted (N=3)", "Targeted (N=All)", "Targeted (Style Emb.)"])
    plt.xticks(rotation=45)
    plt.xlabel("Inversion Method")
    plt.ylabel("Median Style Similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./plots/targetted_author_similarity_{key}.pdf")
    plt.close()
    
def umap_plot():
    base_path = "/data1/foobar/changepoint/MUD_inverse/data/data.jsonl.filtered.cleaned_kmeans_100/inverse_output"
    untargeted = os.path.join(base_path, "none_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=100")
    targeted = os.path.join(base_path, "none_targetted_6400_temperature=0.7_top_p=0.9.jsonln=5.targetted_mode=author")

    data = pd.read_json(untargeted, lines=True)
    data_targeted = pd.read_json(targeted, lines=True)
    data_targeted = data_targeted[data_targeted.author_id_x == data_targeted.author_id_y]
    data_targeted = data_targeted[["rephrase_x", "inverse"]]
    data_targeted.inverse = data_targeted.inverse.apply(lambda x: x[0])
    rephrase = data_targeted.rephrase_x.tolist()[:100]
    inverse_targeted = data_targeted.inverse.tolist()[:100]
    data = data[data.rephrase.isin(rephrase)]
    human = data.unit.tolist()
    inverse_untargeted = data.inverse.apply(lambda x: x[0]).tolist()
    
    luar, luar_tok = load_luar_model_and_tokenizer()
    luar.to("cuda")
    
    human_emb = get_luar_instance_embeddings(human, luar, luar_tok, progress_bar=True)
    untargeted_emb = get_luar_instance_embeddings(inverse_untargeted, luar, luar_tok, progress_bar=True)
    rephrase_emb = get_luar_instance_embeddings(rephrase, luar, luar_tok, progress_bar=True)
    # target_emb = get_luar_instance_embeddings(inverse_targeted, luar, luar_tok, progress_bar=True)

    # all_embeddings = torch.cat([human_emb, untargeted_emb, rephrase_emb, target_emb], dim=0)
    all_embeddings = torch.cat([human_emb, untargeted_emb, rephrase_emb], dim=0)
    all_embeddings = all_embeddings.cpu().detach().numpy()
    all_labels = ["Human"] * len(human_emb) + ["Inverse (Untargeted)"] * len(untargeted_emb) + ["Paraphrase"] * len(rephrase_emb) # + ["Inverse (Targeted)"] * len(target_emb)
    all_labels = np.array(all_labels)
    
    mapping = umap.UMAP(metric="cosine").fit_transform(all_embeddings)
    
    fig, ax = plt.subplots()
    ax.set_axis_off()
    # Scatterplot with colourblind friendly colors and markers:
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    markers = ["x", "o", "+", "s"]
    for i, label in enumerate(np.unique(all_labels)):
        idx = all_labels == label
        plt.scatter(mapping[idx, 0], mapping[idx, 1], c=colors[i], marker=markers[i], label=label)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./plots/umap.pdf")
    plt.close()

os.makedirs("./plots", exist_ok=True)
# targeted_similarity_plot("inverse_all")
# targeted_similarity_plot("inverse_max")
# targeted_similarity_plot("inverse_expected")
targeted_similarity_plot("inverse_single")
# umap_plot()