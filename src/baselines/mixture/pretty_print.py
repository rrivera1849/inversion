
import os
import json

import numpy as np
from termcolor import colored

from metric_utils import calculate_metrics

def table_1():
    # Untargetted Inversion Similarity & Basic Metrics
    
    fname = "./metrics/none_6400_temperature=0.7_top_p=0.9_luar_plagiarism.json"
    data_style = json.loads(open(fname).read())

    fname = "./metrics/none_6400_temperature=0.7_top_p=0.9_sbert_plagiarism.json"
    data_semantic = json.loads(open(fname).read())
    
    fname = "./metrics/none_6400_temperature=0.7_top_p=0.9_simple_untargeted.json"
    simple_metrics = json.loads(open(fname).read())

    rephrase_sim_target = np.mean(
        [sim for sim, label in zip(data_style["rephrase"]["similarities"], data_style["rephrase"]["labels"]) if label == 1]
    )
    rephrase_sim_non_target = np.mean(
        [sim for sim, label in zip(data_style["rephrase"]["similarities"], data_style["rephrase"]["labels"]) if label == 0]
    )
    rephrase_semantic_sim_target = np.mean(
        [sim for sim, label in zip(data_semantic["rephrase"]["similarities"], data_semantic["rephrase"]["labels"]) if label == 1]
    )
    rephrase_bleu = simple_metrics["bleu"]["rephrase"]

    print(f"\\bf Paraphrases & N/A & {rephrase_sim_target:.2f} & {rephrase_sim_non_target:.2f} & {rephrase_semantic_sim_target:.2f} & {rephrase_bleu:.2f} \\\\")

    order = ["single", "max", "expected", "all"]
    print("\\bf Inversion (Untargeted) & & & & & \\\\")
    for ord in order:
        key = f"inverse_{ord}"
        sim_target = np.mean(
            [sim for sim, label in zip(data_style[key]["similarities"], data_style[key]["labels"]) if label == 1]
        )
        sim_non_target = np.mean(
            [sim for sim, label in zip(data_style[key]["similarities"], data_style[key]["labels"]) if label == 0]
        )
        
        semantic_sim_target = np.mean(
            [sim for sim, label in zip(data_semantic[key]["similarities"], data_semantic[key]["labels"]) if label == 1]
        )
        
        if ord == "all":
            bleu = "-"
        else:
            bleu = simple_metrics["bleu"][f"inverse_{ord}"]
            bleu = f"{bleu:.2f}"
        
        if ord == "all":
            ord_print = "Aggregate"
        elif ord == "expected":
            ord_print = "Expectation"
        else:
            ord_print = ord.capitalize()
        print(f"& {ord_print} & {sim_target:.2f} & {sim_non_target:.2f} & {semantic_sim_target:.2f} & {bleu} \\\\")

    # TODO Semantic Similarity

def table_2():
    # Plagiarism Detection with Untargeted Inversion
    
    fname = "./metrics/none_6400_temperature=0.7_top_p=0.9_luar_plagiarism.json"
    data = json.loads(open(fname).read())

    rephrase_EER = data["rephrase"]["EER"]
    print(f"\\bf Paraphrases & {rephrase_EER:.2f} \\\\")
    data["rephrase"]["AUC"]
    
    keys = ["single", "max", "expected", "all"]
    best = 99999999
    for key in keys:
        if data[f"inverse_{key}"]["EER"] < best:
            best = data[f"inverse_{key}"]["EER"]
            best_key = key
            
    inverse_EER = data[f"inverse_{best_key}"]["EER"]
    print(f"\\bf Inversion (Untargeted / {best_key}) & {inverse_EER:.2f} \\\\")

def table_3():
    # Ablation: Temperature Inference
    
    files = [
        "none_6400_temperature=0.3_top_p=0.9_luar_plagiarism.json",
        "none_6400_temperature=0.5_top_p=0.9_luar_plagiarism.json",
        "none_6400_temperature=0.6_top_p=0.9_luar_plagiarism.json",
        "none_6400_temperature=0.7_top_p=0.9_luar_plagiarism.json",
        "none_6400_temperature=0.8_top_p=0.9_luar_plagiarism.json",
        "none_6400_temperature=0.9_top_p=0.9_luar_plagiarism.json",
        "none_6400_temperature=1.5_top_p=0.9_luar_plagiarism.json",
    ]

    sims = []
    for fname in files:
        assert os.path.exists(f"./metrics/{fname}")
        data = json.loads(open(f"./metrics/{fname}").read())
        similarities = data["inverse_max"]["similarities"]
        labels = data["inverse_max"]["labels"]
        similarities_target = [sim for sim, label in zip(similarities, labels) if label == 1]
        sims.append(np.mean(similarities_target))

    files = [
        "none_6400_temperature=0.3_top_p=0.9_simple_untargeted.json",
        "none_6400_temperature=0.5_top_p=0.9_simple_untargeted.json",
        "none_6400_temperature=0.6_top_p=0.9_simple_untargeted.json",
        "none_6400_temperature=0.7_top_p=0.9_simple_untargeted.json",
        "none_6400_temperature=0.8_top_p=0.9_simple_untargeted.json",
        "none_6400_temperature=0.9_top_p=0.9_simple_untargeted.json",
        "none_6400_temperature=1.5_top_p=0.9_simple_untargeted.json",
    ]
    
    bleu = []
    for fname in files:
        assert os.path.exists(f"./metrics/{fname}")
        data = json.loads(open(f"./metrics/{fname}").read())
        bleu.append(data["bleu"]["inverse_max"])
    
    temperatures = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.5]
    for temp, sim, b in zip(temperatures, sims, bleu):
        print(f"{temp} & {sim:.2f} & {b:.2f} \\\\")
    
def macro_metric_author(labels, similarities, key="EER"):
    step_size = 100
    values = []
    for i in range(0, len(labels), step_size):
        labels_macro = labels[i:i+step_size]
        similarities_macro = similarities[i:i+step_size]
        metrics = calculate_metrics(labels_macro, similarities_macro)
        values.append(metrics[key])
    return np.mean(values)

def table_4(macro=False):
    # Author ID
    
    fname = "./metrics/none_6400_temperature=0.7_top_p=0.9_luar_author.json"
    data_style = json.loads(open(fname).read())

    AUC = macro_metric_author(data_style["rephrase"]["labels"], data_style["rephrase"]["similarities"], key="AUC")
    print(f"\\bf Paraphrases & N/A & {AUC:.2f} \\\\")
    print("\\midrule")
    
    print("\\bf Inversion Untargeted & & \\\\")
    order = ["single", "max", "expected", "all"]
    for ord in order:
        key = f"inverse_{ord}"
        if ord == "all":
            ord_print = "Aggregate"
        elif ord == "expected":
            ord_print = "Expectation"
        else:
            ord_print = ord.capitalize()

        if macro:
            AUC = macro_metric_author(data_style[key]["labels"], data_style[key]["similarities"], key="AUC")
        else:
            AUC = data_style[key]["AUC"]
        print(f"& {ord_print} & {AUC:.2f} \\\\")

    fname = "./metrics/none_targetted=examples_6400_temperature=0.7_top_p=0.9.vllm_n=5.targetted_mode=author_num_examples=2_luar_author.json"
    data_style = json.loads(open(fname).read())
    
    print("\\bf Inversion Targeted & & \\\\")
    order = ["single", "max", "expected", "all"]
    for ord in order:
        key = f"inverse_{ord}"
        if ord == "all":
            ord_print = "Aggregate"
        elif ord == "expected":
            ord_print = "Expectation"
        else:
            ord_print = ord.capitalize()

        AUC = macro_metric_author(data_style[key]["labels"], data_style[key]["similarities"], key="AUC")
        print(f"& {ord_print} & {AUC:.2f} \\\\")
    print("\\midrule")

print(colored("Table 1", "blue"))
table_1()
print()
# print(colored("Table 2", "blue"))
# table_2()
# print(colored("Table 3", "blue"))
# table_3()
# print(colored("Table 4", "blue"))
# table_4(macro=True)