
import os
import json

import numpy as np
from termcolor import colored

from metric_utils import calculate_metrics

METRICS_FNAME = "./metrics/new"
def load_metrics_from_file(
    filename: str,
    dataset_name: str,
    mode: str,
    metrics_type: str,
):
    path = os.path.join(METRICS_FNAME, dataset_name, mode, metrics_type, filename)
    data = json.loads(open(path).read())
    return data

def get_similarities(data: dict, key: str, target: int = 1):
    similarities = data[key]["similarities"]
    labels = data[key]["labels"]
    similarities = [sim for sim, label in zip(similarities, labels) if label == target]
    return similarities

def table_1():
    # Inversion Results on Novel Authors & Writing Samples
    
    # string = "\\bf Paraphrases & - & "
    # for metrics_type in ["crud", "sbert", "basic"]:
    #     mode = "plagiarism" if metrics_type != "basic" else ""
    #     data = load_metrics_from_file(
    #         "none_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=100",
    #         "data.jsonl.filtered.cleaned_kmeans_100",
    #         mode=mode,
    #         metrics_type=metrics_type,
    #     )
    #     if metrics_type == "crud":
    #         sim_target = get_similarities(data, "rephrase", 1)
    #         sim_other = get_similarities(data, "rephrase", 0)
    #         # string += f"{np.mean(sim_target):.2f} & {np.mean(sim_other):.2f} & "
    #         string += f"{np.mean(sim_target):.2f} & "
    #     elif metrics_type == "sbert":
    #         sim_target = get_similarities(data, "rephrase", 1)
    #         string += f"{np.mean(sim_target):.2f} & "
    #     else:
    #         BLEU = data["bleu"]["rephrase"]
    #         string += f"{BLEU:.2f} \\\\"
    # print(string)

    name_to_file = {
        "GPT-4": "gpt4_inverse.jsonl.n=5",
        # "GPT-4 (In-Context)": "gpt4_inverse_in_context_correct.jsonl",
        # "output2prompt": "output2prompt_6400_temperature=0.7_top_p=0.9.jsonl.n=5",
        # "Untargeted Inversion": "none_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=5",
        # "Targeted Inversion (In-Context)": "none_targetted=examples_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=5.targetted_mode=eval_all",
        # "Targeted Inversion (Style Emb.)": "none_targetted_6400_temperature=0.7_top_p=0.9.jsonln=5.targetted_mode=eval_all",
    }
    for name, fname in name_to_file.items():
        header_string = f"\\bf {name} & & & & & \\\\"
        print(header_string)
        
        type_strings = {
            "inverse_single": "& Single & ",
            "inverse_max": "& Max & ",
            "inverse_expected": "& Expectation & ",
            "inverse_all": "& Aggregate & ",
        }

        dataset_name = "gpt4" if "GPT-4" in name else "data.jsonl.filtered.cleaned_kmeans_100"
        for metrics_type in ["crud", "sbert", "basic"]:
            mode = "plagiarism" if metrics_type != "basic" else ""
            data = load_metrics_from_file(
                fname,
                dataset_name,
                mode=mode,
                metrics_type=metrics_type,
            )
            for type in type_strings.keys():
                if metrics_type == "crud":
                    sim_target = get_similarities(data, type, 1)
                    sim_other = get_similarities(data, type, 0)
                    # type_strings[type] += f"{np.mean(sim_target):.2f} & {np.mean(sim_other):.2f} & "
                    type_strings[type] += f"{np.mean(sim_target):.2f} & "
                elif metrics_type == "sbert":
                    sim_target = get_similarities(data, type, 1)
                    type_strings[type] += f"{np.mean(sim_target):.2f} & "
                else:
                    if type != "inverse_all":
                        BLEU = data["bleu"][type]
                        type_strings[type] += f"{BLEU:.2f} \\\\"
                    else:
                        type_strings[type] += "- \\\\"

        print(type_strings["inverse_single"])
        print(type_strings["inverse_max"])
        print(type_strings["inverse_expected"])
        print(type_strings["inverse_all"])

def table_2(metric_key="EER"):
    # Plagiarism Detection with Untargeted Inversion

    data = load_metrics_from_file(
        "none_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=100",
        "data.jsonl.filtered.cleaned_kmeans_100",
        mode="plagiarism",
        metrics_type="crud",
    )
    rephrase_metric = data["rephrase"][metric_key]
    print(f"\\bf Paraphrases & {rephrase_metric:.2f} \\\\")
    
    # keys = ["single", "max", "expected", "all"]
    keys = ["single", "max", "expected"]
    best = 99999999
    for key in keys:
        if data[f"inverse_{key}"][metric_key] < best:
            best = data[f"inverse_{key}"][metric_key]
            best_key = key
            
    inverse_metric = data[f"inverse_{best_key}"][metric_key]
    print(f"\\bf Inversion (Untargeted) / {best_key} & {inverse_metric:.2f} \\\\")
    
    data = load_metrics_from_file(
        "gpt4_inverse_in_context_correct.jsonl",
        "gpt4",
        mode="plagiarism",
        metrics_type="crud",
    )
    best = 99999999
    for key in keys:
        if data[f"inverse_{key}"][metric_key] < best:
            best = data[f"inverse_{key}"][metric_key]
            best_key = key
    print(f"\\bf GPT-4 (In-Context) / {best_key} & {data[f'inverse_{best_key}'][metric_key]:.2f} \\\\")

    data = load_metrics_from_file(
        "gpt4_inverse.jsonl",
        "gpt4",
        mode="plagiarism",
        metrics_type="crud",
    )
    best = 99999999
    for key in keys:
        if data[f"inverse_{key}"][metric_key] < best:
            best = data[f"inverse_{key}"][metric_key]
            best_key = key
    print(f"\\bf GPT-4 / {best_key} & {data[f'inverse_{best_key}'][metric_key]:.2f} \\\\")

    data = load_metrics_from_file(
        "output2prompt_6400_temperature=0.7_top_p=0.9.jsonl.n=100",
        "data.jsonl.filtered.cleaned_kmeans_100",
        mode="plagiarism",
        metrics_type="crud",
    )
    best = 99999999
    for key in keys:
        if data[f"inverse_{key}"][metric_key] < best:
            best = data[f"inverse_{key}"][metric_key]
            best_key = key
    print(f"\\bf output2prompt & {data[f'inverse_{best_key}'][metric_key]:.2f} \\\\")

def table_3():
    # Ablation: Temperature Inference

   files = [
        "none_6400_temperature=0.3_top_p=0.9.jsonl.vllm_n=100",
        "none_6400_temperature=0.5_top_p=0.9.jsonl.vllm_n=100",
        "none_6400_temperature=0.6_top_p=0.9.jsonl.vllm_n=100",
        "none_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=100",
        "none_6400_temperature=0.8_top_p=0.9.jsonl.vllm_n=100",
        "none_6400_temperature=0.9_top_p=0.9.jsonl.vllm_n=100",
        "none_6400_temperature=1.5_top_p=0.9.jsonl.vllm_n=100",
    ]
   temperatures = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.5]
   
   for temp, filename in zip(temperatures, files):
       data = load_metrics_from_file(
           filename,
           "data.jsonl.filtered.cleaned_kmeans_100",
           mode="plagiarism",
           metrics_type="crud",
       )
       similarities = get_similarities(data, "inverse_max", 1)
       similarities = np.mean(similarities)

       data = load_metrics_from_file(
           filename,
           "data.jsonl.filtered.cleaned_kmeans_100",
           mode="",
           metrics_type="basic",
       )
       bleu = data["bleu"]["inverse_max"]
       print(f"\\bf {temp} & {similarities:.2f} & {bleu:.2f} \\\\")
            
def macro_metric_author(labels, similarities, key="EER"):
    step_size = 100
    values = []
    for i in range(0, len(labels), step_size):
        labels_macro = labels[i:i+step_size]
        similarities_macro = similarities[i:i+step_size]
        metrics = calculate_metrics(labels_macro, similarities_macro)
        values.append(metrics[key])
    return np.mean(values)

def table_4(key="EER"):
    # Author ID
    
    name_to_file = {
        "Inversion Untargeted": "none_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=100",
        "Targeted Inversion (In-Context)": "none_targetted=examples_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=5.targetted_mode=author",
        "Targeted Inversion (Style Emb.)": "none_targetted_6400_temperature=0.7_top_p=0.9.jsonln=5.targetted_mode=author",
        "output2prompt": "output2prompt_6400_temperature=0.7_top_p=0.9.jsonl.n=100",
        "GPT-4": "gpt4_inverse.jsonl",
        "GPT-4 (In-Context)": "gpt4_inverse_in_context_correct.jsonl",
    }
    for j, (name, filename) in enumerate(name_to_file.items()):
        dataset_name = "data.jsonl.filtered.cleaned_kmeans_100" if "GPT-4" not in name else "gpt4"
        data = load_metrics_from_file(
            filename,
            dataset_name,
            mode="author",
            metrics_type="crud",
        )
        
        if j == 0:
            metric = macro_metric_author(data["rephrase"]["labels"], data["rephrase"]["similarities"], key=key)
            print(f"\\bf Paraphrases & {metric:.2f} \\\\")
    
        # inverse_keys = ["inverse_single", "inverse_max", "inverse_expected", "inverse_all"]
        inverse_keys = ["inverse_all"]
        metrics_values = [
            macro_metric_author(data[invkey]["labels"], data[invkey]["similarities"], key=key)
            for invkey in inverse_keys
        ]
        if key == "AUC":
            index = np.argmax(metrics_values)
        else:
            index = np.argmin(metrics_values)

        metric = metrics_values[index]
        mapping = {
            "inverse_single": "Single",
            "inverse_max": "Max",
            "inverse_expected": "Expectation",
            "inverse_all": "Aggregate",
        }
        best_key = mapping[inverse_keys[index]]
        print(f"\\bf {name} / {best_key} & {metric:.2f} \\\\")

def table_5():
    # Inversion Results on Novel Authors & Writing Samples
    name_to_file = {
        "Untargeted Inversion": "none_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=100.gpt4",
    }
    for name, fname in name_to_file.items():
        header_string = f"\\bf {name} & & & & & \\\\"
        
        type_strings = {
            "rephrase": "\\bf Paraphrases & - & ",
            "inverse_single": "& Single & ",
            "inverse_max": "& Max & ",
            "inverse_expected": "& Expectation & ",
            "inverse_all": "& Aggregate & ",
        }

        dataset_name = "data.jsonl.filtered.cleaned_kmeans_100"
        for metrics_type in ["crud", "sbert", "basic"]:
            mode = "plagiarism" if metrics_type != "basic" else ""
            data = load_metrics_from_file(
                fname,
                dataset_name,
                mode=mode,
                metrics_type=metrics_type,
            )
            for type in type_strings.keys():
                if metrics_type == "crud":
                    sim_target = get_similarities(data, type, 1)
                    sim_other = get_similarities(data, type, 0)
                    # type_strings[type] += f"{np.mean(sim_target):.2f} & {np.mean(sim_other):.2f} & "
                    type_strings[type] += f"{np.mean(sim_target):.2f} & "
                elif metrics_type == "sbert":
                    sim_target = get_similarities(data, type, 1)
                    type_strings[type] += f"{np.mean(sim_target):.2f} & "
                else:
                    if type != "inverse_all":
                        BLEU = data["bleu"][type]
                        type_strings[type] += f"{BLEU:.2f} \\\\"
                    else:
                        type_strings[type] += "- \\\\"

        print(type_strings["rephrase"])
        print(header_string)
        print(type_strings["inverse_single"])
        print(type_strings["inverse_max"])
        print(type_strings["inverse_expected"])
        print(type_strings["inverse_all"])

def table_6():
    # Inversion Results on Machine Text
    name_to_file = {
        "GPT-4": "gpt4_inverse_machine-paraphrase.jsonl",
        "Output2Prompt": "output2prompt-rebuttal_6400_temperature=0.7_top_p=0.9.jsonl.n=100",
        "Untargeted Inversion": "none_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=100",
    }
    for name, fname in name_to_file.items():
        
        type_strings = {
            "rephrase": "\\bf Paraphrases & - & ",
            "inverse_single": "& Single & ",
            "inverse_max": "& Max & ",
            "inverse_expected": "& Expectation & ",
            "inverse_all": "& Aggregate & ",
        }

        for metrics_type in ["crud", "sbert", "basic"]:
            dataset_name = "gpt4" if "gpt4" in name.lower() else "data.jsonl.filtered.respond_reddit.cleaned"
            mode = "plagiarism" if metrics_type != "basic" else ""
            data = load_metrics_from_file(
                fname,
                dataset_name,
                mode=mode,
                metrics_type=metrics_type,
            )
            for type in type_strings.keys():
                if metrics_type == "crud":
                    sim_target = get_similarities(data, type, 1)
                    sim_other = get_similarities(data, type, 0)
                    # type_strings[type] += f"{np.mean(sim_target):.2f} & {np.mean(sim_other):.2f} & "
                    type_strings[type] += f"{np.mean(sim_target):.2f} & "
                elif metrics_type == "sbert":
                    sim_target = get_similarities(data, type, 1)
                    type_strings[type] += f"{np.mean(sim_target):.2f} & "
                else:
                    if type != "inverse_all":
                        BLEU = data["bleu"][type]
                        type_strings[type] += f"{BLEU:.2f} \\\\"
                    else:
                        type_strings[type] += "- \\\\"

        print(type_strings["rephrase"])
        header_string = f"\\bf {name} & & & & \\\\"
        print(header_string)
        print(type_strings["inverse_single"])
        print(type_strings["inverse_max"])
        print(type_strings["inverse_expected"])
        print(type_strings["inverse_all"])
        
def table_7():
    # Inversion Results on Novel Authors & Writing Samples
    dataset_names = [
        "temperature=0.3.cleaned",
        "temperature=0.5.cleaned",
        "data.jsonl.filtered.cleaned_kmeans_100",
    ]
    filename = "none_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=100"
    temperature = [0.3, 0.5, 0.7]
    for temp, dset_name in zip(temperature, dataset_names):
        string = f"{temp} & "
        for metrics_type in ["crud", "basic"]:
            mode = "plagiarism" if metrics_type != "basic" else ""
            data = load_metrics_from_file(
                filename,
                dset_name,
                mode=mode,
                metrics_type=metrics_type,
            )
            
            if metrics_type == "crud":
                sim_target = get_similarities(data, "inverse_max", 1)
                string += f"{np.mean(sim_target):.2f} & "
            elif metrics_type == "basic":
                BLEU = data["bleu"]["inverse_max"]
                string += f"{BLEU:.2f} \\\\"
        print(string)

def table_8():
    # Inversion Results on Novel Authors & Writing Samples
    name_to_file = {
        "Llama-3-8B": "baseline_Meta-Llama-3-8B-Instruct_in-context=False.jsonl",
        "Mistral-7b": "baseline_Mistral-7B-Instruct-v0.3_in-context=False.jsonl",
        "Phi-3": "baseline_Phi-3-mini-4k-instruct_in-context=False.jsonl",
        "Llama-3-8B (In-Context)": "baseline_Meta-Llama-3-8B-Instruct_in-context=True_n=100.jsonl",
        "Mistral-7b (In-Context)": "baseline_Mistral-7B-Instruct-v0.3_in-context=True_n=100.jsonl",
        "Phi-3 (In-Context)": "baseline_Phi-3-mini-4k-instruct_in-context=True_n=100.jsonl",
    }

    for name, filename in name_to_file.items():
        data = load_metrics_from_file(
            filename,
            "data.jsonl.filtered.cleaned_kmeans_100",
            mode="",
            metrics_type="basic",
        )
        BLEU = data["bleu"]["inverse_max"]
        print(f"\\bf {name} & {BLEU:.2f} \\\\")

def table_9():
    
    filenames = [
        "none_targetted=examples_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=5.targetted_mode=author_num_examples=1",
        "none_targetted=examples_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=5.targetted_mode=author_num_examples=2",
        "none_targetted=examples_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=5.targetted_mode=author_num_examples=3",
        "none_targetted=examples_6400_temperature=0.7_top_p=0.9.jsonl.vllm_n=5.targetted_mode=author",
        "none_targetted_6400_temperature=0.7_top_p=0.9.jsonln=5.targetted_mode=author",
    ]
    
    for fname in filenames:
        data = load_metrics_from_file(
            fname,
            "data.jsonl.filtered.cleaned_kmeans_100",
            mode="author",
            metrics_type="crud",
        )
        similarities_target = get_similarities(data, "inverse_all", 1)
        similarities_other = get_similarities(data, "inverse_all", 0)

        mean_target = np.mean(similarities_target)
        mean_other = np.mean(similarities_other)
        delta = np.median(similarities_target) - np.median(similarities_other)
        print(f"{fname} & {mean_target:.2f} & {mean_other:.2f} & {delta:.2f} \\\\")

# print(colored("Table 1", "blue"))
# table_1()
# print()
# print(colored("Table 2", "blue"))
# table_2("EER")
# table_2("AUC")
# print()
# print(colored("Table 3", "blue"))
# table_3()
print(colored("Table 4", "blue"))
table_4("EER")
# print(colored("Table 5", "blue"))
# table_5()
# print(colored("Table 6", "blue"))
# table_6()
# table_7()
# table_8()
# table_9()