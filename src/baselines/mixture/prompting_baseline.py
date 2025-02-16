# Can LLMs invert their own outputs?

import os
import sys
import random; random.seed(43)
from argparse import ArgumentParser

import pandas as pd
from vllm import (
    LLM,
    SamplingParams,
)

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3",
                    choices=["mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Meta-Llama-3-8B-Instruct", "microsoft/Phi-3-mini-4k-instruct"])
parser.add_argument("--in_context", action="store_true")
parser.add_argument("--num_examples", type=int, default=100)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

TEST_FPATH = "/data1/yubnub/changepoint/MUD_inverse/data/data.jsonl.filtered.cleaned_kmeans_100/inverse_output/test.small.jsonl"

def build_inverse_prompt_gpt4(
    generation: str,
    examples: list[str] = None
) -> str:
    if examples is not None:
        seperator = "\n-----\n"
        header = "Here are examples of paraphrases and their original:\n"
        for example in examples:
            header += f"Paraphrase: {example[0]}\n"
            header += f"Original: {example[1]}"
            header += seperator
    else:
        header = ""

    base_instruction = "The following passage is a mix of human and machine text, recover the original human text:"
    instruction = base_instruction
    prompt = f"{header}{instruction} {generation}"
    return prompt

def create_examples(
    df: pd.DataFrame, 
    in_context: bool = False
) -> list[str]:
    original_units = []
    original_rephrases = []
    inversion_prompts = []
    
    for _, row in df.iterrows():
        rephrases = row["rephrase"]
        units = row["unit"]
        assert len(rephrases) == len(units)
        
        for j, rephrase in enumerate(rephrases):
            if in_context:
                not_used_rephrases = rephrases[:j] + rephrases[j+1:]
                not_used_units = units[:j] + units[j+1:]
                random.shuffle(not_used_rephrases)
                random.shuffle(not_used_units)
                prompt = build_inverse_prompt_gpt4(rephrase, list(zip(not_used_rephrases, not_used_units))[:5])
                inversion_prompts.append(prompt)
            else:
                prompt = build_inverse_prompt_gpt4(rephrase)
                inversion_prompts.append(prompt)
                
            original_units.append(units[j])
            original_rephrases.append(rephrase)
            
    return inversion_prompts, original_units, original_rephrases

def main():
    df = pd.read_json(TEST_FPATH, lines=True)
    model_basename = os.path.basename(args.model_name)
    df = df[df.model_name == model_basename]
    df = df.groupby("author_id").agg(list).reset_index()

    if args.debug:
        df = df.head(2)
    
    os.makedirs("./baseline_results", exist_ok=True)
    prompts, units, rephrases = create_examples(df, in_context=args.in_context)

    model = LLM(args.model_name)
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=128+32,
        n=args.num_examples,
    )

    inversions = []
    for j in range(0, len(prompts), 32):
        out = model.generate(prompts[j:j+32], sampling_params)
        out = [list(set([o.text for o in out.outputs])) for out in out]
        inversions.extend(out)

    save_df = pd.DataFrame({
        "unit": units,
        "rephrase": rephrases,
        "inverse": inversions,
        "inverse_prompt": prompts,
    })

    save_df.to_json(f"./baseline_results/baseline_{model_basename}_in-context={args.in_context}_n={args.num_examples}.jsonl", lines=True, orient="records")

    return 0

if __name__ == "__main__":
    sys.exit(main())
