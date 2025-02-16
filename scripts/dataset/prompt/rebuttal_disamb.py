
import json
import sys

def main():
    # fname_1 = "./outputs/inverse_prompts_machine-paraphrase.jsonl"
    # fname_2 = "./outputs/inverse_prompts_machine-paraphrase.jsonl.result.backup"

    # data_all = []
    # with open(fname_1, "r") as fin:
    #     for line in fin:
    #         d = json.loads(line)
    #         data_all.append(d)
         
    # with open(fname_2, "r") as fin:
    #     it = 0
    #     for line in fin:
    #         try:
    #             d = json.loads(line)
    #             is_done = d[1]['choices'][0]['finish_reason'] == "stop"
    #             if is_done:
    #                 index = data_all.index(d[0])
    #                 data_all.pop(index)
    #             it += 1
    #             if it % 1000 == 0:
    #                 print(f"iteration {it}")
    #         except:
    #             continue

    # with open("./outputs/inverse_prompts_machine-paraphrase.jsonl.remaining", "w+") as fout:
    #     for d in data_all:
    #         fout.write(json.dumps(d))
    #         fout.write("\n")

    fname_1 = "./outputs/inverse_prompts_machine-paraphrase.jsonl.result.backup"
    fname_2 = "./outputs/inverse_prompts_machine-paraphrase.jsonl.remaining.result"

    data_all = []
    with open(fname_1, "r") as fin:
        for line in fin:
            try:
                d = json.loads(line)
                is_done = d[1]['choices'][0]['finish_reason'] == "stop"
                if is_done:
                    data_all.append(d)
            except:
                continue
    with open(fname_2, "r") as fin:
        for line in fin:
            d = json.loads(line)
            data_all.append(d)

    with open("./outputs/inverse_prompts_machine-paraphrase.jsonl.result", "w+") as fout:
        for d in data_all:
            fout.write(json.dumps(d))
            fout.write("\n")
            
    
    return 0

if __name__ == "__main__":
    sys.exit(main())