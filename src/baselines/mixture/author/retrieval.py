
import os
import sys

import Levenshtein
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from einops import rearrange, reduce, repeat
from sklearn.metrics import pairwise_distances

from config import LUARConfig
from model import LUAR

def get_levenshtein_tags(
    string1: str, 
    string2: str, 
    tokenizer: AutoTokenizer,
):
    """Code taken and adapted from: 
        https://github.com/machelreid/lewis/blob/master/roberta-tagger/preprocess-roberta-tagger.py

       TODO: Comment the function and cleanup
    """
    str1 = tokenizer(string1)
    str2 = tokenizer(string2)
    tokens = list(set(str1 + str2))
    # to assure we get an independent character for every individual token in each sentence
    characters = list(
        set(
            "1234567890-qwertyuiopasdfghjklzxcvbQWERTYUIOOASDFGHJKLZXCVBNMnm,./;[]-+_=|}{><?!!@#$%^&*()iœ∑´®†\¨ˆøπ¬˚∆˙©ƒ∂ßåååååΩ≈ç√∫˜µ≤øˆ˙©ƒ¨∂®ß´∑ß´\†∂®¨あグノシーは、ネット上に存在する様々な情報を独自のアルゴリズムで収集し、評価付けを行い、ユーザーに届ける情報キュレーションサービスである。CEOの福島は、「ユーザーが面白いと思うコンテンツを、ニュースに限らず配信していった結果、自然とエンターテインメント性の強いメディアになった」と述べている†ƒ√˚©電気通信事業者（でんきつうしんじぎょうしゃ）とは、一般に固定電話や携帯電話等の電気通信サービスを提供する会社の総称。「音声やデータを運ぶ」というところから通信キャリア（または単にキャリア）や通信回線事業者（または単に回線事業者）と呼ばれることもある。携帯電話専業の会社については携帯会社と呼ぶことが多いがDocomoなどの携帯電話回線会社とAppleなどの携帯電話製造会社との混同されるため呼ばれ方が変わりつつある。回線事業者または回線会社として扱われる。∆˙˚∫∆…√©∆ç®™£∞¢§¶•ªªªª¸˛Ç◊ı˜Â¯Â¯˘¿ÆÚÒÔÓ˝ÏÎÍÅÅÅÅŒ„´‰ˇÁ¨ˆØ∏∏∏∏””’»±—‚·°‡ﬂﬁ›‹⁄`乙 了 又 与 及 丈 刃 凡 勺 互 弔 井 升 丹 乏 匁 屯 介 冗 凶 刈 匹 厄 双 孔 幻 斗 斤 且 丙 甲 凸 丘 斥 仙 凹 召 巨 占 囚 奴 尼 巧 払 汁 玄 甘 矛 込 弐 朱 吏 劣 充 妄 企 仰 伐 伏 刑 旬 旨 匠 叫 吐 吉 如 妃 尽 帆 忙 扱 朽 朴 汚 汗 江 壮 缶 肌 舟 芋 芝 巡 迅 亜 更 寿 励 含 佐 伺 伸 但 伯 伴 呉 克 却 吟 吹 呈 壱 坑 坊 妊 妨 妙 肖 尿 尾 岐 攻 忌 床 廷 忍 戒 戻 抗 抄 択 把 抜 扶 抑 杉 沖 沢 沈 没 妥 狂 秀 肝 即 芳 辛 迎 邦 岳 奉 享 盲 依 佳 侍 侮 併 免 刺 劾 卓 叔 坪 奇 奔 姓 宜 尚 屈 岬 弦 征 彼 怪 怖 肩 房 押 拐 拒 拠 拘 拙 拓 抽 抵 拍 披 抱 抹 昆 昇 枢 析 杯 枠 欧 肯 殴 況 沼 泥 泊 泌 沸 泡 炎 炊 炉 邪 祈 祉 突 肢 肪 到 茎 苗 茂 迭 迫 邸 阻 附 斉 甚 帥 衷 幽 為 盾 卑 哀 亭 帝 侯 俊 侵 促 俗 盆 冠 削 勅 貞 卸 厘 怠 叙 咲 垣 契 姻 孤 封 峡 峠 弧 悔 恒 恨 怒 威 括 挟 拷 挑 施 是 冒 架 枯 柄 柳 皆 洪 浄 津 洞 牲 狭 狩 珍 某 疫 柔 砕 窃 糾 耐 胎 胆 胞 臭 荒 荘 虐 訂 赴 軌 逃 郊 郎 香 剛 衰 畝 恋 倹 倒 倣 俸 倫 翁 兼 准 凍 剣 剖 脅 匿 栽 索 桑 唆 哲 埋 娯 娠 姫 娘 宴 宰 宵 峰 貢 唐 徐 悦 恐 恭 恵 悟 悩 扇 振 捜 挿 捕 敏 核 桟 栓 桃 殊 殉 浦 浸 泰 浜 浮 涙 浪 烈 畜 珠 畔 疾 症 疲 眠 砲 祥 称 租 秩 粋 紛 紡 紋 耗 恥 脂 朕"
        )
    )
    st1 = "".join([characters[tokens.index(x)] for x in str1])
    st2 = "".join([characters[tokens.index(x)] for x in str2])
    output_list = ["KEEP"] * len(str1)
    output = Levenshtein.editops(st1, st2)
    indices = [i[1] for i in output if i[0] != "insert"]
    delete_indices = [i[1] for i in output if i[0] == "delete"]

    new_str2 = {}
    for something in output:
        if something[0] != "delete":
            try:
                new_str2[something[1]].append(str2[something[2]])
            except:
                new_str2[something[1]] = [str2[something[2]]]
                
    for x in indices:
        try:
            output_list[x] = "MASK"
        except:
            pass
    for x in delete_indices:
        try:
            output_list[x] = "DELETE"
        except:
            pass
        
    new_list = output_list.copy()
    for i, j in enumerate(output_list):
        if j == "KEEP":
            new_list[i] = str1[i]
    mask2fill = []
    former_key = 0
    for j, i in enumerate(sorted(new_str2.keys())):
        sep = ["</s>"] if (i - former_key > 1 and j != 0) else []
        mask2fill.extend(sep + new_str2[i])
        former_key = i
    mask2fill = mask2fill
    return output_list

def mean_pooling(
    token_embeddings, 
    attention_mask,
    indices: list[int] = None,
):
    """Mean Pooling as described in the SBERT paper.
    """
    if indices is not None:
        
        indices = [j for i in indices for j in i]
        all_token_embeddings = []
        for j, index in enumerate(indices):
            idx = torch.LongTensor([index]).to(token_embeddings.device)
            embeddings = torch.index_select(token_embeddings[j:j+1], 1, idx)
            att_mask = torch.index_select(attention_mask[j:j+1], 1, idx)
            sum_embeddings = reduce(embeddings * att_mask.unsqueeze(-1), 'b l d -> b d', 'sum')
            sum_mask = torch.clamp(reduce(att_mask.unsqueeze(-1), 'b l d -> b d', 'sum'), min=1e-9)
            all_token_embeddings.append(sum_embeddings / sum_mask)
        return torch.cat(all_token_embeddings, dim=0)

    input_mask_expanded = repeat(attention_mask, 'b l -> b l d', d=768).type(token_embeddings.type())
    sum_embeddings = reduce(token_embeddings * input_mask_expanded, 'b l d -> b d', 'sum')
    sum_mask = torch.clamp(reduce(input_mask_expanded, 'b l d -> b d', 'sum'), min=1e-9)
    return sum_embeddings / sum_mask


# def debug_run_dummy(
#     model: LUAR, 
#     tokenizer: AutoTokenizer,
# ):
#     """Runs a debug example.
#     """
#     s = "This is a string, wow."
#     inputs = tokenizer(s, return_tensors="pt", padding=True, truncation=True)
#     inputs["input_ids"] = inputs["input_ids"].unsqueeze(1)
#     inputs["attention_mask"] = inputs["attention_mask"].unsqueeze(1)
#     inputs.to(model.device)
    
#     regular_embedding = model(**inputs)
#     regular_embedding = F.normalize(regular_embedding, p=2, dim=-1)

#     strings = []
#     similarities = []
#     for j in range(len(inputs["input_ids"][0][0][1:-1])):
#         ids = inputs["input_ids"][0][0].cpu().numpy().tolist()
#         string = tokenizer.decode([idx for k, idx in enumerate(ids) if k != j+1], skip_special_tokens=True)
#         mean_pool_indices = [[j+1]]
#         mean_pool_embedding = model(**inputs, mean_pooling_indices=mean_pool_indices)
#         mean_pool_embedding = F.normalize(mean_pool_embedding, p=2, dim=-1)
#         similarity = F.cosine_similarity(regular_embedding, mean_pool_embedding, dim=-1)

#         strings.append(string)
#         similarities.append(similarity.item())
        
#     print("Original String:", s)
#     sorted_indices = np.argsort(similarities)
#     for index in sorted_indices:
#         print(strings[index], similarities[index])

def ranking(queries, 
            targets,
            query_authors, 
            target_authors, 
            metric="cosine",
            return_distances=False,
):
    num_queries = len(query_authors)
    ranks = np.zeros((num_queries), dtype=np.float32)
    reciprocal_ranks = np.zeros((num_queries), dtype=np.float32)
    
    distances = pairwise_distances(queries, Y=targets, metric=metric, n_jobs=-1)

    for i in range(num_queries):
        dist = distances[i]
        sorted_indices = np.argsort(dist)
        sorted_target_authors = target_authors[sorted_indices]
        ranks[i] = np.where(sorted_target_authors == query_authors[i])[0].item() + 1
        reciprocal_ranks[i] = 1.0 / float(ranks[i])
        
    return_dict = {
        "R@01": np.sum(np.less_equal(ranks, 1)) / np.float32(num_queries),
        "R@08": np.sum(np.less_equal(ranks, 8)) / np.float32(num_queries),
        "R@16": np.sum(np.less_equal(ranks, 16)) / np.float32(num_queries),
        "R@32": np.sum(np.less_equal(ranks, 32)) / np.float32(num_queries),
        "R@64": np.sum(np.less_equal(ranks, 64)) / np.float32(num_queries),
        "MRR": np.mean(reciprocal_ranks)
    }
    
    if return_distances:
        return return_dict, distances

    return return_dict

def main():    
    model = SentenceTransformer("AnnaWegmann/Style-Embedding")
    tokenizer = AutoTokenizer.from_pretrained("AnnaWegmann/Style-Embedding")

    df = pd.read_json("../test_data/author.jsonl", lines=True)
    df = df[df["model_name"].apply(lambda x: "Mistral" in x)]

    ids = np.array(df["id"].tolist())
    units = df["unit"].tolist()
    generations = df.generation.tolist()
    
    unit_embeddings = model.encode(units, normalize_embeddings=True)
    generation_embeddings = model.encode(generations, normalize_embeddings=True)
    ranking_dict = ranking(generation_embeddings, unit_embeddings, ids, ids, metric="cosine")
    print("Base Ranking:", ranking_dict)
    
    selective_embeddings = []
    token_embeddings_generation = model.encode(generations, convert_to_tensor=True, output_value="token_embeddings")
    for j, token_embeddings in enumerate(token_embeddings_generation):
        attention_mask = tokenizer(generations[j], return_tensors="pt", padding=True, truncation=True)["attention_mask"]
        attention_mask = attention_mask.to(token_embeddings.device)
        
        tags = get_levenshtein_tags(generations[j], units[j], tokenizer.tokenize)
        indices_to_keep = [i for i, tag in enumerate(tags) if tag == "KEEP"]
        if len(indices_to_keep) == 0:
            import pdb; pdb.set_trace()
            selective_embeddings.append(mean_pooling(token_embeddings_generation[j].unsqueeze(0), attention_mask))
            continue
        
        mean_pool_indices = [[idx+1 for idx in indices_to_keep]]
        selective_embedding = mean_pooling(token_embeddings_generation[j].unsqueeze(0), attention_mask, mean_pool_indices)
        selective_embeddings.append(selective_embedding)
    selective_embeddings = torch.cat(selective_embeddings, dim=0).cpu().numpy()
    ranking_selective = ranking(selective_embeddings, unit_embeddings, ids, ids, metric="cosine")
    print("Selective Ranking:", ranking_selective)
    # attention_mask = tokenizer(generation, return_tensors="pt", padding=True, truncation=True)["attention_mask"]
    # attention_mask = attention_mask.to(token_embeddings_generation.device)
    
    # tags = get_levenshtein_tags(generation, unit, tokenizer.tokenize)
    # indices_to_keep = [i for i, tag in enumerate(tags) if tag == "KEEP"]
    # mean_pool_indices = [[idx+1 for idx in indices_to_keep]]
    # selective_embedding = mean_pooling(token_embeddings_generation, attention_mask, mean_pool_indices)

    # tokens = tokenizer.tokenize(generation)
    # import random; random.seed(43)
    # random_indices = random.sample(range(len(tokens)), k=len(mean_pool_indices[0]))
    # random_mean_pool_indices = [[idx+1 for idx in random_indices]]
    # random_selective_embedding = mean_pooling(token_embeddings_generation, attention_mask, random_mean_pool_indices)
    # random_selective_embedding = F.normalize(random_selective_embedding, p=2, dim=-1)

    # sim_gen = F.cosine_similarity(regular_embedding, generation_embedding, dim=-1).item()
    # sim_sel = F.cosine_similarity(regular_embedding, selective_embedding, dim=-1).item()
    # sim_rand = F.cosine_similarity(regular_embedding, random_selective_embedding, dim=-1).item()

    # print("Generation Similarity:", sim_gen)
    # print("Selective Similarity:", sim_sel)
    # print("Random Similarity:", sim_rand)

    return 0


if __name__ == "__main__":
    sys.exit(main())