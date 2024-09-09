
import math
from typing import Union

import Levenshtein
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from termcolor import colored
from transformers import AutoTokenizer
from tqdm import tqdm

from model import MixturePredictor

mixture_perc_to_chekpoint = {
    0.2: "./outputs/roberta-large_stratified_perc=0.2/checkpoints/checkpoint_9/",
    0.4: "./outputs/roberta-large_stratified_perc=0.4/checkpoints/checkpoint_4/",
    0.6: "./outputs/roberta-large_stratified_perc=0.6/checkpoints/checkpoint_8/",
    0.8: "./outputs/roberta-large_stratified_perc=0.8/checkpoints/checkpoint_8/",
    1.0: "./outputs/roberta-large_stratified/checkpoints/checkpoint_9/",
}

def load_mixture_predictor(perc: float = 1.0):
    print("MAKE SURE YOU LOAD THE RIGHT MIXTURE PREDICTOR DUMMY")
    import sys; sys.exit()
    model = MixturePredictor()
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    path = mixture_perc_to_chekpoint[perc]
    print(colored("Loading STRATIFIED Mixture Predictor from: {}".format(path), "yellow"))
    accelerator.load_state(path)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return model

def get_mixture_weights(
    model: MixturePredictor,
    data: Union[list[str], list[dict]],
    key: Union[str, None] = "generation",
    batch_size: int = 512,
    return_sequence_probs: bool = False,
    progress_bar: bool = True,
):
    if key is not None:
        text = [d[key] for d in data]
    else:
        text = data
        
    all_seq_preds = []
    all_token_preds = []

    iterator = range(0, len(data), batch_size)
    if progress_bar:
        iterator = tqdm(iterator)
    for i in iterator:
        batch = text[i:i+batch_size]
        sequence_preds, token_preds = model.predict(batch)
        all_seq_preds.extend(sequence_preds)
        all_token_preds.extend(token_preds)

    all_token_preds = [F.softmax(pred, dim=-1) for pred in all_token_preds]
    all_token_preds = [pred.cpu().tolist() for pred in all_token_preds]
    if return_sequence_probs:
        all_seq_preds = [F.softmax(pred, dim=-1) for pred in all_seq_preds]
        all_seq_preds = [pred.cpu().tolist() for pred in all_seq_preds]
        return all_seq_preds, all_token_preds

    return all_token_preds

def build_inverse_prompt(
    generation: str,
    original: str,
    tokens: list[list[str]] = None,
    mixture_probs: list[list[tuple[int, int]]] = None,
    prompt_type: str = "none",
) -> str:
    if prompt_type == "tokens":
        tokens_to_keep = "Keep these tokens while rephrasing, Ġ is a single space: "
        added_tokens = False
        for token, probs in zip(tokens, mixture_probs):
            if probs[0] > 0.5:
                tokens_to_keep += f"{token}, "
                added_tokens = True
                
        if not added_tokens:
            instruction = "Rephrase:"
        else:
            instruction = f"""{tokens_to_keep[:-2]}\nRephrase:"""
    elif prompt_type in ["probs" or "logprobs"]:
        log_fn = math.log if prompt_type == "logprobs" else lambda x: x

        token_and_probs = ""
        for token, probs in zip(tokens, mixture_probs):
            token_and_probs += f"{token}[{log_fn(probs[0]):.2f}], "
        token_and_probs = token_and_probs[:-2]

        if prompt_type == "probs":
            header = "Input Format: Token_1[prob_human], Token_2[prob_human], ... Token_N[prob_human]"
        else:
            header = "Input Format: Token_1[log_prob_human], Token_2[log_prob_human], ... Token_N[log_prob_human]"
        instruction = f"""{header}\n{token_and_probs}\nRephrase:"""
    else:
        instruction = "Rephrase:"
            
    prompt = f"[INST] {instruction} {generation} [/INST]\nOutput: {original}"
    return prompt

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
