#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import argparse
from pathlib import Path

# ================== 口语词配置 ==================
# 可以按需扩展
FILLER_WORDS = [
    # "好的",
    # "嗯", "对", "啊", "哦", "好的",
    # "好",
    # "嗯", "对", "啊", "哦", "好的",
    # "嗯嗯", "啊啊", "哦哦",
    # "恩", "呃", "噢", "好吧"
]


# ================== 文本正则化相关 ==================
def add_space_between_chars(text: str) -> str:
    """在每个字符之间加空格：'今天上课' -> '今 天 上 课'"""
    text = text.strip()
    if not text:
        return ""
    return " ".join(list(text))


def truncate_repeated_chars(text: str, max_repeat: int = 5) -> str:
    """将连续重复的同一字符截断到最多 max_repeat 次"""
    if not text:
        return text
    res = []
    last = None
    cnt = 0
    for ch in text:
        if ch == last:
            cnt += 1
            if cnt <= max_repeat:
                res.append(ch)
        else:
            last = ch
            cnt = 1
            res.append(ch)
    return "".join(res)


def normalize_segment(seg: str) -> str:
    """
    对非 <spkX> 片段做文本正则化：
    - 删除口语词
    - 删除标点、空格等
    - 截断重复字符
    """
    if not seg:
        return ""

    # 1) 删除口语词
    for w in FILLER_WORDS:
        seg = seg.replace(w, "")

    # 2) 删除标点、空格等，只保留汉字、字母、数字、下划线
    #   \u4e00-\u9fff 匹配中日韩统一表意文字
    seg = re.sub(r"[^\w\u4e00-\u9fff]+", "", seg)

    # 3) 截断连续重复字符
    seg = truncate_repeated_chars(seg, max_repeat=5)

    return seg


# ================== 说话人划分 ==================
def text_to_spk_dict(text: str) -> dict:
    """
    把一整句文本（句子在前，<spkX> 在后）解析成
        {"spk1": "文 本 ...", "spk2": "文 本 ...", ...}
    并在内部完成文本正则化 + 每字后加空格。

    规则：
    - 每一段“正常文本”后面跟一个 <spkX>，这段文本归到这个 spkX；
    - 若最后还有没被任何 <spkX> 接住的文本，则默认归到 spk1；
    - 同一 spk 的多段文本直接拼接。
    """
    if not text:
        return {}

    # 普通文本 / <spkX> / 普通文本 / <spkY> / ...
    parts = re.split(r"(<spk\d+>)", text)

    pending_raw = ""         # 尚未分配的“原始文本”
    spk_dict = {}            # { "spk1": "规 范 化 后 的 文 本 ...", ... }

    for part in parts:
        if not part:
            continue

        # 如果是一个 <spkX> 标签
        if re.fullmatch(r"<spk\d+>", part):
            spk = part[1:-1]  # "<spk1>" -> "spk1"

            # 把刚刚积累的 pending_raw 做正则化 + 加空格，挂到该 spk
            norm = normalize_segment(pending_raw)
            if norm:
                norm_spaced = add_space_between_chars(norm)
                spk_dict[spk] = spk_dict.get(spk, "") + norm_spaced + " "

            pending_raw = ""   # 清空缓冲，继续累积后面的文本

        else:
            # 普通文本，先累计，等遇到下一个 <spkX> 再统一归属
            pending_raw += part

    # 如果文本最后还有没被任何 <spkX> 接住的内容，就默认给 spk1
    if pending_raw.strip():
        spk = "spk1"
        norm = normalize_segment(pending_raw)
        if norm:
            norm_spaced = add_space_between_chars(norm)
            spk_dict[spk] = spk_dict.get(spk, "") + norm_spaced + " "

    # 去掉尾部多余空格
    for spk in list(spk_dict.keys()):
        spk_dict[spk] = spk_dict[spk].strip()

    return spk_dict


def sort_spk_key(spk_name: str) -> int:
    """按 spk 后面的数字排序：spk1, spk2, spk3 ..."""
    m = re.search(r"\d+", spk_name)
    if m:
        return int(m.group())
    return 0


# ================== STM 相关 ==================
def stm_line(
    utt_id: str,
    text: str,
    channel: str = "1",
    speaker: str = "spk_all",
    start: float = 0.0,
    end: float = 1.0,
    label: str = "<o>",
) -> str:
    """
    STM 格式：
      file_id channel speaker start_time end_time label text
    """
    return f"{utt_id} {channel} {speaker} {start:.2f} {end:.2f} {label} {text}".strip()


def extract_utt_id_from_wav(wav_path: str) -> str:
    """从 wav 路径中提取 utt_id，这里简单用 basename 去掉后缀"""
    return Path(wav_path).stem


# ================== 主流程 ==================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_jsonl",
        type=str,
        default="/pengjing/workspace/project/west/examples/aishell/asr/data/alimeeting_segments_test.jsonl",
        help="reference jsonl，包含 wav 和 txt（带 <spkX>）",
    )
    parser.add_argument(
        "--hyp_jsonl",
        type=str,
        default="/pengjing/workspace/project/west/examples/aishell/asr/exp/firered_from_ziyi_all_projector_lora/checkpoint-3000/alimeeting_test_result.jsonl",
        help="hypothesis jsonl，包含 txt（带 <spkX>）",
    )
    parser.add_argument(
        "--out_ref",
        type=str,
        default="/pengjing/workspace/nfs/tmp/out/ref.stm",
        help="输出 ref.stm 路径",
    )
    parser.add_argument(
        "--out_hyp",
        type=str,
        default="/pengjing/workspace/nfs/tmp/out/hyp.stm",
        help="输出 hyp.stm 路径",
    )
    parser.add_argument(
        "--dur_per_spk",
        type=float,
        default=10.0,
        help="给每个说话人分配的伪时长（秒），用于 STM start/end",
    )
    args = parser.parse_args()

    # 读 ref
    ref_lines = []
    with open(args.ref_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ref_lines.append(json.loads(line))

    # 读 hyp
    hyp_lines = []
    with open(args.hyp_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            hyp_lines.append(json.loads(line))

    if len(ref_lines) != len(hyp_lines):
        print(
            f"[WARN] ref 行数 = {len(ref_lines)}, hyp 行数 = {len(hyp_lines)}，"
            f"行数不一致，将按最小长度对齐。"
        )
    n = min(len(ref_lines), len(hyp_lines))

    with open(args.out_ref, "w", encoding="utf-8") as f_ref, open(
        args.out_hyp, "w", encoding="utf-8"
    ) as f_hyp:
        for i in range(n):
            ref_item = ref_lines[i]
            hyp_item = hyp_lines[i]

            wav_path = ref_item["wav"]
            ref_txt_raw = ref_item.get("txt", "")
            hyp_txt_raw = hyp_item.get("txt", "")

            utt_id = extract_utt_id_from_wav(wav_path)

            # 按说话人拆分 & 文本正则化
            ref_spk_dict = text_to_spk_dict(ref_txt_raw)
            hyp_spk_dict = text_to_spk_dict(hyp_txt_raw)

            # 统一说话人集合，按 spk 编号排序
            all_spks = sorted(
                set(ref_spk_dict.keys()) | set(hyp_spk_dict.keys()),
                key=sort_spk_key,
            )

            if not all_spks:
                # 没有任何非空文本，跳过
                continue

            # 每个说话人一行 STM，给伪时间戳
                        # 每个说话人一行 STM，给伪时间戳
            for idx, spk in enumerate(all_spks):
                start = idx * args.dur_per_spk
                end = (idx + 1) * args.dur_per_spk

                ref_text = ref_spk_dict.get(spk, "")
                hyp_text = hyp_spk_dict.get(spk, "")

                # 如果 ref 和 hyp 该 spk 都完全为空，就可以整体跳过
                if not ref_text and not hyp_text:
                    continue

                # --- 只在这一侧有非空文本时才写行 ---
                if ref_text:
                    ref_stm = stm_line(
                        utt_id=utt_id,
                        text=ref_text,
                        channel="1",
                        speaker=spk,
                        start=start,
                        end=end,
                        label="<o>",
                    )
                    f_ref.write(ref_stm + "\n")

                if hyp_text:
                    hyp_stm = stm_line(
                        utt_id=utt_id,
                        text=hyp_text,
                        channel="1",
                        speaker=spk,
                        start=start,
                        end=end,
                        label="<o>",
                    )
                    f_hyp.write(hyp_stm + "\n")

    print(f"[Done] 写入 ref STM: {args.out_ref}")
    print(f"[Done] 写入 hyp STM: {args.out_hyp}")


if __name__ == "__main__":
    main()
