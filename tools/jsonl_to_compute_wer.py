#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import argparse


def clean_text(text: str) -> str:
    """
    - 删除 <spkX> 标签
    - 删除标点、空格等，只保留：汉字、字母、数字、下划线
    """
    if not text:
        return ""

    # 1) 删除所有 <spk数字> 标签
    text = re.sub(r"<spk\d+>", " ", text)

    # 2) 删除标点、空格等，只保留汉字、字母、数字、下划线
    #    \u4e00-\u9fff 为中日韩统一表意文字
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)

    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_jsonl",
        type=str,
        default="/pengjing/workspace/project/west/examples/aishell/asr/data/alimeeting_segments_test.jsonl",
        help="reference jsonl，包含 txt（带 <spkX>）",
    )
    parser.add_argument(
        "--hyp_jsonl",
        type=str,
        default="/pengjing/workspace/project/west/examples/aishell/asr/exp/firered_from_ziyi_all_projector_lora/checkpoint-4000/alimeeting_test_result.jsonl",
        help="hypothesis jsonl，包含 txt（带 <spkX>）",
    )
    parser.add_argument(
        "--out_ref_jsonl",
        type=str,
        default="/pengjing/workspace/nfs/tmp/out/ref.jsonl",
        help="输出 ref_clean.jsonl 路径",
    )
    parser.add_argument(
        "--out_hyp_jsonl",
        type=str,
        default="/pengjing/workspace/nfs/tmp/out/hyp.jsonl",
        help="输出 hyp_clean.jsonl 路径",
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

    with open(args.out_ref_jsonl, "w", encoding="utf-8") as f_ref_out, \
         open(args.out_hyp_jsonl, "w", encoding="utf-8") as f_hyp_out:

        for i in range(n):
            ref_item = ref_lines[i]
            hyp_item = hyp_lines[i]

            ref_txt_raw = ref_item.get("txt", "")
            hyp_txt_raw = hyp_item.get("txt", "")

            ref_clean = clean_text(ref_txt_raw)
            hyp_clean = clean_text(hyp_txt_raw)

            # 保留 wav 方便对齐，也可以按需再加其它字段
            ref_out = {
                "wav": ref_item.get("wav", ""),
                "txt": ref_clean,
            }
            hyp_out = {
                "wav": ref_item.get("wav", ""),  # 用 ref 的 wav 对齐
                "txt": hyp_clean,
            }

            f_ref_out.write(json.dumps(ref_out, ensure_ascii=False) + "\n")
            f_hyp_out.write(json.dumps(hyp_out, ensure_ascii=False) + "\n")

    print(f"[Done] 写入 ref_clean.jsonl: {args.out_ref_jsonl}")
    print(f"[Done] 写入 hyp_clean.jsonl: {args.out_hyp_jsonl}")


if __name__ == "__main__":
    main()
