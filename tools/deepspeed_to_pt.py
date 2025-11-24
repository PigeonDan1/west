#!/usr/bin/env python3
"""
用法
python ds_ckpt_to_pt.py  checkpoint-12345  [--tag 12345]  -o  consolidated.ckpt.pt
"""
import argparse
import os
import torch
import pathlib
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint #get_model_state_file, load_state_dict_from_zero_checkpoint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir",default="/pengjing/workspace/project/west/examples/aishell/asr/exp/firered_stage2_from_ziyi_ckpt_7500_all_projector_lora/checkpoint-500", type=pathlib.Path,
                        help="DeepSpeed checkpoint 目录（含 optimizer 子目录）")
    parser.add_argument("-t", "--tag", default=None,
                        help="checkpoint tag，默认用 ckpt_dir 名字后的数字")
    parser.add_argument("-o", "--output", default="/pengjing/workspace/nfs/tmp/ckpt/trained/model.pt",
                        help="输出普通 pt 文件名")
    return parser.parse_args()

def main():
    args = parse_args()
    ckpt_dir = args.ckpt_dir.expanduser().resolve()

    # # 1. 自动推断 tag
    # if args.tag is None:
    #     # 假设目录名是 checkpoint-12345
    #     tag = ckpt_dir.name.split("-")[-1]
    # else:
    #     tag = args.tag

    tag = None

    # 2. 用 DeepSpeed 官方工具把 zero 拆片还原成完整 FP32 模型权重
    print(">>> 正在把 ZeRO 分片还原成完整模型权重 ...")
    # model_state_dict = load_state_dict_from_zero_checkpoint(
    #     model=None,  # 不给模型也能直接返回 state_dict
    #     checkpoint_dir=str(ckpt_dir),
    #     tag=tag
    # )
    model_state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir, tag)

    # # 3. （可选）把当时手工保存的 frozen 片段也合并进来
    # #    如果你保存时把 FROZEN_PARAM_FRAGMENTS 写进了 optim 文件，
    # #    它会出现在 client_state 里；这里我们把它重新覆盖回模型。
    # optim_files = list(ckpt_dir.glob(f"*/zero_pp_rank_*_optim_states.pt"))
    # if optim_files:
    #     client_state = torch.load(optim_files[0], map_location="cpu").get("client_state", {})
    #     frozen = client_state.get("FROZEN_PARAM_FRAGMENTS", {})
    #     if frozen:
    #         print(">>> 合并 frozen 参数片段 ...")
    #         for name, param in frozen.items():
    #             model_state_dict[name] = param

    # 4. 保存成普通 pt
    out_file = args.output
    out_file = os.path.join(
        ckpt_dir.parent,
        out_file
    )
    torch.save(model_state_dict, out_file)
    print(f">>> 已保存完整状态字典 -> {out_file}")
    # print(f"    共 {len(model_state_dict)} 个键，总大小 {model_state_dict.element_size()*sum(v.numel() for v in model_state_dict.values())/1024**2:.1f} MB")

if __name__ == "__main__":
    main()