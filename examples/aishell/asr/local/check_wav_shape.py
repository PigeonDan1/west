# import torchaudio
# import json
# import tqdm

# if __name__ == "__main__":
#     jsonl_data = "train.jsonl"
#     with open(jsonl_data, "r", encoding="utf-8") as f:
#         lines = f.readlines()
#         breakpoint()
#         for line in tqdm.tqdm(lines):
#             meta = json.loads(line.strip())
#             wav_path = meta["wav"]
#             waveform, sample_rate = torchaudio.load(wav_path)
#             if waveform.shape[0] != 1:
#                 print(f"Warning: {wav_path} has {waveform.shape[0]} channels.")
#             # print(waveform.shape)

import argparse
import datetime
import torchaudio
import json
import tqdm
from concurrent.futures import ProcessPoolExecutor
import os

# -------------- 1. 纯函数：单个样本检查 --------------
def check_one(line: str):
    """
    返回 (wav_path, n_channels, error_string)
    如果一切正常 error_string 为 None
    """
    try:
        meta = json.loads(line.strip())
        wav_path = meta["wav"]
        waveform, sr = torchaudio.load(wav_path)
        return wav_path, tuple(waveform.shape), None
    except Exception as e:
        return meta.get("wav", "???"), -1, str(e)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check wav shape in jsonl file."
    )
    parser.add_argument(
        "-i", "--input_jsonl_data",
        type=str,
        required=True,
        help="Path to the input jsonl file.",
    )
    return parser.parse_args()

# -------------- 2. 多进程主入口 --------------
if __name__ == "__main__":
    args = get_args()
    jsonl_data = args.input_jsonl_data
    num_workers = min(16, os.cpu_count())          # 可按机器调整

    check_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_log = os.path.join(
        os.path.dirname(jsonl_data),
        f"check_wav_shape.{check_time}.log",
    )

    # 读所有行
    with open(jsonl_data, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 进程池并行 map
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        # imap 保证顺序 + tqdm 可更新
        results = list(
            tqdm.tqdm(
                pool.map(check_one, lines, chunksize=32),
                total=len(lines),
                desc="checking audio",
            )
        )

    # -------------- 3. 事后汇总 --------------
    output_lines = []
    for wav_path, (n_ch, num_samples, *_), err in results:
        if err is not None:
            output_lines.append(f"[Error] {wav_path}: {err}")
        elif num_samples == 0:
            output_lines.append(f"Warning: {wav_path} has {num_samples} samples.")
        elif n_ch != 1:
            output_lines.append(f"Warning: {wav_path} has {n_ch} channels.")

    # 输出结果
    with open(output_log, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")