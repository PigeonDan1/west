# Copyright 2025 Binbin Zhang(binbzha@qq.com)
export HF_ENDPOINT="https://hf-mirror.com"
[ ! -s west ] && ln -s ../../../west
[ ! -s tools ] && ln -s ../../../tools
export PYTHONPATH=$PYTHONPATH:$PWD
# Change this to all your available gpus, such as "0,1,2,3"
export CUDA_VISIBLE_DEVICES="0,1,2"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

set -euo pipefail

stage=train # data/train/decode
data=data
dir=exp/firered_from_ziyi_stage2_all_projector_lora #stage1-asr_projector_frozen-from_ziyi_4000_steps
steps=10000 #5000  # training steps

# Available model_conf are in conf, such as:
# conf/qwen2-7b_firered.json
# conf/qwen3-1.7b-lora_firered.json
# conf/qwen2-7b-lora_paraformer.json
# conf/qwen2-1.5b-lora_whisper-large-v3-turbo.json
model_conf="conf/qwen2-7b_firered_sa_asr_projector_lora.jsonl" #conf/qwen2-7b_firered.json
decode_conf=conf/generation_config.json

. tools/parse_options.sh || exit 1;

if [ $stage == "data" ] || [ $stage == "all" ]; then
    echo ${stage}
    
    echo "Prepare required data"
fi

if [ $stage == "train" ] || [ $stage == "all" ]; then
    echo ${stage}

    # pack_size 16384
    torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus west/bin/train.py \
        --model_config_or_dir $model_conf \
        --data_path $data/train.clean.jsonl \
        --output_dir $dir \
        --pack_size 6000 \
        --bf16 True \
        --max_steps $steps \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --save_strategy "steps" \
        --save_steps 1000 \
        --save_total_limit 100 \
        --learning_rate 5e-5 \
        --weight_decay 0.01 \
        --adam_beta2 0.95 \
        --warmup_ratio 0.5 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --report_to "tensorboard" \
        --gradient_checkpointing \
        --dataloader_num_workers 2 \
        --dataloader_prefetch_factor 10 \
        --ignore_data_skip True \
        --deepspeed conf/ds_config_zero2.json \
        --accelerator_config conf/accelerator_config.json
fi


if [ $stage == "decode" ] || [ $stage == "all" ]; then
    echo ${stage}
    # mdir=exp/Qwen2-7B-Instruct-firered/debug/aishell_mlc_data-stage1-asr_projector_trainable-from_ziyi_4000_steps/checkpoint-600
    # cp $decode_conf $mdir
    # python west/bin/decode.py \
    #     --data_path $data/test.jsonl \
    #     --model_dir $mdir \
    #     --result_path $mdir/result.jsonl
    # python tools/compute_wer.py --char=1 --v=1 \
    #     $data/test.jsonl $mdir/result.jsonl > $mdir/result.wer

    # python west/bin/decode.py \
    #     --data_path $data/mlc_dev_freshed.jsonl \
    #     --model_dir $mdir \
    #     --result_path $mdir/mlc_dev_result.jsonl
    # python tools/compute_wer.py --char=1 --v=1 \
    #     $data/mlc_dev_freshed.jsonl $mdir/mlc_dev_result.jsonl > $mdir/mlc_dev_result.wer

    # python west/bin/decode.py \
    #     --data_path $data/test.jsonl \
    #     --model_dir $mdir \
    #     --result_path $mdir/train_head_100_result.jsonl
    # python tools/compute_wer.py --char=1 --v=1 \
    #     $data/test.jsonl $mdir/train_head_100_result.jsonl > $mdir/train_head_100_result.wer

    # python west/bin/decode.py \
    #     --data_path $data/test.jsonl \
    #     --model_dir $mdir \
    #     --result_path $mdir/train_head_1_duplicated_result.jsonl
    # python tools/compute_wer.py --char=1 --v=1 \
    #     $data/test.jsonl $mdir/train_head_1_duplicated_result.jsonl > $mdir/train_head_1_duplicated_result.wer

    mdir=/pengjing/workspace/project/west/examples/aishell/asr/exp/firered_from_ziyi_all_projector_lora/checkpoint-4000
    cp $decode_conf $mdir
    python west/bin/decode.py \
        --data_path $data/test.haoyu.jsonl \
        --model_dir $mdir \
        --result_path $mdir/train_test_result.jsonl
    # python tools/compute_wer.py --char=1 --v=1 \
    #     $data/test.haoyu.jsonl $mdir/aishell4_train_head20_result.jsonl > $mdir/aishell4_train_head20_result.wer

fi
