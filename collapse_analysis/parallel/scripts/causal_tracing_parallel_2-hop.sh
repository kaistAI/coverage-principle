#!/bin/bash

# 스크립트가 있는 디렉토리로 이동
cd "$(dirname "$0")"

CURRENT_DIR=$(pwd)
cd "$(dirname "$CURRENT_DIR")"

CUDA_VISIBLE_DEVICES=3 python causal_tracing_parallel_2-hop.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/parallel2hop.50.60000.diff-f123.inf_wd-0.1_layer-8_head-12_seed-42 \
    --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
    --step_list 500 final_checkpoint \
    --atomic_idx 1 \
    --batch_size 4096

CUDA_VISIBLE_DEVICES=3 python causal_tracing_parallel_2-hop.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/parallel2hop.50.60000.diff-f123.inf_wd-0.1_layer-8_head-12_seed-42 \
    --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
    --step_list 500 final_checkpoint \
    --atomic_idx 2 \
    --batch_size 4096