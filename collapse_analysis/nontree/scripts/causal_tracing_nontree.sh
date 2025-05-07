#!/bin/bash

# 스크립트가 있는 디렉토리로 이동
cd "$(dirname "$0")"

CURRENT_DIR=$(pwd)
cd "$(dirname "$CURRENT_DIR")"

CUDA_VISIBLE_DEVICES=3 python causal_tracing_nontree.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/nontree.50.50000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42 \
    --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
    --step_list 50000 100000 \
    --atomic_idx 1 \
    --batch_size 2048

# CUDA_VISIBLE_DEVICES=2 python causal_tracing_nontree.py \
#     --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/nontree.50.50000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42 \
#     --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
#     --step_list 50000 100000 \
#     --atomic_idx 4 \
#     --batch_size 4096

# CUDA_VISIBLE_DEVICES=2 python causal_tracing_nontree.py \
#     --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/nontree.50.50000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42 \
#     --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
#     --step_list 50000 100000 \
#     --atomic_idx 5 \
#     --batch_size 4096