#!/bin/bash

# 스크립트가 있는 디렉토리로 이동
cd "$(dirname "$0")"

CURRENT_DIR=$(pwd)
cd "$(dirname "$CURRENT_DIR")"

CUDA_VISIBLE_DEVICES=3 python ../causal_tracing_2-hop.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.70.50000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42 \
    --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
    --step_list final_checkpoint \
    --batch_size 2048 \
    --metric_type "rank"

CUDA_VISIBLE_DEVICES=3 python ../causal_tracing_2-hop.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.70.50000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42 \
    --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
    --step_list final_checkpoint \
    --batch_size 2048 \
    --metric_type "prob"

CUDA_VISIBLE_DEVICES=3 python ../causal_tracing_2-hop.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.100.100000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42 \
    --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
    --step_list 1750 \
    --batch_size 2048 \
    --metric_type "rank"

CUDA_VISIBLE_DEVICES=3 python ../causal_tracing_2-hop.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.100.100000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42 \
    --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
    --step_list 1750 \
    --batch_size 2048 \
    --metric_type "prob"

CUDA_VISIBLE_DEVICES=3 python ../causal_tracing_2-hop.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.150.300000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42 \
    --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
    --step_list final_checkpoint \
    --batch_size 2048 \
    --metric_type "rank"

CUDA_VISIBLE_DEVICES=3 python ../causal_tracing_2-hop.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.150.300000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42 \
    --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
    --step_list final_checkpoint \
    --batch_size 2048 \
    --metric_type "prob"

CUDA_VISIBLE_DEVICES=3 python ../causal_tracing_2-hop.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.200.600000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42 \
    --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
    --step_list final_checkpoint \
    --batch_size 2048 \
    --metric_type "rank"

CUDA_VISIBLE_DEVICES=3 python ../causal_tracing_2-hop.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.200.600000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42 \
    --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
    --step_list final_checkpoint \
    --batch_size 2048 \
    --metric_type "prob"