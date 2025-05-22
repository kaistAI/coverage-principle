#!/bin/bash

# CUDA_VISIBLE_DEVICES=1 python ../causal_tracing_2-hop_coverage.py \
#     --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.50.10000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42_extended \
#     --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
#     --step_list 250 5000 10000 25000 50000 \
#     --batch_size 2048 \
#     --detailed_grouping 1 \
#     --metric_type "rank"

CUDA_VISIBLE_DEVICES=3 python ../causal_tracing_2-hop_coverage.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.50.10000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42_extended \
    --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
    --step_list 250 5000 10000 25000 50000 \
    --batch_size 2048 \
    --detailed_grouping 2 \
    --metric_type "rank"

# CUDA_VISIBLE_DEVICES=1 python ../causal_tracing_2-hop_coverage.py \
#     --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.50.10000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42_extended \
#     --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
#     --step_list 250 5000 10000 25000 50000 \
#     --batch_size 2048 \
#     --detailed_grouping 3 \
#     --metric_type "rank"

# CUDA_VISIBLE_DEVICES=1 python ../causal_tracing_2-hop_coverage.py \
#     --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.50.10000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42_extended \
#     --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
#     --step_list 250 5000 10000 25000 50000 \
#     --batch_size 2048 \
#     --detailed_grouping 1 \
#     --metric_type "prob"

CUDA_VISIBLE_DEVICES=3 python ../causal_tracing_2-hop_coverage.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.50.10000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42_extended \
    --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
    --step_list 250 5000 10000 25000 50000 \
    --batch_size 2048 \
    --detailed_grouping 2 \
    --metric_type "prob"

# CUDA_VISIBLE_DEVICES=1 python ../causal_tracing_2-hop_coverage.py \
#     --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.50.10000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42_extended \
#     --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
#     --step_list 250 5000 10000 25000 50000 \
#     --batch_size 2048 \
#     --detailed_grouping 3 \
#     --metric_type "prob"


