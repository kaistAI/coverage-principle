#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python ../causal_tracing_parallel_2-hop.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/parallel2hop.50.60000.diff-f123.inf_wd-0.1_layer-8_head-12_seed-42 \
    --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
    --step_list final_checkpoint \
    --atomic_idx 1 \
    --batch_size 4096 \
    --metric_type "rank"

CUDA_VISIBLE_DEVICES=2 python ../causal_tracing_parallel_2-hop.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/parallel2hop.50.60000.diff-f123.inf_wd-0.1_layer-8_head-12_seed-42 \
    --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
    --step_list final_checkpoint \
    --atomic_idx 1 \
    --batch_size 4096 \
    --metric_type "prob"

CUDA_VISIBLE_DEVICES=2 python ../causal_tracing_parallel_2-hop.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/parallel2hop.50.60000.diff-f123.inf_wd-0.1_layer-8_head-12_seed-42 \
    --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
    --step_list final_checkpoint \
    --atomic_idx 2 \
    --batch_size 4096 \
    --metric_type "rank"

CUDA_VISIBLE_DEVICES=2 python ../causal_tracing_parallel_2-hop.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/parallel2hop.50.60000.diff-f123.inf_wd-0.1_layer-8_head-12_seed-42 \
    --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
    --step_list final_checkpoint \
    --atomic_idx 2 \
    --batch_size 4096 \
    --metric_type "prob"
