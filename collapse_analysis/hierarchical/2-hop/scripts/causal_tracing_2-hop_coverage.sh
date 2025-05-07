#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python ../causal_tracing_2-hop_coverage.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.50.10000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42_extended \
    --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
    --step_list 50000 \
    --batch_size 2048
