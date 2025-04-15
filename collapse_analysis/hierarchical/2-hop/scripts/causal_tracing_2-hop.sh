#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../causal_tracing_2-hop.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.30.9000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42 \
    --step_list 250 final_checkpoint \
    --batch_size 4096

CUDA_VISIBLE_DEVICES=0 python ../causal_tracing_2-hop.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.70.50000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42 \
    --step_list 500 final_checkpoint \
    --batch_size 4096

CUDA_VISIBLE_DEVICES=0 python ../causal_tracing_2-hop.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.100.100000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42 \
    --step_list 750 1750 \
    --batch_size 4096

CUDA_VISIBLE_DEVICES=0 python ../causal_tracing_2-hop.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.150.270000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42 \
    --step_list 1750 final_checkpoint \
    --batch_size 4096