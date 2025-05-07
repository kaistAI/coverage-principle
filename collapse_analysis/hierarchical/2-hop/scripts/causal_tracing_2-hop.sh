#!/bin/bash

# CUDA_VISIBLE_DEVICES=2 python ../causal_tracing_2-hop.py \
#     --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.50.27000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42 \
#     --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
#     --step_list final_checkpoint \
#     --batch_size 1024

# CUDA_VISIBLE_DEVICES=2 python ../causal_tracing_2-hop.py \
#     --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.70.50000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42 \
#     --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
#     --step_list 750 final_checkpoint \
#     --batch_size 1024

# CUDA_VISIBLE_DEVICES=2 python ../causal_tracing_2-hop.py \
#     --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.100.100000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42 \
#     --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
#     --step_list 1750 \
#     --batch_size 1024

CUDA_VISIBLE_DEVICES=1 python ../causal_tracing_2-hop.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.150.300000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42 \
    --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
    --step_list 1250 final_checkpoint \
    --batch_size 1024

# CUDA_VISIBLE_DEVICES=2 python ../causal_tracing_2-hop.py \
#     --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.200.600000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42 \
#     --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
#     --step_list 1750 final_checkpoint \
#     --batch_size 1024