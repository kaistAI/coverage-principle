#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../causal_tracing_3-hop.py \
    --model_dir /mnt/nas/jinho/GrokkedTransformer/trained_checkpoints/threehop.50.200000.diff-f123.inf_wd-0.1_layer-8_head-12_seed-42 \
    --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
    --step_list 1250 final_checkpoint \
    --atomic_idx 1 \
    --batch_size 4096

CUDA_VISIBLE_DEVICES=0 python ../causal_tracing_3-hop.py \
    --model_dir /mnt/nas/jinho/GrokkedTransformer/trained_checkpoints/threehop.50.200000.diff-f123.inf_wd-0.1_layer-8_head-12_seed-42 \
    --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
    --step_list 1250 final_checkpoint \
    --atomic_idx 2 \
    --batch_size 4096