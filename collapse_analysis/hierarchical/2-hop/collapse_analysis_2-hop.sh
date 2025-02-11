#!/bin/bash

for step in 250 3500 30000 300000
do
    CHECKPOINT_DIR="/mnt/nas/jinho/GrokkedTransformer/trained_checkpoints/composition.2000.200.inf-controlled_wd-0.1_layer-8_head-12_seed-42/checkpoint-${step}"
    python collapse_analysis.py \
        --ckpt ${CHECKPOINT_DIR}/ \
        --layer_pos_pairs "[(5,1)]" \
        --save_dir /mnt/nas/jinho/GrokkedTransformer/collapse_analysis \
        --merge_id_data
done