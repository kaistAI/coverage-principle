#!/bin/bash

MODE=residual

for POS in 2 3 4
do
    for LAYER in 1 2 3 4 5 6 7 8
    do
        for ATOMIC_IDX in 4 5
        do
            for STEP in 250
            do
                CHECKPOINT_DIR="/mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/threehop.50.20000.diff-f123.cot_wd-0.1_layer-8_head-12_seed-42/final_checkpoint"
                CUDA_VISIBLE_DEVICES=1 python collapse_analysis_3-hop.py \
                    --ckpt ${CHECKPOINT_DIR}/ \
                    --layer_pos_pairs "[(${LAYER},${POS})]" \
                    --save_dir /mnt/nas/jinho/GrokkedTransformer/collapse_analysis/3-hop/ \
                    --atomic_idx ${ATOMIC_IDX} \
                    --mode ${MODE}
            done
        done
    done
done

for POS in 2 3 4
do
    for LAYER in 1 2 3 4 5 6 7 8
    do
        for ATOMIC_IDX in 4 5
        do
            for STEP in 250 10000
            do
                CHECKPOINT_DIR="/mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/threehop.50.150000.diff-f123.cot_wd-0.1_layer-8_head-12_seed-42/checkpoint-${STEP}"
                CUDA_VISIBLE_DEVICES=1 python collapse_analysis_3-hop.py \
                    --ckpt ${CHECKPOINT_DIR}/ \
                    --layer_pos_pairs "[(${LAYER},${POS})]" \
                    --save_dir /mnt/nas/jinho/GrokkedTransformer/collapse_analysis/3-hop/ \
                    --atomic_idx ${ATOMIC_IDX} \
                    --mode ${MODE}
            done
        done
    done
done