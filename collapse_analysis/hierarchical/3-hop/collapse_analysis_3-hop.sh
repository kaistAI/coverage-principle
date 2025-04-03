#!/bin/bash

MODE=residual

for ATOMIC_IDX in 1 2 3
do
    for POS in 1 2 3
    do
        for STEP in 2250 80000 190000
        do
            for LAYER in 1 2 3 4 5 6 7 8
            do
                CHECKPOINT_DIR="/mnt/nas/jinho/GrokkedTransformer/trained_checkpoints/threehop.70.diff-f123.inf_no-dense_wd-0.1_layer-8_head-12_seed-42/checkpoint-${STEP}"
                CUDA_VISIBLE_DEVICES=0 python collapse_analysis_3-hop.py \
                    --ckpt ${CHECKPOINT_DIR}/ \
                    --layer_pos_pairs "[(${LAYER},${POS})]" \
                    --save_dir /mnt/nas/jinho/GrokkedTransformer/collapse_analysis/3-hop/ \
                    --atomic_idx ${ATOMIC_IDX} \
                    --mode ${MODE}
            done
        done
    done
done