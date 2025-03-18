#!/bin/bash
# STEP=250
# CHECKPOINT_DIR="/mnt/nas/jinho/GrokkedTransformer/trained_checkpoints/composition.2000.200.inf-controlled_wd-0.1_layer-8_head-12_seed-42/checkpoint-${STEP}"

ATOMIC_IDX=1
DATASET=parallel2hop.50.100000.diff-f123.inf
MODE=residual

for POS in 1
do
    for STEP in 90000
    do
        for LAYER in 1 2 3 4 5 6 7
        do
            CHECKPOINT_DIR="/mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/${DATASET}_wd-0.1_layer-8_head-12_seed-42/checkpoint-${STEP}"
            CUDA_VISIBLE_DEVICES=3 python collapse_analysis.py \
                --ckpt ${CHECKPOINT_DIR}/ \
                --layer_pos_pairs "[(${LAYER},${POS})]" \
                --save_dir "/mnt/nas/hoyeon/GrokkedTransformer/collapse_analysis/${DATASET}/(${LAYER},${POS})" \
                --atomic_idx ${ATOMIC_IDX} \
                --data_dir "/mnt/sda/hoyeon/GrokkedTransformer/data/${DATASET}"
        done
    done
done
    # --id_train_save_fname id_train_${step}_dedup.json \
    # --id_test_save_fname id_test_${step}_dedup.json \
    # --ood_save_fname ood_${step}_dedup.json \
    # --nonsense_save_fname nonsense_${step}_dedup.json