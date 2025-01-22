#!/bin/bash
# STEP=250
# CHECKPOINT_DIR="/mnt/nas/jinho/GrokkedTransformer/trained_checkpoints/composition.2000.200.inf-controlled_wd-0.1_layer-8_head-12_seed-42/checkpoint-${STEP}"

# for step in 250
for step in 250 3500 30000 300000
do
    CHECKPOINT_DIR="/mnt/nas/jinho/GrokkedTransformer/trained_checkpoints/composition.2000.200.inf-controlled_wd-0.1_layer-8_head-12_seed-42/checkpoint-${step}"
    python collapse_analysis_copy.py \
        --ckpt ${CHECKPOINT_DIR}/ \
        --layer_pos_pairs "[(5,1)]" \
        --save_dir /mnt/nas/jinho/GrokkedTransformer/collapse_analysis
done
    # --id_train_save_fname id_train_${step}_dedup.json \
    # --id_test_save_fname id_test_${step}_dedup.json \
    # --ood_save_fname ood_${step}_dedup.json \
    # --nonsense_save_fname nonsense_${step}_dedup.json