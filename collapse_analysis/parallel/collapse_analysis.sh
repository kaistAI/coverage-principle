#!/bin/bash
# STEP=250
# CHECKPOINT_DIR="/mnt/nas/jinho/GrokkedTransformer/trained_checkpoints/composition.2000.200.inf-controlled_wd-0.1_layer-8_head-12_seed-42/checkpoint-${STEP}"

atomic_idx=2

# for step in 250
for step in 170000
do
    CHECKPOINT_DIR="/mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/parallel_dense_wd-_layer-_head-_seed-/checkpoint-${step}"
    python collapse_analysis.py \
        --ckpt ${CHECKPOINT_DIR}/ \
        --layer_pos_pairs "[(2,3)]" \
        --save_dir /mnt/nas/hoyeon/GrokkedTransformer/collapse_analysis/parallel/${atomic_idx} \
        --atomic_idx ${atomic_idx} \
        --data_dir "/mnt/sda/hoyeon/GrokkedTransformer/dataset_generation/parallel/diff_f123"
done
    # --id_train_save_fname id_train_${step}_dedup.json \
    # --id_test_save_fname id_test_${step}_dedup.json \
    # --ood_save_fname ood_${step}_dedup.json \
    # --nonsense_save_fname nonsense_${step}_dedup.json