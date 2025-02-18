#!/bin/bash
# STEP=250
# CHECKPOINT_DIR="/mnt/nas/jinho/GrokkedTransformer/trained_checkpoints/composition.2000.200.inf-controlled_wd-0.1_layer-8_head-12_seed-42/checkpoint-${STEP}"

atomic_idx=3
layer=4
pos=3
key=${layer}${pos}
mode=residual
DIR=/mnt/nas/hoyeon/GrokkedTransformer/collapse_analysis/threehop/${mode}/${atomic_idx}/threehop

# for step in 250
for step in 170000
do
    CHECKPOINT_DIR="/mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/threehop_dense_wd-_layer-_head-_seed-/checkpoint-${step}"
    CUDA_VISIBLE_DEVICES=0 python collapse_analysis.py \
        --ckpt ${CHECKPOINT_DIR}/ \
        --layer_pos_pairs "[(${layer},${pos})]" \
        --save_dir /mnt/nas/hoyeon/GrokkedTransformer/collapse_analysis/threehop/${mode}/${atomic_idx} \
        --atomic_idx ${atomic_idx} \
        --data_dir "/mnt/sda/hoyeon/GrokkedTransformer/dataset_generation/threehop/diff_f123" \
        --mode ${mode}
done

mkdir -p /mnt/nas/hoyeon/GrokkedTransformer/collapse_analysis/threehop/${mode}/${atomic_idx}/threehop/170000/${key}
mv /mnt/nas/hoyeon/GrokkedTransformer/collapse_analysis/threehop/${mode}/${atomic_idx}/threehop/170000/*.json /mnt/nas/hoyeon/GrokkedTransformer/collapse_analysis/threehop/${mode}/${atomic_idx}/threehop/170000/${key}

for step in 170000
do
    CUDA_VISIBLE_DEVICES= python measure_metrics.py \
        --id_train_file ${DIR}/${step}/${key}/id_train_dedup.json \
        --id_test_file ${DIR}/${step}/${key}/id_test_dedup.json \
        --ood_file ${DIR}/${step}/${key}/ood_dedup.json \
        --output_dir "/mnt/sda/hoyeon/GrokkedTransformer/collapse_analysis/results/threehop/${mode}/${atomic_idx}/(${layer},${pos})/step${step}" \
        --pca_vis \
        --reduce_dim 3 \
        --pca_scope global \
        --reduce_method tsne
done