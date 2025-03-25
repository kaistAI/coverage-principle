#!/bin/bash
# STEP=250
# CHECKPOINT_DIR="/mnt/nas/jinho/GrokkedTransformer/trained_checkpoints/composition.2000.200.inf-controlled_wd-0.1_layer-8_head-12_seed-42/checkpoint-${STEP}"

atomic_idx=1
# layer=$2
# pos=$3
key=${layer}${pos}
mode=residual

exp_name=nontree.50.10000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42
DIR=/mnt/nas/hoyeon/GrokkedTransformer/collapse_analysis/${mode}/nontree.50.10000.diff-f12.inf/f${atomic_idx}

for pos in 1 2
do
    for layer in 1 2 3 4 5 6 7 8
    do
        for step in 7500
        do
            CHECKPOINT_DIR="/mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/${exp_name}/checkpoint-${step}"
            CUDA_VISIBLE_DEVICES=3 python collapse_analysis.py \
                --ckpt ${CHECKPOINT_DIR}/ \
                --layer_pos_pairs "[(${layer},${pos})]" \
                --save_dir /mnt/nas/hoyeon/GrokkedTransformer/collapse_analysis \
                --base_dir /mnt/sda/hoyeon/GrokkedTransformer \
                --atomic_idx ${atomic_idx} \
                --mode ${mode}

            CUDA_VISIBLE_DEVICES=3 python measure_metrics.py \
                --id_train_file ${DIR}/\(${layer},${pos}\)/${step}/id_train_dedup.json \
                --id_test_file ${DIR}/\(${layer},${pos}\)/${step}/id_test_dedup.json \
                --ood_file ${DIR}/\(${layer},${pos}\)/${step}/ood_dedup.json \
                --output_dir "/mnt/sda/hoyeon/GrokkedTransformer/collapse_analysis/results/${exp_name}/${mode}/${atomic_idx}/(${layer},${pos})/step${step}" \
                --pca_vis \
                --reduce_dim 2 \
                --pca_scope global \
                --reduce_method tsne
        done
    done
done


/mnt/nas/hoyeon/GrokkedTransformer/collapse_analysis/residual/nontree.50.10000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42/
/mnt/nas/hoyeon/GrokkedTransformer/collapse_analysis/residual/nontree.50.10000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42/