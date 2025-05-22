#!/bin/bash
# STEP=250
# CHECKPOINT_DIR="/mnt/nas/jinho/GrokkedTransformer/trained_checkpoints/composition.2000.200.inf-controlled_wd-0.1_layer-8_head-12_seed-42/checkpoint-${STEP}"

# atomic_idx=
# layer=$2
# pos=$3
key=${layer}${pos}
mode=residual

exp_name=twohop.50.10000.diff-f12.cot_wd-0.1_layer-8_head-12_seed-42


for atomic_idx in 2
do
    DIR=/mnt/nas/hoyeon/GrokkedTransformer/collapse_analysis/${mode}/twohop.50.10000.diff-f12.cot/f${atomic_idx}
    for pos in 3
    do
        for layer in 1
        do
            for step in final_checkpoint
            do
                CHECKPOINT_DIR="/mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/${exp_name}/final_checkpoint"
                CUDA_VISIBLE_DEVICES=0 python collapse_analysis.py \
                    --ckpt ${CHECKPOINT_DIR}/ \
                    --layer_pos_pairs "[(${layer},${pos})]" \
                    --save_dir /mnt/nas/hoyeon/GrokkedTransformer/collapse_analysis \
                    --base_dir /mnt/sda/hoyeon/GrokkedTransformer \
                    --atomic_idx ${atomic_idx} \
                    --mode ${mode}

                CUDA_VISIBLE_DEVICES=0 python measure_metrics.py \
                    --id_train_file ${DIR}/\(${layer},${pos}\)/${step}/id_train_dedup.json \
                    --id_test_file ${DIR}/\(${layer},${pos}\)/${step}/id_test_dedup.json \
                    --ood_file ${DIR}/\(${layer},${pos}\)/${step}/ood_dedup.json \
                    --output_dir "/mnt/sda/hoyeon/GrokkedTransformer/collapse_analysis/results/${exp_name}/${mode}/${atomic_idx}/(${layer},${pos})/step${step}" \
                    --pca_vis \
                    --reduce_dim 2 \
                    --pca_scope global \
                    --reduce_method pca \
                    --save_plots
            done
        done
    done
done
