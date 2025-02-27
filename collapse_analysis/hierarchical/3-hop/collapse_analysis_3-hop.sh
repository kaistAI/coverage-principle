#!/bin/bash

ATOMIC_IDX=1
# layer=7
# POS=1
key=${layer}${pos}
MODE=residual

for POS in 1
do
    for STEP in 2250 80000 190000
    do
        for LAYER in 1 2 3 4 5 6 7
        do
            CHECKPOINT_DIR="/mnt/nas/jinho/GrokkedTransformer/trained_checkpoints/threehop.70.diff-f123.inf_no-dense_wd-0.1_layer-8_head-12_seed-42/checkpoint-${STEP}"
            CUDA_VISIBLE_DEVICES=1 python collapse_analysis_3-hop_revised.py \
                --ckpt ${CHECKPOINT_DIR}/ \
                --layer_pos_pairs "[(${LAYER},${POS})]" \
                --save_dir /mnt/nas/jinho/GrokkedTransformer/collapse_analysis/3-hop/ \
                --atomic_idx ${ATOMIC_IDX} \
                --mode ${MODE}
        done
    done
done

# mkdir -p /mnt/nas/hoyeon/GrokkedTransformer/collapse_analysis/threehop/${mode}/${atomic_idx}/threehop/170000/${key}
# mv /mnt/nas/hoyeon/GrokkedTransformer/collapse_analysis/threehop/${mode}/${atomic_idx}/threehop/170000/*.json /mnt/nas/hoyeon/GrokkedTransformer/collapse_analysis/threehop/${mode}/${atomic_idx}/threehop/170000/${key}

# for step in 170000
# do
#     CUDA_VISIBLE_DEVICES= python measure_metrics.py \
#         --id_train_file ${DIR}/${step}/${key}/id_train_dedup.json \
#         --id_test_file ${DIR}/${step}/${key}/id_test_dedup.json \
#         --ood_file ${DIR}/${step}/${key}/ood_dedup.json \
#         --output_dir "/mnt/sda/hoyeon/GrokkedTransformer/collapse_analysis/results/threehop/${mode}/${atomic_idx}/(${layer},${pos})/step${step}" \
#         --save_plots
# done