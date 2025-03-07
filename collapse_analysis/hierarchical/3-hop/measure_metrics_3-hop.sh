ATOMIC_IDX=3
layer=8
pos=3

KEY=${layer}${pos}
DEDUP_DIR=/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/3-hop/residual/threehop.70.diff-f123.inf
DATASET="${DEDUP_DIR##*/}"
MODE="${DEDUP_DIR%/*}"
MODE="${MODE##*/}"

for POS in 3
do
    for STEP in 2250 80000 190000
    do
        for LAYER in 8
        do
            CUDA_VISIBLE_DEVICES=3 python measure_metrics_3-hop.py \
                --id_train_file "${DEDUP_DIR}/f${ATOMIC_IDX}/(${LAYER},${POS})/${STEP}/id_train_dedup.json" \
                --id_test_file "${DEDUP_DIR}/f${ATOMIC_IDX}/(${LAYER},${POS})/${STEP}/id_test_dedup.json" \
                --ood_file "${DEDUP_DIR}/f${ATOMIC_IDX}/(${LAYER},${POS})/${STEP}/ood_dedup.json" \
                --output_dir "/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/results/3-hop/${MODE}/${DATASET}/f${ATOMIC_IDX}/(${LAYER},${POS})/step${STEP}" \
                --pca_vis \
                --reduce_dim 2 \
                --pca_scope global \
                --reduce_method pca
        done
    done
done