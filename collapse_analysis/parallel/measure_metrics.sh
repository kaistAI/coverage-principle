
ATOMIC_IDX=1
DATASET=parallel2hop.50.100000.diff-f123.inf
DIR=/mnt/nas/hoyeon/GrokkedTransformer/collapse_analysis/${DATASET}
LAYER=1
POS=1

for POS in 1
do
    for STEP in 90000
    do
        for LAYER in 1 2 3 4 5 6 7
        do
            CUDA_VISIBLE_DEVICES=3 python measure_metrics.py \
                --id_train_file "${DIR}/(${LAYER},${POS})/${STEP}/id_train_dedup.json" \
                --id_test_file "${DIR}/(${LAYER},${POS})/${STEP}/id_test_dedup.json" \
                --ood_file "${DIR}/(${LAYER},${POS})/${STEP}/ood_dedup.json" \
                --output_dir "/mnt/sda/hoyeon/GrokkedTransformer/collapse_analysis/results/${DATASET}/${ATOMIC_IDX}/(${LAYER},${POS})/step${STEP}" \
                --save_plots \
                --pca_vis \
                --reduce_method "tsne"
        done
    done
done