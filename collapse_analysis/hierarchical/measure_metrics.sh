atomic_idx=2
layer=4
pos=2

key=${layer}${pos}
mode=residual
DIR=/mnt/nas/hoyeon/GrokkedTransformer/collapse_analysis/threehop/${mode}/${atomic_idx}/threehop

for step in 170000
do
    CUDA_VISIBLE_DEVICES= python measure_metrics.py \
        --id_train_file ${DIR}/${step}/${key}/id_train_dedup.json \
        --id_test_file ${DIR}/${step}/${key}/id_test_dedup.json \
        --ood_file ${DIR}/${step}/${key}/ood_dedup.json \
        --output_dir "/mnt/sda/hoyeon/GrokkedTransformer/collapse_analysis/results/threehop/${mode}/${atomic_idx}/(${layer},${pos})/step${step}" \
        --save_plots
done