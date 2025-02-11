atomic_idx=2

DIR=/mnt/nas/hoyeon/GrokkedTransformer/collapse_analysis/parallel/${atomic_idx}/parallel

for step in 170000
do
    python measure_metrics.py \
        --id_train_file ${DIR}/${step}/id_train_dedup.json \
        --id_test_file ${DIR}/${step}/id_test_dedup.json \
        --ood_file ${DIR}/${step}/ood_dedup.json \
        --output_dir "/mnt/sda/hoyeon/GrokkedTransformer/collapse_analysis/results/parallel/${atomic_idx}/(2,3)/step${step}"
done