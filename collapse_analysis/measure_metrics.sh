dir='/mnt/nas/hoyeon/collapse_analysis'
# step=2500
# step=31250
step=75000

python measure_metrics.py \
    --id_file ${dir}/id_${step}_dedup.json \
    --ood_file ${dir}/ood_${step}_dedup.json \
    --nonsense_file ${dir}/nonsense_${step}_dedup.json \
    --layer 5 \
    --output_dir /mnt/nas/hoyeon/collapse_analysis/results/step${step}
