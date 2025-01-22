DIR='/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/composition.2000.200.inf-controlled/test/(5,1)'

for step in 250 3500 30000 300000
do
    python measure_metrics_copy.py \
        --id_train_file ${DIR}/${step}/id_dedup.json \
        --ood_file ${DIR}/${step}/ood_dedup.json \
        --nonsense_file ${DIR}/${step}/nonsense_dedup.json \
        --output_dir "/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/results/composition.2000.200.inf-controlled/test/(5,1)/step${step}"
done