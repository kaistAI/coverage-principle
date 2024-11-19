step=400000
base_dir='/mnt/sda/hoyeon/GrokkedTransformer'

python lmi.py \
    --model_path ${base_dir}/checkpoints/composition/composition.2000.200.9.0_0.1_8/checkpoint-${step}/ \
    --results_path ${base_dir}/analysis/mi_results/raw/composition_dedup_${step}.json \
    --save_dir ${base_dir}/analysis/mi_results \
    --save_prefix lmi_analysis