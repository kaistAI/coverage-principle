step=400000
base_dir='/mnt/sda/hoyeon/GrokkedTransformer'

python mi_collect_vectors.py \
    --ckpt ${base_dir}/checkpoints/composition/composition.2000.200.9.0_0.1_8/checkpoint-${step}/ \
    --dataset ${base_dir}/data/composition.2000.200.9.0/test.json \
    --save_dir ${base_dir}/analysis/mi_results/raw/ \
    --save_fname composition_dedup_${step}_debug.json \
    --device cuda:0 \
    --debug