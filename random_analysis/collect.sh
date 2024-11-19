step=400000
base_dir='/mnt/sda/hoyeon/GrokkedTransformer'

python collect.py \
    --ckpt ${base_dir}/checkpoints/composition/composition.2000.200.9.0_0.1_8/checkpoint-${step}/ \
    --dataset ${base_dir}/data/composition.2000.200.9.0/test_wid.json \
    --layer_pos_pairs "[(5,1)]" \
    --save_dir ${base_dir}/results/raw/5-1 \
    --save_fname composition_dedup_${step}.json \
    --device cuda:0