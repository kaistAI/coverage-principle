step=400000

python analysis/collect_vectors.py \
    --ckpt checkpoints/composition/composition.2000.200.9.0_0.1_8/checkpoint-${step}/ \
    --dataset data/composition.2000.200.9.0/test.json \
    --layer_pos_pairs "[(0,0), (0,1), (5,0), (5,1), (5,2), (8,0), (8,1), (8,2)]" \
    --save_dir analysis/results/raw/ \
    --save_fname composition_dedup_${step}.json \
    --device cuda:0