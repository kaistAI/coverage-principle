step=2000

python collapse_analysis.py \
    --ckpt /mnt/sda/hoyeon/GrokkedTransformer/checkpoints/composition/composition.2000.200.9.0_0.1_8/checkpoint-${step}/ \
    --train_dataset data/composition.2000.200.9.0/train.json \
    --test_dataset data/composition.2000.200.9.0/test.json \
    --valid_dataset data/composition.2000.200.9.0/valid.json \
    --layer_pos_pairs "[(5,1)]" \
    --save_dir collapse_analysis \
    --id_save_fname id_${step}.json \
    --ood_save_fname ood_${step}.json