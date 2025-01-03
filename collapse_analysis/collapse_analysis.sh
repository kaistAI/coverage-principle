step=75000
CHECKPOINT_DIR="/mnt/nas/jinho/GrokkedTransformer/trained_checkpoints/composition.2000.200.9.0-controlled_wd-0.1_layer-8_head-12_seed-42/dense_checkpoint_3/checkpoint-${step}"

python collapse_analysis.py \
    --base_dir /home/hoyeon/GrokkedTransformer \
    --ckpt ${CHECKPOINT_DIR}/ \
    --train_dataset data/composition.2000.200.9.0/train.json \
    --test_dataset data/composition.2000.200.inf-controlled/test.json \
    --valid_dataset data/composition.2000.200.inf-controlled/valid.json \
    --layer_pos_pairs "[(5,1)]" \
    --save_dir /mnt/nas/hoyeon/collapse_analysis/w_atomic \
    --id_save_fname id_${step}_dedup.json \
    --ood_save_fname ood_${step}_dedup.json \
    --nonsense_save_fname nonsense_${step}_dedup.json