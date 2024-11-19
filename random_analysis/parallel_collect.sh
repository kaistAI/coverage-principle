BASE_DIR="/mnt/sda/hoyeon/GrokkedTransformer"
CHECKPOINT_DIR="/mnt/nas/jinho/GrokkedTransformer/trained_checkpoints/composition.2000.200.9.0_wd-0.1_layer-8_head-12_seed-42_trajectory/composition.2000.200.9.0_wd-0.1_layer-8_head-12_seed-42_trajectory/dense_checkpoint_2"
DATASET_FILE="data/composition.2000.200.9.0/test.json"
LAYER_POS_PAIRS="[(5,1)]"
EXP_NAME="reproduce/dense_2"

python parallel_collect.py \
    --base_dir ${BASE_DIR} \
    --checkpoints_dir ${CHECKPOINT_DIR} \
    --layer_pos_pairs ${LAYER_POS_PAIRS} \
    --save_dir_base ${BASE_DIR}/random_analysis/results/raw/${EXP_NAME}