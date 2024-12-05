NUM=1
BASE_DIR="/mnt/sda/hoyeon/GrokkedTransformer"
CHECKPOINT_DIR="/mnt/nas/jinho/GrokkedTransformer/trained_checkpoints/composition.2000.200.inf-controlled_wd-0.1_layer-8_head-12_seed-42/dense_checkpoint_${NUM}"
DATASET_FILE="data/composition.2000.200.inf-controlled/test.json"
LAYER_POS_PAIRS="[(5,1)]"
EXP_NAME="wo_atomic/dense_${NUM}"

python parallel_collect.py \
    --base_dir ${BASE_DIR} \
    --checkpoints_dir ${CHECKPOINT_DIR} \
    --layer_pos_pairs ${LAYER_POS_PAIRS} \
    --save_dir_base ${BASE_DIR}/random_analysis/results/raw/${EXP_NAME} \
    --save_fname_pattern "composition_{}.json" \
    --dataset ${DATASET_FILE}