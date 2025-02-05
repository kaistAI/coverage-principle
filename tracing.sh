task=parallel

CUDA_VISIBLE_DEVICES=3 python causal_tracing_general.py \
    --model_dir /mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/${task}_dense_wd-_layer-_head-_seed-/ \
    --step_list 50000 172500 \
    --save_path causal_tracing/${task}/ \
    --target_layer 8 \
    --data_file dataset_generation/${task}/diff_f123/test.json \
    --type type_0 \
    --max_positions 4


task=reproduce
CUDA_VISIBLE_DEVICES=3 python causal_tracing_general.py \
    --model_dir /mnt/nas/jinho/GrokkedTransformer/trained_checkpoints/composition.2000.200.inf-controlled_wd-0.1_layer-8_head-12_seed-42/ \
    --step_list 5000 150000 \
    --save_path causal_tracing/${task}/ \
    --target_layer 8 \
    --data_file data/composition.2000.200.inf-controlled/test.json \
    --type test_inferred_id \
    --max_positions 3