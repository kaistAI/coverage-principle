#!/bin/bash

# MODEL_PATH=gpt2

DATASET=./data/$1/
WEIGHT_DECAY=$2
N_LAYERS=$3
N_HEADS=$4
SEED=$5
echo SEED: $SEED
OUTPUT_DIR=/mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/with_duplicate/$1_"wd"-$2_"layer"-$3_"head"-$4_"seed"-$5
MODEL_PATH=/mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/with_duplicate/$1_"wd"-$2_"layer"-$3_"head"-$4_"seed"-$5/$6
# CUDA_VISIBLE_DEVICES=$GPU python main.py \
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 main.py \
--data_dir $DATASET \
--model_name_or_path ${MODEL_PATH} \
--add_tokens \
--n_layer $N_LAYERS \
--n_head $N_HEADS \
--fp16 \
--do_eval \
--do_train \
--evaluate_during_training \
--predict_during_training \
--weight_decay $WEIGHT_DECAY \
--scheduler constant_schedule_with_warmup \
--output_dir $OUTPUT_DIR \
--save_step 999999999 \
--save_step_dense 99999999999 \
--save_step_dense_interval 9999999999999 \
--max_steps 1 \
--manual_seed $SEED \
--train_batch_size 1 \
--eval_batch_size 512 \
--gradient_accumulation_steps 1 \
--learning_rate 0 \
--max_seq_length 6 \
--max_length 6 \
--block_size 6 \
--wandb_project "coverage_principle" \
--wandb_entity "lklab_kaist" \
--wandb_exp [EVAL/small]$1_"wd"$2_"layer"$3_"head"$4_"seed"$5 \
--overwrite_output_dir