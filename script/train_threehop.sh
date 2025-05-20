#!/bin/bash

MODEL_PATH=gpt2

DATASET=./data/$1/
WEIGHT_DECAY=$2
N_LAYERS=$3
N_HEADS=$4
GPU=$5
SEED=$6

NUM_GPUS=$(echo "$GPU" | tr -cd '0-9' | wc -c)
OUTPUT_DIR=/mnt/nas/jinho/GrokkedTransformer/trained_checkpoints/$1_"no-dense_wd"-$2_"layer"-$3_"head"-$4_"seed"-$6

# CUDA_VISIBLE_DEVICES=$GPU python main.py \
CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port 12345 main.py \
--data_dir $DATASET \
--model_name_or_path ${MODEL_PATH} \
--init_weights \
--add_tokens \
--n_layer $N_LAYERS \
--n_head $N_HEADS \
--fp16 \
--do_train \
--overwrite_output_dir \
--evaluate_during_training \
--predict_during_training \
--weight_decay $WEIGHT_DECAY \
--scheduler constant_schedule_with_warmup \
--output_dir $OUTPUT_DIR \
--save_step 2500 \
--save_step_dense 5000 \
--save_step_dense_interval 250 \
--max_steps 200000 \
--manual_seed $SEED \
--train_batch_size 2048 \
--eval_batch_size 4096 \
--gradient_accumulation_steps 2 \
--learning_rate 4e-4 \
--max_seq_length 10 \
--max_length 10 \
--block_size 10 \
--wandb_project "GrokkedTransformer_reproduce" \
--wandb_entity "lklab_kaist" \
--wandb_exp $1_"no-dense_wd"-$2_"layer"$3_"head"$4_"seed"$6
# --save_fine_step_list 2500 31250 75000