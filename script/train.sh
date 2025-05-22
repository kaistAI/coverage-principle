#!/bin/bash

MODEL_PATH=gpt2

DATASET=./data/$1/
WEIGHT_DECAY=$2
N_LAYERS=$3
N_HEADS=$4
GPU=$5
SEED=$6

OUTPUT_DIR=CKPT_DIR/trained_checkpoints/$1_"wd"-$2_"layer"-$3_"head"-$4_"seed"-$6

# CUDA_VISIBLE_DEVICES=$GPU python main.py \
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 main.py \
--data_dir $DATASET \
--model_name_or_path ${MODEL_PATH} \
--init_weights \
--add_tokens \
--n_layer $N_LAYERS \
--n_head $N_HEADS \
--fp16 \
--do_train \
--evaluate_during_training \
--predict_during_training \
--weight_decay $WEIGHT_DECAY \
--scheduler constant_schedule_with_warmup \
--output_dir $OUTPUT_DIR \
--save_step 2500 \
--save_step_dense 5000 \
--save_step_dense_interval 250 \
--max_steps 62500 \
--manual_seed $SEED \
--train_batch_size 4096 \
--eval_batch_size 4096 \
--gradient_accumulation_steps 1 \
--learning_rate 1e-4 \
--max_seq_length 10 \
--max_length 10 \
--block_size 10 \
--wandb_exp $1_"wd"$2_"layer"$3_"head"$4_"seed"$6