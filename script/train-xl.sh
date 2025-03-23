#!/bin/bash

MODEL_PATH=openai-community/gpt2-xl

DATASET=./data/$1/
WEIGHT_DECAY=$2
N_LAYERS=$3
N_HEADS=$4
SEED=$5

OUTPUT_DIR=/mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/$1_"wd"-$2_"layer"-$3_"head"-$4_"seed"-$5

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
--do_eval \
--evaluate_during_training \
--predict_during_training \
--weight_decay $WEIGHT_DECAY \
--scheduler constant_schedule_with_warmup \
--output_dir $OUTPUT_DIR \
--save_step 2500 \
--save_step_dense 5000 \
--save_step_dense_interval 250 \
--max_steps 15000 \
--manual_seed $SEED \
--train_batch_size 1024 \
--eval_batch_size 1024 \
--gradient_accumulation_steps 4 \
--learning_rate 8e-4 \
--max_seq_length 5 \
--max_length 5 \
--block_size 5 \
--wandb_project "GrokkedTransformer_reproduce" \
--wandb_entity "lklab_kaist" \
--wandb_exp [XL]$1_"wd"$2_"layer"$3_"head"$4_"seed"$5 \
--overwrite_output_dir