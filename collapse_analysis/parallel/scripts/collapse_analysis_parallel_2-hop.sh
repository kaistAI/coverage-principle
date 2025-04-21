#!/bin/bash

MODE=residual
# CONFIG_FILES를 배열로 선언
CONFIG_FILES=("analysis_list_paper.json")  # 여기에 필요한 config 파일들을 추가하세요

# JSON 파일을 읽어서 jq로 파싱하는 함수
get_steps() {
    local model_dir=$1
    local config_file=$2
    jq -r --arg model "$model_dir" '.[$model].steps[]' "$config_file"
}

get_include_final() {
    local model_dir=$1
    local config_file=$2
    jq -r --arg model "$model_dir" '.[$model].include_final' "$config_file"
}

# 각 MODEL_DIR에 대한 처리를 함수로 정의
process_model() {
    local MODEL_DIR=$1
    local CONFIG_FILE=$2
    
    echo "=== Processing Model: $MODEL_DIR ==="
    echo "=== Using Config File: $CONFIG_FILE ==="
    echo "Steps to analyze: $(get_steps "$MODEL_DIR" "$CONFIG_FILE")"
    echo "Include final checkpoint: $(get_include_final "$MODEL_DIR" "$CONFIG_FILE")"
    
    for POS in 1 2 3
    do
        for LAYER in 1 2 3 4 5 6 7 8 logit prob
        do
            for ATOMIC_IDX in 1 2 3
            do
                for STEP in $(get_steps "$MODEL_DIR" "$CONFIG_FILE")
                do
                    CHECKPOINT_DIR="/mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/${MODEL_DIR}/checkpoint-${STEP}"
                    CUDA_VISIBLE_DEVICES=3 python ../collapse_analysis_parallel_2-hop.py \
                        --ckpt ${CHECKPOINT_DIR}/ \
                        --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
                        --layer_pos_pairs "[(${LAYER},${POS})]" \
                        --save_dir /mnt/nas/jinho/GrokkedTransformer/collapse_analysis/parallel/ \
                        --atomic_idx ${ATOMIC_IDX} \
                        --mode ${MODE} &
                done

                if [ "$(get_include_final "$MODEL_DIR" "$CONFIG_FILE")" = "true" ]; then
                    CHECKPOINT_DIR="/mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/${MODEL_DIR}/final_checkpoint"
                    CUDA_VISIBLE_DEVICES=3 python ../collapse_analysis_parallel_2-hop.py \
                        --ckpt ${CHECKPOINT_DIR}/ \
                        --data_dir /mnt/sda/hoyeon/GrokkedTransformer \
                        --layer_pos_pairs "[(${LAYER},${POS})]" \
                        --save_dir /mnt/nas/jinho/GrokkedTransformer/collapse_analysis/parallel/ \
                        --atomic_idx ${ATOMIC_IDX} \
                        --mode ${MODE} &
                fi
            done
            wait
        done
    done
    echo "=== Finished processing $MODEL_DIR ==="
    echo "======================================"
}

# 각 CONFIG_FILE에 대해 처리
for CONFIG_FILE in "${CONFIG_FILES[@]}"
do
    echo "=== Processing Config File: $CONFIG_FILE ==="
    
    # 모델 디렉토리 목록 가져오기
    MODEL_DIRS=$(jq -r 'keys[]' "$CONFIG_FILE")

    echo "=== Found Model Directories: ==="
    echo "$MODEL_DIRS"
    echo "==============================="

    # 각 MODEL_DIR에 대한 처리를 백그라운드로 실행
    for MODEL_DIR in $MODEL_DIRS
    do
        process_model "$MODEL_DIR" "$CONFIG_FILE" &
    done

    # 현재 CONFIG_FILE의 모든 MODEL_DIR 처리가 완료될 때까지 대기
    wait
    echo "=== Finished processing $CONFIG_FILE ==="
    echo "========================================"
done

echo "All model processing completed!"