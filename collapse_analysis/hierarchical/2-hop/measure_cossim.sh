#!/bin/bash

MODE=residual
CONFIG_FILE="analysis_steps_test.json"

# JSON 파일을 읽어서 jq로 파싱하는 함수
get_steps() {
    local model_dir=$1
    jq -r --arg model "$model_dir" '.[$model].steps[]' "$CONFIG_FILE"
}

get_include_final() {
    local model_dir=$1
    jq -r --arg model "$model_dir" '.[$model].include_final' "$CONFIG_FILE"
}

# 각 MODEL_DIR에 대한 처리를 함수로 정의
process_model() {
    local MODEL_DIR=$1
    
    # MODEL_DIR에서 필요한 부분만 추출 (예: "twohop.150.270000.diff-f12.inf")
    SHORT_MODEL_DIR=$(echo "$MODEL_DIR" | sed -E 's/_.*$//')
    
    echo "=== Processing Model: $SHORT_MODEL_DIR ==="
    echo "Steps to analyze: $(get_steps "$MODEL_DIR")"
    echo "Include final checkpoint: $(get_include_final "$MODEL_DIR")"
    
    DEDUP_DIR="/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/2-hop/${MODE}/${SHORT_MODEL_DIR}"
    DATASET="${SHORT_MODEL_DIR}"
    
    for POS in 1 2
    do
        for LAYER in 1 2 3 4 5 6 7 8 logit prob
        do
            for ATOMIC_IDX in 1 2
            do
                for STEP in $(get_steps "$MODEL_DIR")
                do
                    CUDA_VISIBLE_DEVICES=0 python measure_cossim.py \
                        --id_train_file "${DEDUP_DIR}/f${ATOMIC_IDX}/(${LAYER},${POS})/${STEP}/id_train_dedup.json" \
                        --id_test_file "${DEDUP_DIR}/f${ATOMIC_IDX}/(${LAYER},${POS})/${STEP}/id_test_dedup.json" \
                        --ood_file "${DEDUP_DIR}/f${ATOMIC_IDX}/(${LAYER},${POS})/${STEP}/ood_dedup.json" \
                        --output_dir "/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/results/2-hop/${MODE}/${DATASET}/f${ATOMIC_IDX}/(${LAYER},${POS})/step${STEP}" &
                done
                wait
            done
        done
    done
    echo "=== Finished processing $MODEL_DIR ==="
    echo "======================================"
}

# 모델 디렉토리 목록 가져오기
MODEL_DIRS=$(jq -r 'keys[]' "$CONFIG_FILE")

echo "=== Config File Path: $CONFIG_FILE ==="
echo "=== Found Model Directories: ==="
echo "$MODEL_DIRS"
echo "==============================="

# 각 MODEL_DIR에 대한 처리를 백그라운드로 실행
for MODEL_DIR in $MODEL_DIRS
do
    process_model "$MODEL_DIR" &
done

# 모든 MODEL_DIR 처리가 완료될 때까지 대기
wait

echo "All model processing completed!"