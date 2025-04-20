#!/bin/bash

# 에러 발생 시 스크립트 중단
set -e

# 현재 스크립트의 디렉토리 경로
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

MODE=residual
CONFIG_FILES=("analysis_list_paper.json" "analysis_list_cot_paper.json")

# JSON 파일을 읽어서 jq로 파싱하는 함수
get_steps() {
    local model_dir=$1
    local config_file=$2
    jq -r --arg model "$model_dir" '.[$model].steps[]' "$SCRIPT_DIR/$config_file"
}

get_include_final() {
    local model_dir=$1
    local config_file=$2
    jq -r --arg model "$model_dir" '.[$model].include_final' "$SCRIPT_DIR/$config_file"
}

# 각 MODEL_DIR에 대한 처리를 함수로 정의
process_model() {
    local MODEL_DIR=$1
    local CONFIG_FILE=$2
    
    # MODEL_DIR에서 필요한 부분만 추출
    SHORT_MODEL_DIR=$(echo "$MODEL_DIR" | sed -E 's/_.*$//')
    
    echo "=== Processing Model: $SHORT_MODEL_DIR ==="
    echo "=== Using Config File: $CONFIG_FILE ==="
    echo "Steps to analyze: $(get_steps "$MODEL_DIR" "$CONFIG_FILE")"
    echo "Include final checkpoint: $(get_include_final "$MODEL_DIR" "$CONFIG_FILE")"
    
    # CONFIG_FILE에 따라 다른 디렉토리 경로 설정
    if [ "$CONFIG_FILE" = "analysis_list_paper.json" ]; then
        POS_RANGE=(1 2)
        ATOMIC_IDX_RANGE=(1 2)
    else
        POS_RANGE=(1 2 3)
        ATOMIC_IDX_RANGE=(1 2)
    fi
    DEDUP_DIR="/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/2-hop/${MODE}/${SHORT_MODEL_DIR}"
    DATASET="${SHORT_MODEL_DIR}"
    
    for POS in "${POS_RANGE[@]}"
    do
        for LAYER in 1 2 3 4 5 6 7 8 logit prob
        do
            for ATOMIC_IDX in "${ATOMIC_IDX_RANGE[@]}"
            do
                for STEP in $(get_steps "$MODEL_DIR" "$CONFIG_FILE")
                do
                    if [ ! -d "${DEDUP_DIR}/f${ATOMIC_IDX}/(${LAYER},${POS})/${STEP}" ]; then
                        echo "Warning: Directory ${DEDUP_DIR}/f${ATOMIC_IDX}/(${LAYER},${POS})/${STEP} does not exist"
                        continue
                    fi

                    python "$PARENT_DIR/plotting_collapse.py" \
                        --id_train_file "${DEDUP_DIR}/f${ATOMIC_IDX}/(${LAYER},${POS})/${STEP}/id_train_dedup.json" \
                        --id_test_file "${DEDUP_DIR}/f${ATOMIC_IDX}/(${LAYER},${POS})/${STEP}/id_test_dedup.json" \
                        --ood_file "${DEDUP_DIR}/f${ATOMIC_IDX}/(${LAYER},${POS})/${STEP}/ood_dedup.json" \
                        --output_dir "/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/results/2-hop/${MODE}/${DATASET}/f${ATOMIC_IDX}/(${LAYER},${POS})/step${STEP}" \
                        --save_plots \
                        --reduce_method tsne \
                        --pca_scope global \
                        --pca_n 5 &
                done

                if [ "$(get_include_final "$MODEL_DIR" "$CONFIG_FILE")" = "true" ]; then
                    if [ ! -d "${DEDUP_DIR}/f${ATOMIC_IDX}/(${LAYER},${POS})/final_checkpoint" ]; then
                        echo "Warning: Directory ${DEDUP_DIR}/f${ATOMIC_IDX}/(${LAYER},${POS})/final_checkpoint does not exist"
                        continue
                    fi

                    python "$PARENT_DIR/plotting_collapse.py" \
                        --id_train_file "${DEDUP_DIR}/f${ATOMIC_IDX}/(${LAYER},${POS})/final_checkpoint/id_train_dedup.json" \
                        --id_test_file "${DEDUP_DIR}/f${ATOMIC_IDX}/(${LAYER},${POS})/final_checkpoint/id_test_dedup.json" \
                        --ood_file "${DEDUP_DIR}/f${ATOMIC_IDX}/(${LAYER},${POS})/final_checkpoint/ood_dedup.json" \
                        --output_dir "/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/results/2-hop/${MODE}/${DATASET}/f${ATOMIC_IDX}/(${LAYER},${POS})/stepfinal_checkpoint" \
                        --save_plots \
                        --reduce_method tsne \
                        --pca_scope global \
                        --pca_n 5 &
                fi
                wait
            done
        done
    done
    echo "=== Finished processing $MODEL_DIR ==="
    echo "======================================"
}

# 각 CONFIG_FILE에 대해 처리
for CONFIG_FILE in "${CONFIG_FILES[@]}"
do
    if [ ! -f "$SCRIPT_DIR/$CONFIG_FILE" ]; then
        echo "Error: Config file $CONFIG_FILE not found"
        exit 1
    fi

    echo "=== Processing Config File: $CONFIG_FILE ==="
    
    # 모델 디렉토리 목록 가져오기
    MODEL_DIRS=$(jq -r 'keys[]' "$SCRIPT_DIR/$CONFIG_FILE")

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