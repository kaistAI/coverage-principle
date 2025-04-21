#!/bin/bash

# 현재 디렉토리 저장
CURRENT_DIR=$(pwd)

# 스크립트가 있는 디렉토리로 이동
cd "$(dirname "$0")"

# 각 분석 스크립트 실행
echo "Running collapse_analysis_parallel_2-hop.sh..."
bash collapse_analysis_parallel_2-hop.sh

echo "Running measure_cossim.sh..."
bash measure_cossim.sh

echo "Running plotting_collapse.sh..."
bash plotting_collapse.sh

# 원래 디렉토리로 돌아가기
cd "$CURRENT_DIR"
