#!/bin/bash

# Paths
INPUT_DIR="/mnt/sda/hoyeon/GrokkedTransformer/random_analysis/results/trajectories/test"  # Root directory containing group subdirectories
OUTPUT_DIR="/mnt/sda/hoyeon/GrokkedTransformer/random_analysis/results/analysis/test"      # Directory to save analysis outputs

# Groups and Vector Types
GROUPS=("train_inferred" "test_inferred_iid" "test_inferred_ood")
VECTOR_TYPES=("post_attention" "post_mlp" "residual")

# Identifier Field
ID_FIELD="id"  # Change to 'input_text' if your data uses 'input_text' as the identifier

# Execute the post_analysis.py script
python trajectory_analysis.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --groups "train_inferred" "test_inferred_iid" "test_inferred_ood" \
    --vector_types "post_attention" "post_mlp" "residual" \
    --id_field "$ID_FIELD"