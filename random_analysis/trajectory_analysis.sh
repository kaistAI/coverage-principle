#!/bin/bash

exp_name="reproduce/dense_3"

# Paths
INPUT_DIR="/mnt/sda/hoyeon/GrokkedTransformer/random_analysis/results/trajectories/${exp_name}"  # Root directory containing group subdirectories
OUTPUT_DIR="/mnt/sda/hoyeon/GrokkedTransformer/random_analysis/results/analysis/${exp_name}"      # Directory to save analysis outputs

# Groups and Vector Types
GROUPS=("train_inferred" "test_inferred_iid" "test_inferred_ood" "test_nonsenses")
VECTOR_TYPES=("post_attention" "post_mlp" "residual")

# Identifier Field
ID_FIELD="id"  # Change to 'input_text' if your data uses 'input_text' as the identifier

# --groups "train_inferred" "test_inferred_iid" "test_inferred_ood" "test_nonsenses" \
# Execute the post_analysis.py script
python trajectory_analysis.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --groups "train_inferred" "test_inferred_iid" "test_inferred_ood" "test_nonsenses" \
    --vector_types "post_mlp"