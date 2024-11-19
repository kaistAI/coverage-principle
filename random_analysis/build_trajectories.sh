base_dir='/mnt/sda/hoyeon/GrokkedTransformer/random_analysis'
exp_name='test'

python build_trajectories.py \
    --save_dir ${base_dir}/results/raw/${exp_name} \
    --output_dir ${base_dir}/results/trajectories/${exp_name} \
    --groups train_inferred test_inferred_iid test_inferred_ood \
    --layer 5 \
    --position 1 \
    --id_field id  # Use 'input_text' if 'id' is not available