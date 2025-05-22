CUDA_VISIBLE_DEVICES=3 python ../logit-lens_2-hop.py \
    --model_dir "/mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.50.10000.same-f12.with-atomic_wd-0.1_layer-8_head-12_seed-42" \
    --step_list 40000 \
    --batch_size 4096

CUDA_VISIBLE_DEVICES=3 python ../logit-lens_2-hop.py \
    --model_dir "/mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/twohop.50.10000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42_extended" \
    --step_list 50000 \
    --batch_size 4096