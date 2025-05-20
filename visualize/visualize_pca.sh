#! /bin/bash

python visualize_pca.py \
    --pca_files "/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/2-hop/residual/twohop.50.10000.diff-f12.inf_detailed_grouping_2/f1/(3,1)/50000/id_test_covered_low_cutoff_dedup.json" "/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/2-hop/residual/twohop.50.10000.diff-f12.inf_detailed_grouping_2/f1/(3,1)/50000/id_test_covered_high_cutoff_dedup.json" "/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/2-hop/residual/twohop.50.10000.diff-f12.inf_detailed_grouping_2/f1/(3,1)/50000/ood_dedup.json" \
    --reduce_method pca \
    --pca_n 5 \
    --reduce_dim 2 \
    --local

python visualize_pca.py \
    --pca_files "/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/2-hop/residual/twohop.50.10000.diff-f12.inf_detailed_grouping_2/f1/(3,1)/50000/id_test_covered_low_cutoff_dedup.json" "/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/2-hop/residual/twohop.50.10000.diff-f12.inf_detailed_grouping_2/f1/(3,1)/50000/id_test_covered_high_cutoff_dedup.json" "/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/2-hop/residual/twohop.50.10000.diff-f12.inf_detailed_grouping_2/f1/(3,1)/50000/ood_dedup.json" \
    --reduce_method pca \
    --pca_n 5 \
    --reduce_dim 2

# python visualize_pca.py \
#     --pca_files "/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/2-hop/residual/twohop.50.10000.diff-f12.inf_detailed_grouping_2/f1/(3,1)/50000/id_test_covered_low_cutoff_dedup.json" "/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/2-hop/residual/twohop.50.10000.diff-f12.inf_detailed_grouping_2/f1/(3,1)/50000/id_test_covered_high_cutoff_dedup.json" "/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/2-hop/residual/twohop.50.10000.diff-f12.inf_detailed_grouping_2/f1/(3,1)/50000/ood_dedup.json" \
#     --reduce_method tsne \
#     --pca_n 5 \
#     --reduce_dim 2
    

# For Non-tree DAG Figure
# python combine_heatmaps_for_section_5-2.py \
#     -r "/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/results/nontree/residual/nontree.50.10000.diff-f12.inf"
