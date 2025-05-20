#! /bin/bash

# For 2hop Figure
python combine_heatmaps_for_section_5-2.py \
    --root_dir "/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/results/2-hop/residual/twohop.50.10000.diff-f12.inf_detailed_grouping_2" \
    --reduce_method pca \
    --id_test_files "/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/2-hop/residual/twohop.50.10000.diff-f12.inf_detailed_grouping_2/f1/(3,1)/50000/id_test_covered_low_cutoff_dedup.json" "/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/2-hop/residual/twohop.50.10000.diff-f12.inf_detailed_grouping_2/f1/(3,1)/50000/id_test_covered_mid_cutoff_dedup.json" "/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/2-hop/residual/twohop.50.10000.diff-f12.inf_detailed_grouping_2/f1/(4,1)/50000/id_test_covered_high_cutoff_dedup.json" \
    --pca_n 5 \
    --reduce_dim 3
    

# For Non-tree DAG Figure
# python combine_heatmaps_for_section_5-2.py \
#     -r "/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/results/nontree/residual/nontree.50.10000.diff-f12.inf"
