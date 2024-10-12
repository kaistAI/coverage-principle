import json
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def _visualize_causal_strength(causal_strength, checkpoint_name, args):
    organized_causal_strength = [
        [causal_strength['h_1'], causal_strength['r1_1'], causal_strength['r2_1']],
        [causal_strength['h_2'], causal_strength['r1_2'], causal_strength['r2_2']],
        [causal_strength['h_3'], causal_strength['r1_3'], causal_strength['r2_3']],
        [causal_strength['h_4'], causal_strength['r1_4'], causal_strength['r2_4']],
        [causal_strength['h_5'], causal_strength['r1_5'], causal_strength['r2_5']],
        [causal_strength['h_6'], causal_strength['r1_6'], causal_strength['r2_6']],
        [causal_strength['h_7'], causal_strength['r1_7'], causal_strength['r2_7']]
    ]

    colors = ["y", "w", "g"]  # Yellow, White, Green
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

    # Creating the heatmap
    plt.figure(figsize=(4, 7))
    ax = sns.heatmap(organized_causal_strength, annot=False, cmap=cmap, cbar=True, vmin=-1, vmax=1, 
                    yticklabels=['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Layer 6', 'Layer 7'], 
                    xticklabels=['h', 'r1', 'r2'])

    # Adding a bold outside border
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)  # Make the border bold

    # Rotate the y-axis tick labels to horizontal
    plt.yticks(rotation=0, fontsize=12)
    plt.xticks(fontsize=12)

    # Customize the color bar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)  # Change the tick label size

    # Adding a bold outside border for the color bar
    cbar.outline.set_linewidth(1.5)  # Set the linewidth of the color bar border
    cbar.outline.set_edgecolor('black')  # Set the color of the color bar border

    # Show the heatmap
    os.makedirs(os.path.join(os.path.dirname(args.json_path), "composition"), exist_ok=True)
    plt.savefig(os.path.join(os.path.dirname(args.json_path), "composition", f"{checkpoint_name}.png"), dpi=300, bbox_inches='tight')

def visualize_causal_strength_composition(args):
    
    with open(args.json_path, "r") as f:
        raw_datas = json.load(f)
        
    # min & max checkpoint steps
    
    all_checkpoints = [key for key in raw_datas.keys()]
    all_checkpoints.sort(key=lambda var: int(var.split("-")[1]))
    min_checkpoint_step = all_checkpoints[0]
    max_checkpoint_step = all_checkpoints[-1]
    
    min_checkpoint_causal_strength = None
    max_checkpoint_causal_strength = None
    
    for checkpoint, raw_data in raw_datas.items():
        total_result_num = len(raw_data)

        mean_reciprocal_rank = {}
        causal_strength = {}

        for key in raw_data[0].keys():
            if key == "rank_before":
                continue
            mean_reciprocal_rank[key] = []
            if "b_rank_pos1" in key or "r2_rank_pos2" in key:
                continue
            causal_strength[key] = 0
        
        for result in raw_data:
            original_rank = result["rank_before"]
            for layer_num in range(1, 8):
                for intervention in ["b_rank_pos1_", "r2_rank_pos2_", "h_", "r1_", "r2_"]:
                    key = intervention + str(layer_num)
                    if intervention in ["h_", "r1_", "r2_"]:
                        if result[key] != original_rank:
                            causal_strength[key] += 1
                    mean_reciprocal_rank[key].append(1 / (1 + result[key]))
                    
        for key in causal_strength:
            causal_strength[key] = causal_strength[key] / total_result_num
        for key in mean_reciprocal_rank:
            mean_reciprocal_rank[key] = sum(mean_reciprocal_rank[key]) / len(mean_reciprocal_rank[key])
            
        if checkpoint == min_checkpoint_step:
            min_checkpoint_causal_strength = causal_strength
        elif checkpoint == max_checkpoint_step:
            max_checkpoint_causal_strength = causal_strength
        _visualize_causal_strength(causal_strength, checkpoint, args)

    difference_causal_strength = {}
    for key in max_checkpoint_causal_strength.keys():
        difference_causal_strength[key] = max_checkpoint_causal_strength[key] - min_checkpoint_causal_strength[key]
    _visualize_causal_strength(difference_causal_strength, f"{max_checkpoint_step}-{min_checkpoint_step}", args)
    

def visualize_causal_strength_comparison(raw_data):
    raise NotImplementedError


def main(args):
    if os.path.splitext(os.path.basename(args.json_path))[0] == "composition":
        visualize_causal_strength_composition(args)
    else:
        visualize_causal_strength_comparison(args)


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_path", default=None, type=str, required=True, help="result JSON file path")

    args = parser.parse_args()
    main(args)