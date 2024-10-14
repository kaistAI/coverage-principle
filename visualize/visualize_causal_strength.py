import json
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def _visualize_causal_strength(causal_strength, checkpoint_name, args):
    if args.task == "composition":
        label = ["h", "r1", "r2"]
    else:
        label = ["a", "e1", "e2"]
    
    organized_causal_strength = [[causal_strength[f"{label[0]}_{layer_num}"], causal_strength[f"{label[1]}_{layer_num}"], causal_strength[f"{label[2]}_{layer_num}"]] for layer_num in range(1, 8)]

    colors = ["y", "w", "g"]  # Yellow, White, Green
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

    # Creating the heatmap
    plt.figure(figsize=(4, 7))
    ax = sns.heatmap(organized_causal_strength, annot=False, cmap=cmap, cbar=True, vmin=-1, vmax=1, 
                    yticklabels=['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Layer 6', 'Layer 7'], 
                    xticklabels=label)

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

    # Save the heatmap
    os.makedirs(os.path.join(os.path.dirname(args.json_path), os.path.splitext(os.path.basename(args.json_path))[0]), exist_ok=True)
    plt.savefig(os.path.join(os.path.dirname(args.json_path), os.path.splitext(os.path.basename(args.json_path))[0], f"{checkpoint_name}.png"), dpi=300, bbox_inches='tight')

def visualize_causal_strength(args):
    
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
            # Every keys will have MMR except "rank_before" key
            if key == "rank_before":
                continue
            mean_reciprocal_rank[key] = []
            # This keys only used in MMR not in Causal Strength
            if ((args.task == "composition" 
                 and any((k in key for k in ["b_rank_pos1", "r2_rank_pos2"])))
            or (args.task == "comparison" 
                and any(k in key for k in ["val1_rank_pos2", "val2_rank_pos4", "label0_rank_pos0", 
                                           "label1_rank_pos0", "label2_rank_pos0"]))
            ):
                continue
            causal_strength[key] = 0
            
        if args.task == "composition":
            all_keys = ["b_rank_pos1_", "r2_rank_pos2_", "h_", "r1_", "r2_"]
            causal_strength_keys = ["h_", "r1_", "r2_"]
        elif args.task == "comparison":
            all_keys = ["val1_rank_pos2_", "val2_rank_pos4_", "label0_rank_pos0_", 
                        "label1_rank_pos0_", "label2_rank_pos0_", "e1_", "e2_", "a_"]
            causal_strength_keys = ["e1_", "e2_", "a_"]
        else:
            raise NotImplementedError
        
        for result in raw_data:
            original_rank = result["rank_before"]
            for layer_num in range(1, 8):
                for intervention in all_keys:
                    key = intervention + str(layer_num)
                    if intervention in causal_strength_keys:
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


def main(args):
    if "composition" in os.path.splitext(os.path.basename(args.json_path))[0]:
        args.task = "composition"
    else:
        args.task = "comparison"
    visualize_causal_strength(args)


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_path", default=None, type=str, required=True, help="result JSON file path")

    args = parser.parse_args()
    main(args)