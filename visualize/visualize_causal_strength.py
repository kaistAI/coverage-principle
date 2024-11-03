import json
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from collections import Counter

def _visualize_causal_strength(causal_strength, checkpoint_name, args):
    if args.task == "composition":
        label = ["h", "r1", "r2"]
    else:
        label = ["a", "e1", "e2"]
        
    if args.soft_ver:
        organized_causal_strength = [[[0, 0, 0] for _ in range(1, args.num_layer)], [[0, 0, 0] for _ in range(1, args.num_layer)]]    # First is for mean value, second is for median value
        # Save each entities real value statistics
        for i in range(1, args.num_layer):
            for j, intervention in enumerate(label):
                plt.figure(figsize=(20,10))
                key = intervention + "_" + str(i)
            
                plt.hist(causal_strength[key], bins=range(min(causal_strength[key]), max(causal_strength[key]) + 2), edgecolor='black')
                plt.title(f"{key} change statistics")
                plt.xlabel('Value')
                plt.ylabel('Count')
                
                # Calculate the median index and add a vertical red line
                median_value = np.median(causal_strength[key])
                mean_value = np.mean(causal_strength[key])
                organized_causal_strength[0][i-1][j] = mean_value
                organized_causal_strength[1][i-1][j] = median_value
                plt.axvline(x=median_value, color='red', linestyle='--', label='Median')
                std_dev = np.std(causal_strength[key])
                
                plt.grid(True)
                # Add median and standard deviation text
                plt.text(0.95, 0.05, f'Median: {median_value:.2f}\nStd Dev: {std_dev:.2f}',
                        verticalalignment='bottom', horizontalalignment='right',
                        transform=plt.gca().transAxes, fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.5))
                os.makedirs(os.path.join(os.path.dirname(args.json_path), os.path.splitext(os.path.basename(args.json_path))[0], "statistics"), exist_ok=True)
                plt.savefig(os.path.join(os.path.dirname(args.json_path), os.path.splitext(os.path.basename(args.json_path))[0], "statistics", f"{checkpoint_name}-{key}.png"), dpi=300, bbox_inches='tight')
                plt.close()

        colors = ["w", "g"]  # Yellow, White, Green
        mean_vmin = min(min(sublist) for sublist in organized_causal_strength[0])
        mean_vmax = max(max(sublist) for sublist in organized_causal_strength[0])
        median_vmin = min(min(sublist) for sublist in organized_causal_strength[1])
        median_vmax = max(max(sublist) for sublist in organized_causal_strength[1])
    else:
        organized_causal_strength = [[causal_strength[f"{label[0]}_{layer_num}"], causal_strength[f"{label[1]}_{layer_num}"], causal_strength[f"{label[2]}_{layer_num}"]] for layer_num in range(1, args.num_layer)]
        colors = ["y", "w", "g"]  # Yellow, White, Green
        vmin = -1
        vmax  = 1
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    
    if args.soft_ver:
        # Creating the heatmap for mean value
        plt.figure(figsize=(4, 7))
        ax = sns.heatmap(organized_causal_strength[0], annot=True, fmt=".2f", cmap=cmap, cbar=True, vmin=mean_vmin, vmax=mean_vmax, 
                        yticklabels=['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Layer 6', 'Layer 7'], 
                        xticklabels=label,
                        annot_kws={"size": 10, "ha": 'center', "va": 'center'})

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
        plt.savefig(os.path.join(os.path.dirname(args.json_path), os.path.splitext(os.path.basename(args.json_path))[0], f"{checkpoint_name}_soft-ver_mean.png"), dpi=300, bbox_inches='tight')
        
        # Creating the heatmap for median value
        plt.figure(figsize=(4, 7))
        ax = sns.heatmap(organized_causal_strength[1], annot=True, fmt=".2f", cmap=cmap, cbar=True, vmin=median_vmin, vmax=median_vmax, 
                        yticklabels=[f"Layer {i}" for i in range(1, args.num_layer)], 
                        xticklabels=label,
                        annot_kws={"size": 10, "ha": 'center', "va": 'center'})

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
        plt.savefig(os.path.join(os.path.dirname(args.json_path), os.path.splitext(os.path.basename(args.json_path))[0], f"{checkpoint_name}_soft-ver_median.png"), dpi=300, bbox_inches='tight')
    else:
        # Creating the heatmap
        plt.figure(figsize=(4, 7))
        ax = sns.heatmap(organized_causal_strength, annot=True, fmt=".2f", cmap=cmap, cbar=True, vmin=vmin, vmax=vmax, 
                        yticklabels=[f"Layer {i}" for i in range(1, args.num_layer)], 
                        xticklabels=label,
                        annot_kws={"size": 10, "ha": 'center', "va": 'center'})

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
            if args.soft_ver:
                causal_strength[key] = []
            else:
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
            for layer_num in range(1, args.num_layer):
                for intervention in all_keys:
                    key = intervention + str(layer_num)
                    if intervention in causal_strength_keys:
                        if args.soft_ver:
                            causal_strength[key].append(result[key] - original_rank)
                        else:
                            if result[key] != original_rank:
                                causal_strength[key] += 1
                    mean_reciprocal_rank[key].append(1 / (1 + result[key]))
        if not args.soft_ver:
            for key in causal_strength:
                causal_strength[key] = causal_strength[key] / total_result_num
        for key in mean_reciprocal_rank:
            mean_reciprocal_rank[key] = sum(mean_reciprocal_rank[key]) / len(mean_reciprocal_rank[key])
        
        if checkpoint == min_checkpoint_step:
            min_checkpoint_causal_strength = causal_strength
        elif checkpoint == max_checkpoint_step:
            max_checkpoint_causal_strength = causal_strength
        _visualize_causal_strength(causal_strength, checkpoint, args)
        print("Each checkpoint visualization done!")

    difference_causal_strength = {}
    for key in max_checkpoint_causal_strength.keys():
        if args.soft_ver:
            difference_causal_strength[key] = [a - b for a, b in zip(max_checkpoint_causal_strength[key], min_checkpoint_causal_strength[key])]
        else:
            difference_causal_strength[key] = max_checkpoint_causal_strength[key] - min_checkpoint_causal_strength[key]
    _visualize_causal_strength(difference_causal_strength, f"{max_checkpoint_step}-{min_checkpoint_step}", args)
    print(f"Difference visualization done!")


def main(args):
    if "composition" in os.path.splitext(os.path.basename(args.json_path))[0]:
        args.task = "composition"
    else:
        args.task = "comparison"
    visualize_causal_strength(args)


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_path", default=None, type=str, required=True, help="result JSON file path")
    parser.add_argument("--num_layer", default=8, type=int, help="number of layer of the model")
    parser.add_argument("--soft_ver", action="store_true")

    args = parser.parse_args()
    main(args)