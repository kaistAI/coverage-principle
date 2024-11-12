import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import matplotlib.pyplot as plt
import numpy as np


def compute_activations(model, input_strs, tokenizer, batch_size, device):
    all_attention_maps = dict()
    
    for batch_idx in range(0, len(input_strs), batch_size):
        batch = input_strs[batch_idx:batch_idx+batch_size]
        
        batch_ids = tokenizer(batch, return_tensors="pt", padding=True)
        input_ids, attention_mask = batch_ids['input_ids'].to(device), batch_ids["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, 
                            attention_mask=attention_mask, 
                            output_attentions=True
                            )

        attentions = outputs.attentions
        
        num_attention_layers = len(attentions)
        num_heads = attentions[0].shape[1]
        
        for layer_idx in range(num_attention_layers):
            attention_map_sum = torch.sum(attentions[layer_idx], dim=0, keepdim=True)
            if batch_idx == 0:
                all_attention_maps[f"Layer_{layer_idx+1}"] = [attention_map_sum[0, head].cpu().numpy() for head in range(num_heads)]
            else:
                for i in range(num_heads):
                    all_attention_maps[f"Layer_{layer_idx+1}"][i] = np.add(all_attention_maps[f"Layer_{layer_idx}"][i], attention_map_sum[0, i].cpu().numpy())
    for layer_idx in range(num_attention_layers):
        for map in all_attention_maps[f"Layer_{layer_idx+1}"]:
            map /= len(input_strs)
    return all_attention_maps


def heatmap_for_each_head(all_attention_maps, save_dir, checkpoint):
    num_attn_layer = len(list(all_attention_maps.keys()))
    num_heads = len(all_attention_maps[list(all_attention_maps.keys())[0]])

    # Set up the matplotlib figure
    fig, axes = plt.subplots(num_attn_layer, num_heads, figsize=(3 * num_heads, 3 * num_attn_layer))
    fig.suptitle("Attention Maps Across Layers and Heads", fontsize=16)

    # Plot each attention map in the corresponding subplot
    for i, (layer, head_matrices) in enumerate(all_attention_maps.items()):
        for j, attention_matrix in enumerate(head_matrices):
            ax = axes[i, j]
            ax.matshow(attention_matrix, cmap='viridis', vmin=0, vmax=1)
            for ip in range(attention_matrix.shape[0]):
                for jp in range(attention_matrix.shape[1]):
                    text = f"{attention_matrix[ip, jp]:.1f}"
                    threshold = 1 / 2.
                    color = "white" if attention_matrix[ip, jp] < threshold else "black"
                    ax.text(jp, ip, text,
                            ha='center', va='center',
                            color=color)
            ax.set_title(f"{layer} - Head {j}")
            ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the main title
    plt.savefig(f"{save_dir}/{checkpoint}-head.png")


def heatmap_for_each_input(all_attention_maps, save_dir, checkpoint):
    organized_all_attention_maps = dict()

    for layer_num, head_attention_maps in all_attention_maps.items():
        attention_map_for_input = [[] for _ in range(head_attention_maps[0].shape[0])]
        # print(f"attention_map_for_input: {attention_map_for_input}")
        
        for attention_map in head_attention_maps:
            # print(attention_map)
            for i in range(attention_map.shape[0]):
                attention_map_for_input[i].append(attention_map[i,:])
        for i in range(len(attention_map_for_input)):
            attention_map_for_input[i] = np.stack(attention_map_for_input[i], axis=0)
        organized_all_attention_maps[layer_num] = attention_map_for_input
        
    for layer_num, input_attention_maps in organized_all_attention_maps.items():
        for attention_map in input_attention_maps:
            assert np.allclose(np.sum(attention_map, axis=1), 1)
    
    num_attn_layer = len(list(organized_all_attention_maps.keys()))
    seq_len = len(organized_all_attention_maps[list(organized_all_attention_maps.keys())[0]])

    # Set up the matplotlib figure
    fig, axes = plt.subplots(num_attn_layer, seq_len-1, figsize=(2 * seq_len, 3 * num_attn_layer))
    fig.suptitle("Attention Maps Across Layers and Heads", fontsize=10)

    # Plot each attention map in the corresponding subplot
    for i, (layer, matrices) in enumerate(organized_all_attention_maps.items()):
        for j, attention_matrix in enumerate(matrices):
            if j == 0:
                continue
            ax = axes[i, j-1]
            ax.matshow(attention_matrix, cmap='viridis', vmin=0, vmax=1, aspect=0.3)
            for ip in range(attention_matrix.shape[0]):
                for jp in range(attention_matrix.shape[1]):
                    text = f"{attention_matrix[ip, jp]:.1f}"
                    threshold = 1 / 2.
                    color = "white" if attention_matrix[ip, jp] < threshold else "black"
                    ax.text(jp, ip, text,
                            ha='center', va='center',
                            color=color)
            if i == 0:
                if j == 1:
                    ax.set_title("<r_1>", fontweight='bold', fontsize=10)
                elif j == 2:
                    ax.set_title("<r_2>", fontweight='bold', fontsize=10)
            if j == 1:
                ax.text(-0.3, 0.5, f"{layer}",
                    fontsize=10, fontweight='bold', 
                    transform=ax.transAxes, rotation=0)
            ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the main title
    plt.savefig(f"{save_dir}/{checkpoint}-input.png")


def main(args):
    all_checkpoints = [f"checkpoint-{checkpoint}" for checkpoint in args.step_list]
    assert all(os.path.isdir(os.path.join(args.model_dir, checkpoint)) for checkpoint in all_checkpoints)
    all_checkpoints.sort(key=lambda var: int(var.split("-")[1]))
    
    device = "cuda:0"

    for checkpoint in all_checkpoints:
        model_path = os.path.join(args.model_dir, checkpoint)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", args.model_dir.split("/")[-1].split("_")[0], "test.json")) as f:
            pred_data = json.load(f)
        
        d = dict()
        for item in pred_data:
            t = item['type']
            if t not in d:
                d[t] = []
            d[t].append(item)
            
        input_datas = [data["input_text"] for data in d[args.data_type]]

        all_attention_maps = compute_activations(model, input_datas, tokenizer, 8192, device)
        
        save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "attention_map", f"{args.model_dir.split('/')[-1]}-{args.data_type}")
        os.makedirs(save_dir, exist_ok=True)
        
        heatmap_for_each_head(all_attention_maps, save_dir, checkpoint)
        heatmap_for_each_input(all_attention_maps, save_dir, checkpoint)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=None, type=str, help="parent directory of saved model checkpoints")
    parser.add_argument("--step_list", default=None, nargs="+", help="checkpoint's steps to check causal strength")
    parser.add_argument("--data_type", default="train_inferred", type=str, help="type of inference data (atomic_id, atomic_ood, train_inferred, test_inferred_iid, test_inferred_ood, ...)")
    
    args = parser.parse_args()
    
    main(args)