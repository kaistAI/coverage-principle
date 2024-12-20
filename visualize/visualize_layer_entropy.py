import argparse
import os
import json
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
import matplotlib.pyplot as plt


def calculate_entropy(logits):
    
    probabilities = F.softmax(logits, dim=-1)
    
    # Calculate entropy
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1)  # Avoid log(0)
    total_entropy = torch.sum(entropy, dim=0).cpu() # shape : (seq_len)
    return total_entropy


def main(args):
    
    if args.model_dir.split("/")[-1] == "":
        dataset = args.model_dir.split("/")[-2].split("_")[0]
    else:
        dataset = args.model_dir.split("/")[-1].split("_")[0]
        
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    with open(os.path.join(base_dir, "data", dataset, "test.json")) as f:
        test_data = json.load(f)
        
    test_dict = {}
    
    for item in test_data:
        t = item['type']
        if t not in test_dict:
            test_dict[t] = []
        test_dict[t].append(item)
    
    for key, datas in test_dict.items():
        print(f"{key}: {len(datas)}")
        
    all_checkpoints = [checkpoint for checkpoint in os.listdir(args.model_dir) if checkpoint.startswith("checkpoint")]
    all_checkpoints.sort(key=lambda var: int(var.split("-")[1]))
    print(all_checkpoints)
    
    device = torch.device('cuda:1')
    BATCH_SIZE = 4096
    
    all_entropy = dict()
    for key in test_dict.keys():
        all_entropy[key] = dict()
    
    for checkpoint in tqdm(all_checkpoints):
        print("now checkpoint", checkpoint)
        model_path = os.path.join(args.model_dir, checkpoint)
        model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        word_embedding = model.lm_head.weight.data
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left" 
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        
        for data_type, datas in test_dict.items():
            total_batch_size = 0
            type_entropy_avg_for_position = dict()
            for i in range(0, len(datas), BATCH_SIZE):
                batch = datas[i:i+BATCH_SIZE]
                queries = [data['input_text'] for data in batch]
                labels = [data['target_text'] for data in batch]
                batch_size = len(queries)
                tokenized_output = tokenizer(queries, return_tensors="pt", padding=True)
                tokenized_input_ids, attention_mask = tokenized_output["input_ids"], tokenized_output["attention_mask"]
                tokenized_input_ids, attention_mask = tokenized_input_ids.to(device), attention_mask.to(device)
                
                with torch.no_grad():
                    outputs = model(
                        input_ids=tokenized_input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                all_hidden_states = outputs['hidden_states']

                for layer_ind in range(1, args.num_layer+1):
                    hidden_states_orig = all_hidden_states[layer_ind]
                    with torch.no_grad():
                        if layer_ind != args.num_layer:
                            temp = model.transformer.ln_f(hidden_states_orig)
                            temp_logit_lens = torch.matmul(temp, word_embedding.T)
                        else:
                            temp_logit_lens = torch.matmul(hidden_states_orig, word_embedding.T)

                    # Calculate entropy for the logits
                    batch_entropy_for_position = calculate_entropy(temp_logit_lens)
                    if not layer_ind in type_entropy_avg_for_position:
                        type_entropy_avg_for_position[layer_ind] = torch.zeros_like(batch_entropy_for_position)
                    type_entropy_avg_for_position[layer_ind] = type_entropy_avg_for_position[layer_ind] + batch_entropy_for_position
                total_batch_size += batch_size
            for key, data in type_entropy_avg_for_position.items():
                type_entropy_avg_for_position[key] = type_entropy_avg_for_position[key] / total_batch_size
            
            
            for pos in range(type_entropy_avg_for_position[1].shape[0]):
                if not f"pos_{pos}" in all_entropy[data_type]:
                    all_entropy[data_type][f"pos_{pos}"] = dict()
                for layer_ind in range(1, args.num_layer+1):
                    if not layer_ind in all_entropy[data_type][f"pos_{pos}"]:
                        all_entropy[data_type][f"pos_{pos}"][layer_ind] = []
                    all_entropy[data_type][f"pos_{pos}"][layer_ind].append(type_entropy_avg_for_position[layer_ind][pos].item())
            
            
            
            # type_entropy_avg = type_entropy_avg / len(datas)
            # all_entropy[key].append(type_entropy_avg)
            
    base_dir = os.path.join(base_dir, "entropy_analysis")
    
    for pos in ["pos_0", "pos_1", "pos_2"]:
        os.makedirs(os.path.join(base_dir, dataset, pos), exist_ok=True)
        for layer_ind in range(1, 9):
            # Plotting the accuracy values in the same plot
            plt.figure(figsize=(10, 6))
            x_values = [int(checkpoint_step.split("-")[-1]) for checkpoint_step in all_checkpoints]
        
            colors = [
                "black", "darkgray", "orange", "yellow",
                "teal", "cyan", "darkblue", "brown", "red"
            ]

            # Plot each list with different markers and labels
            for i, key in enumerate(all_entropy.keys()):
                if pos in all_entropy[key]:
                    plt.plot(x_values, all_entropy[key][pos][layer_ind], label=f"{key}", color=colors[i], marker="o")

            # Set x-axis to log scale and customize ticks
            plt.xscale('symlog')
            xticks = [int(args.step_list[0]), int(args.step_list[-2])]
            plt.xticks(xticks, labels=[f"{int(tick)}" for tick in xticks])
            # plt.xticks([1250, 12500, 30000], labels=[f"{int(tick)}" for tick in [1250, 12500, 30000]])
            
            # Labeling the plot
            plt.title(f"Entropy for {pos} & Layer {layer_ind}")
            plt.xlabel("Step (Log Scale)")
            plt.ylabel("Entropy")
            plt.legend(loc="upper right")
            plt.grid(True)
            plt.tight_layout()

            # Display the plot
            plt.savefig(os.path.join(base_dir, dataset, pos, f"entropy_layer-{layer_ind}.png"), format="png", dpi=300)
            plt.close()
    

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=None, type=str, help="parent directory of saved model checkpoints")
    parser.add_argument("--step_list", default=None, nargs="+", help="checkpoint's steps to check causal strength")
    parser.add_argument("--num_layer", default=8, type=int, help="number of layer of the model")
    
    args = parser.parse_args()
    
    main(args)
    


