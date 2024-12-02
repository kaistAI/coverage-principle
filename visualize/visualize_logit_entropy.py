import argparse
import os
import json
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
import matplotlib.pyplot as plt


def calculate_entropy(logits):
    
    # Only calculate tail entity's logit
    logits = logits[:,-1,:]
    
    probabilities = F.softmax(logits, dim=-1)
    
    # Calculate entropy
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1)  # Avoid log(0)
    total_entropy = torch.sum(entropy).cpu().item()
    return total_entropy


def main(args):
    if args.model_dir.split("/")[-1] == "":
        dataset = args.model_dir.split("/")[-2].split("_")[0]
    else:
        dataset = args.model_dir.split("/")[-1].split("_")[0]
    
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", dataset, "test.json")) as f:
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
    all_checkpoints = all_checkpoints
    print(all_checkpoints)
    
    device = torch.device('cuda:0')
    BATCH_SIZE = 65536
    
    
    all_entropy = dict()
    for key in test_dict.keys():
        all_entropy[key] = []
    
    for checkpoint in tqdm(all_checkpoints):
        print("now checkpoint", checkpoint)
        model_path = os.path.join(args.model_dir, checkpoint)
        model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left" 
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        
        for key, datas in test_dict.items():
            type_entropy_avg = 0.0
            for i in range(0, len(datas), BATCH_SIZE):
                batch = datas[i:i+BATCH_SIZE]
                queries = [data['input_text'] for data in batch]
                labels = [data['target_text'] for data in batch]
                tokenized_output = tokenizer(queries, return_tensors="pt", padding=True)
                tokenized_input_ids, attention_mask = tokenized_output["input_ids"], tokenized_output["attention_mask"]
                tokenized_input_ids, attention_mask = tokenized_input_ids.to(device), attention_mask.to(device)
                
                with torch.no_grad():
                    outputs = model(
                        input_ids=tokenized_input_ids,
                        attention_mask=attention_mask
                    )
                    logits = outputs.logits

                # Calculate entropy for the logits
                batch_entropy = calculate_entropy(logits)
                type_entropy_avg += batch_entropy
            type_entropy_avg = type_entropy_avg / len(datas)
            all_entropy[key].append(type_entropy_avg)
            
    print(all_entropy)
    
    # Plotting the accuracy values in the same plot
    plt.figure(figsize=(10, 6))
    x_values = [int(checkpoint_step.split("-")[-1]) for checkpoint_step in all_checkpoints]
    
    colors = [
        "black", "darkgray", "orange", "yellow",
        "teal", "cyan", "darkblue", "brown", "red"
    ]

    # Plot each list with different markers and labels
    for i, key in enumerate(all_entropy.keys()):
        plt.plot(x_values, all_entropy[key], label=f"{key}", color=colors[i], marker="o")

    # Set x-axis to log scale and customize ticks
    # plt.xscale("log")
    plt.xscale('symlog')
    plt.xticks([1250, 12500, 30000], labels=[f"{int(tick)}" for tick in [1250, 12500, 30000]])
    

    # Labeling the plot
    plt.xlabel("Step (Log Scale)")
    plt.ylabel("Entropy")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()

    # Display the plot
    plt.savefig(os.path.join(args.model_dir, "entropy_for_nonsense.png"), format="png", dpi=300)
    

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=None, type=str, help="parent directory of saved model checkpoints")
    
    args = parser.parse_args()
    
    main(args)
    


