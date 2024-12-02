import argparse
import os
import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

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

    save_dir = os.path.join(args.model_dir, f"{args.top_k}-prob_distribution")
    os.makedirs(save_dir, exist_ok=True)

    for checkpoint in tqdm(all_checkpoints):
        print("now checkpoint", checkpoint)
        model_path = os.path.join(args.model_dir, checkpoint)
        model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left" 
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        
        prob_for_data_type = dict()
        for key in test_dict.keys():
            prob_for_data_type[key] = {
                "prob": torch.zeros(model.lm_head.weight.shape[0]),
                "num": 0
            }
            
        vocab_list = [data[0] for data in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])]
        added_vocab_list = vocab_list[50257:]
        
        for key, datas in test_dict.items():
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
                
                last_logits = logits[:,-1,:]
                probs = F.softmax(last_logits, dim=-1)
                probs_for_batch = torch.sum(probs, dim=0).squeeze().cpu()
                assert probs_for_batch.shape == prob_for_data_type[key]["prob"].shape
                prob_for_data_type[key]["prob"] = prob_for_data_type[key]["prob"] + probs_for_batch
                prob_for_data_type[key]["num"] = prob_for_data_type[key]["num"] + len(batch)
        
        for key, item in prob_for_data_type.items():
            prob_for_data_type[key] = item["prob"] / item["num"]
        
        # Prepare data for plotting
        top_probs_for_data_type = dict()
        largest_token_n_index_list = []
        largest_prob: float = 0
        for key, item in prob_for_data_type.items():
            top_probs, top_indices = torch.topk(item, args.top_k)
            prob_n_index_list = []
            for prob, index in zip(top_probs, top_indices):
                prob_n_index_list.append((prob.item(), index.item()-50257))
                if key != "test_nonsenses" and prob > largest_prob:
                    largest_prob = prob.item()
            prob_n_index_list = sorted(prob_n_index_list, key=lambda item: item[1])
            top_probs_for_data_type[key] = prob_n_index_list
            largest_token_n_index_list.append((added_vocab_list[top_indices[0].item()-50257], top_indices[0].item()-50257))

        largest_token_n_index_list = sorted(list(set(largest_token_n_index_list)), key=lambda item: item[1])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 2]})

        ax1.spines['bottom'].set_visible(False)
        ax1.set_ylabel("Probability (Clipped)")
        ax1.set_xlabel("Tokens")
        ax1.set_ylim(largest_prob, 1.1)
        ax1.set_xlim(0, largest_token_n_index_list[-1][1])
        ax1.get_xaxis().set_visible(False)
        ax1.set_title("Top 10 Probabilities (Clipped at Threshold)")
        ax1.yaxis.grid()

        ax2.spines['top'].set_visible(False)
        ax2.set_xlabel("Tokens")
        ax2.set_ylim(0, largest_prob + 0.0005)
        ax2.set_xlim(0, largest_token_n_index_list[-1][1])
        ax2.set_xticks([data[1] for data in largest_token_n_index_list], labels=[f"{data[0]}" for data in largest_token_n_index_list])
        ax2.yaxis.grid()

        d = 0.7

        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=15, linestyle="none", color='k', clip_on=False)

        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

        colors = [
            "black", "darkgray", "orange", "yellow",
            "teal", "cyan", "darkblue", "brown", "red"
        ]

        # Plot each list with different markers and labels
        for i, key in enumerate(top_probs_for_data_type.keys()):
            indices = [data[1] for data in top_probs_for_data_type[key]]
            probs = [data[0] for data in top_probs_for_data_type[key]]
            if key != "test_nonsenses":
                ax2.plot(indices, np.clip(probs, None, largest_prob), label=f"{key}", color=colors[i], marker="o")
                for _, index in largest_token_n_index_list:
                    ax2.axvline(index, ymax=1, color='black', linestyle='--', linewidth=0.2)
            else:
                filtered_top_probs_for_data_type = [data for data in top_probs_for_data_type[key] if data[0] > largest_prob]
                indices = [data[1] for data in filtered_top_probs_for_data_type]
                probs = [data[0] for data in filtered_top_probs_for_data_type]
                ax1.plot(indices, probs, label=f"{key}", color=colors[i], marker="o")
                for _, index in largest_token_n_index_list:
                    ax1.axvline(index, color='black', linestyle='--', linewidth=0.2)
                lower_top_probs_for_data_type = [data for data in top_probs_for_data_type[key] if data[0] <= largest_prob]
                indices = [data[1] for data in lower_top_probs_for_data_type]
                probs = [data[0] for data in lower_top_probs_for_data_type]
                ax2.plot(indices, probs, label=f"{key}", color=colors[i], marker="o")

        fig.legend(loc="upper left", bbox_to_anchor=(0.13,0.88))
        plt.savefig(os.path.join(save_dir, f"{checkpoint}.png"), format="png", dpi=300)



if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=None, type=str, help="parent directory of saved model checkpoints")
    parser.add_argument("--top_k", default=10, type=int, help="Top-k tokens you want to represent in figure")
    
    args = parser.parse_args()
    
    main(args)