import argparse
import os
import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np



def main(args):
    
    if args.model_dir.split("/")[-1] == "":
        dataset = args.model_dir.split("/")[-2].split("_")[0]
    else:
        dataset = args.model_dir.split("/")[-1].split("_")[0]
        
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", dataset, "test.json")) as f:
        test_data = json.load(f)
        
    test_dict = {
        "Train(ID)_atomic": [],
        "Train(ID)_inferred": [],
        "Test(ID)": [],
        "Test(OOD)": []
    }
    
    for item in test_data:
        t = item['type']
        if t in ["id_atomic", "ood_atomic"]:
            test_dict["Train(ID)_atomic"].append(item)
        elif t == "train_inferred":
            test_dict["Train(ID)_inferred"].append(item)
        elif t == "test_inferred_iid":
            test_dict["Test(ID)"].append(item)
        elif t == "test_inferred_ood":
            test_dict["Test(OOD)"].append(item)
        else:
            pass

    for key, datas in test_dict.items():
        print(f"{key}: {len(datas)}")
        
    all_checkpoints = [checkpoint for checkpoint in os.listdir(args.model_dir) if checkpoint.startswith("checkpoint")]
    all_checkpoints.sort(key=lambda var: int(var.split("-")[1]))
    print(all_checkpoints)
    
    device = torch.device('cuda:3')
    # device = torch.device('cpu')
    BATCH_SIZE = 65536
    
    all_acc = dict()
    for key in test_dict.keys():
        all_acc[key] = []
    
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
            type_accuracy = 0.0
            for i in range(0, len(datas), BATCH_SIZE):
                batch = datas[i:i+BATCH_SIZE]
                queries = [data['input_text'] for data in batch]
                labels = [data['target_text'] for data in batch]
                decoder_temp = tokenizer(queries, return_tensors="pt", padding=True)
                decoder_input_ids = decoder_temp["input_ids"]
                decoder_input_ids = decoder_input_ids.to(device)
                
                with torch.no_grad():
                    token_id_outputs = model.generate(
                        input_ids=decoder_input_ids,
                        max_new_tokens=2,
                    )
                token_outputs = [output.replace(" ", "") for output in tokenizer.batch_decode(token_id_outputs)]
                for label, prediction in zip(labels, token_outputs):
                    if label == prediction:
                        type_accuracy += 1
                        
            type_accuracy = (type_accuracy / len(datas))
            all_acc[key].append(type_accuracy)
            
    print(all_acc)

    # Plotting the accuracy values in the same plot
    plt.figure(figsize=(10, 6))
    x_values = [int(checkpoint_step.split("-")[-1]) for checkpoint_step in all_checkpoints]

    # Plot each list with different markers and labels
    plt.plot(x_values, all_acc["Train(ID)_atomic"], label="Train(ID)_Atomic", linestyle='-', marker='o')
    plt.plot(x_values, all_acc["Train(ID)_inferred"], label="Train(ID)_Inferred", linestyle='-', marker='x')
    plt.plot(x_values, all_acc["Test(ID)"], label="Test(ID)", linestyle='-', marker='s')
    plt.plot(x_values, all_acc["Test(OOD)"], label="Test(OOD)", linestyle='-', marker='^')

    # Set x-axis to log scale and customize ticks
    plt.xscale("log")
    plt.xticks([1e4, 5e4, 1e5])
    
    # Labeling the plot
    plt.xlabel("Step (Log Scale)")
    plt.ylabel("Accuracy")
    # plt.title("Accuracy Values Across Different Sets")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    # Display the plot
    plt.savefig(os.path.join(args.model_dir, "accuracy.png"), format="png", dpi=300)



if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=None, type=str, help="parent directory of saved model checkpoints")
    
    args = parser.parse_args()
    
    main(args)