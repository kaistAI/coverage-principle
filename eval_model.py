import json
import os
import torch
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from simpletransformers.seq2seq import Seq2SeqModel
from torch.utils.data import DataLoader, SequentialSampler
import argparse

def parse_tokens(text):
    """
    주어진 문자열을 "<"와 ">"를 기준으로 분리하여 토큰 리스트를 생성합니다.
    예: "<t_5><t_23><t_17><t_42><t_33></a>" -> ["t_5", "t_23", "t_17", "t_42", "t_33"]
    """
    tokens = text.replace("</a>", "").strip("><").split("><")
    return tokens


def read_data_source_target(file_name, return_num=False, return_json=False, is_train=True):
    """
    file_name: a .json file containing a list of items, each has 'input_text', 'target_text', as keys
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if return_json:
        if return_num:
            return data, len(data)
        return data

    if is_train:
        keys = [key for key in data[0].keys()]
    else:
        keys = [key for key in data[0].keys()]
        
    source_target_pair = []
    for item in data:
        if is_train:
            instance = [item[key] for key in keys]
            instance.append("train_inferred")
            source_target_pair.append(instance)
        else:
            source_target_pair.append([item[key] for key in keys])
            
    if is_train:
        keys.append("type")

    if return_num:
        return pd.DataFrame(source_target_pair, columns=keys), len(data)
    return pd.DataFrame(source_target_pair, columns=keys)

# def create_masked_dataset(input_file, output_file):
#     # Load the original dataset
#     with open(input_file, 'r') as f:
#         data = json.load(f)
    
#     # Create masked dataset
#     masked_data = []
#     for item in data:
#         tokens = parse_tokens(item['target_text'])
#         changed_input_text = "".join([f"<{token}>" for token in tokens[:5]])
#         # changed_input_text = item['input_text'] + '<mask>'
#         changed_target_text = changed_input_text + f"<{tokens[5]}>" + "</a>"
        
#         masked_item = {
#             'input_text': changed_input_text,
#             'target_text': changed_target_text,
#             'type': item['type']
#         }
#         masked_data.append(masked_item)
    
#     # Save the masked dataset
#     with open(output_file, 'w') as f:
#         json.dump(masked_data, f, indent=2)

def evaluate_model(model_path, train_df, test_df, output_dir):
    
    # Load the model
    model = Seq2SeqModel(
        model_type='gpt2',
        model_name=model_path,
        args={
            'max_length': 8,  # We only care about the first 6 tokens
            "train_batch_size": 2048,
            'eval_batch_size': 1024,
            'fp16': True
        },
        ddp_args={
            'local_rank': -1,
            'rank': -1,
            'gpu': None,
            'world_size': 1,
            'dist_url': 'env://',
            'dist_backend': 'nccl',
        }
    )
    # Create dataset and dataloader
    train_dataset = model.load_and_cache_examples(train_df, evaluate=True)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=model.args.eval_batch_size
    )
    eval_dataset = model.load_and_cache_examples(test_df, evaluate=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=model.args.eval_batch_size,
    )
    
    # Evaluate the model
    model._move_model_to_device()
    
    # 디버깅을 위한 배치 처리 확인
    # for batch_idx, batch in enumerate(train_dataloader):
    #     if batch_idx == 0:  # 첫 번째 배치만 확인
    #         print("First batch sample:")
    #         print("Input IDs shape:", batch['target_ids'].shape)
    #         print("Labels shape:", batch['lm_labels'].shape)
    #         print("Sample input IDs:", batch['target_ids'][0])
    #         print("Sample labels:", batch['lm_labels'][0])
    #         print(f"model.tokenizer: {model.lm_tokenizer.batch_decode(batch['target_ids'])}")
    #         break
    
    # for batch_idx, batch in enumerate(eval_dataloader):
    #     if batch_idx == 0:  # 첫 번째 배치만 확인
    #         print("First batch sample:")
    #         print("Input IDs shape:", batch['target_ids'].shape)
    #         print("Labels shape:", batch['lm_labels'].shape)
    #         print("Sample input IDs:", batch['target_ids'][0])
    #         print("Sample labels:", batch['lm_labels'][0])
    #         print(f"model.tokenizer: {model.lm_tokenizer.batch_decode(batch['target_ids'])}")
    #         break
        
    train_results = model.eval_model(train_dataloader)
    print("Raw train results:", train_results)
    
    test_results = model.eval_model(eval_dataloader)
    print("Raw results:", test_results)
    
    return train_results, test_results

def main():
    
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_name", required=True, help="Path to the model checkpoint")
    
    args = parser.parse_args()
    
    models_path = os.path.join("/mnt/nas/hoyeon/GrokkedTransformer/trained_checkpoints/", args.ckpt_name)
    
    dataset_name = args.ckpt_name.split("_")[0]
    dataset_path = os.path.join("/mnt/sda/hoyeon/GrokkedTransformer/data", dataset_name)
    output_dir = '/home/jinho/repos/GrokkedTransformer/eval_results'
    
    all_checkpoints = [
        checkpoint_name for checkpoint_name in os.listdir(models_path)
        if checkpoint_name.startswith("checkpoint-") or checkpoint_name == "final_checkpoint"
    ]
    
    all_checkpoints.sort(key=lambda x: float('inf') if x == "final_checkpoint" else int(x.split("-")[1]))
    print(f"Found checkpoints: {all_checkpoints}")
    
     # Load train data
    train_df = read_data_source_target(os.path.join(dataset_path, "train.json"), is_train=True)
    # Load test data
    test_df = read_data_source_target(os.path.join(dataset_path, "test.json"), is_train=False)
    
    result = {}
    for checkpoint in all_checkpoints:
        result_for_ckpt = {}
        model_path = os.path.join(models_path, checkpoint)
        print(f"Current model path: {model_path}")
        
        # Evaluate model
        train_results, test_results = evaluate_model(model_path, train_df, test_df, output_dir)
        result_for_ckpt.update(train_results)
        result_for_ckpt.update(test_results)
        result[checkpoint] = result_for_ckpt

    # Save results
    for checkpoint, res in result.items():
        print("#"*50)
        print(f"{checkpoint:^50}")
        for key, value in res.items():
            print(f"{key}: Loss={value[0]:.4f}, Accuracy={value[1]:.4f}")
    
    print("Save Evaluation Results:")
    result_file_name = os.path.join("/home/jinho/repos/GrokkedTransformer/eval_results", f"{args.ckpt_name}.json")
    with open(result_file_name, "w") as f:
        json.dump(result, f, indent=4)
    
    
    

if __name__ == '__main__':
    main()
    