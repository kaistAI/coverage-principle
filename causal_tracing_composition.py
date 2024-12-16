import numpy as np
import json, jsonlines
import matplotlib.pyplot as plt
from eval_qa import eval_file, eval_items
import os
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pandas as pd
from copy import deepcopy
import random
import argparse
from typing import List

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", default=None, type=str, help="parent directory of saved model checkpoints")
    parser.add_argument("--step_list", default=None, nargs="+", help="checkpoint's steps to check causal strength")
    parser.add_argument("--save_path", default=None, type=str, help="path to save result")

    parser.add_argument("--num_layer", default=8, type=int, help="number of layer of the model")
    parser.add_argument("--target_layer", default=None, type=int, help="Target layer number for checking causal tracing")
    parser.add_argument("--data_type", default="train_inferred", type=str, help="type of inference data (atomic_id, atomic_ood, train_inferred, test_inferred_iid, test_inferred_ood, ...)")
    
    args = parser.parse_args()
    
    # Sanity check for composition & comparison
    assert "comparison" not in args.model_dir
    
    if args.model_dir.split("/")[-1] == "":
        dataset = args.model_dir.split("/")[-2].split("_")[0]
    else:
        dataset = args.model_dir.split("/")[-1].split("_")[0]

    device = torch.device('cuda:0')
    np.random.seed(0)
    random.seed(0)

    all_atomic = set()     # (h,r,t)
    atomic_dict = dict()   # (h,r) -> t
    if "inf" in dataset:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "atomic_facts.json")) as f:
            atomic_items = json.load(f)
    else:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "train.json")) as f:
            atomic_items = json.load(f)
 
    for item in tqdm(atomic_items):
        temp = item['target_text'].strip("><").split("><")
        if len(temp) != 4:
            continue
        h,r,t = temp[:3]
        atomic_dict[(h,r)] = t
        all_atomic.add((h,r,t))

    id_atomic = set()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "train.json")) as f:
        train_items = json.load(f)

    for item in tqdm(train_items):
        temp = item['target_text'].strip("><").split("><")
        if len(temp) == 4:
            continue
        h, r1, r2, t = temp[:4]
        b = atomic_dict[(h, r1)]
        assert atomic_dict[(b, r2)] == t
        id_atomic.add((h,r1,b))
        id_atomic.add((b,r2,t))

    ood_atomic = all_atomic - id_atomic
    print("# id_atomic, # ood_atomic:", len(id_atomic), len(ood_atomic))

    h2rt_train = dict()
    for (h,r,t) in id_atomic:
        if h not in h2rt_train:
            h2rt_train[h] = []
        h2rt_train[h].append((r,t))

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "test.json")) as f:
        pred_data = json.load(f)
    
    d = dict()
    for item in pred_data:
        t = item['type']
        if t not in d:
            d[t] = []
        d[t].append(item)

    def return_rank(hd, word_embedding_, token_ids_list, metric='dot', token_list=None):
        if metric == 'dot':
            word_embedding = word_embedding_
        elif metric == 'cos':
            word_embedding = F.normalize(word_embedding_, p=2, dim=1)
        else:
            assert False

        logits_ = torch.matmul(hd, word_embedding.T)
        batch_size, seq_len, vocab_size = logits_.shape[0], logits_.shape[1], logits_.shape[2]
        
        token_ids_list = torch.tensor(token_ids_list).view(batch_size, 1, 1).expand(batch_size, seq_len, vocab_size).to(logits_.device) 
    
        _, sorted_indices = logits_.sort(dim=-1, descending=True)
        rank = (sorted_indices == token_ids_list).nonzero(as_tuple=True)[-1].view(batch_size, seq_len).cpu()

        return rank

    all_checkpoints = [checkpoint for checkpoint in os.listdir(args.model_dir) if checkpoint.startswith("checkpoint")]
    assert all(os.path.isdir(os.path.join(args.model_dir, checkpoint)) for checkpoint in all_checkpoints)
    all_checkpoints.sort(key=lambda var: int(var.split("-")[1]))

    results = {}
    
    split = args.data_type
    # rand_inds = np.random.choice(len(d[split]), 300, replace=False).tolist()
    target_layer = args.target_layer
    total_layer = args.num_layer
    
    BATCH_SIZE = 4096

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

        full_list = []
        for i in range(0, len(d[split]), BATCH_SIZE):
            batch = d[split][i:i+BATCH_SIZE]
            batch_size = len(batch)
            query_list = [data['input_text'] for data in batch]
            temp_dict = dict()
            
            real_h_r1_r2_list = [query.strip("><").split("><") for query in query_list]
            real_b_list, real_t_list, real_r2_list = [], [], []
            for h, r1, r2 in real_h_r1_r2_list:
                # To Do : test_nonsenses is not in atomic_dict
                b = atomic_dict[(h, r1)]
                real_b_list.append(b)
                real_r2_list.append(r2)
                real_t_list.append(atomic_dict[(b, r2)])
            
            tokenizer_output = tokenizer(query_list, return_tensors="pt", padding=True)
            input_ids, attention_mask = tokenizer_output["input_ids"], tokenizer_output["attention_mask"]
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
            all_hidden_states = outputs['hidden_states']
            
            rank_before = return_rank(all_hidden_states[target_layer], word_embedding, tokenizer([f"<{target}>" for target in real_t_list])["input_ids"])[:, -1].tolist()
            temp_dict['rank_before'] = rank_before
            
            # MRRs
            for layer_ind in range(1, total_layer):
                hidden_states_orig = all_hidden_states[layer_ind]
                with torch.no_grad():
                    temp = model.transformer.ln_f(hidden_states_orig)
                    
                temp_dict['b_rank_pos1_'+str(layer_ind)] = return_rank(temp, word_embedding, tokenizer([f"<{target}>" for target in real_b_list])["input_ids"])[:, 1].tolist()
                temp_dict['r2_rank_pos2_'+str(layer_ind)] = return_rank(temp, word_embedding, tokenizer([f"<{target}>" for target in real_r2_list])["input_ids"])[:, 2].tolist()
                temp_dict['t_rank_pos2_'+str(layer_ind)] = return_rank(temp, word_embedding, tokenizer([f"<{target}>" for target in real_t_list])["input_ids"])[:, 2].tolist()
            
            if checkpoint in [f"checkpoint-{step}" for step in args.step_list]:
                # perturb the head entity
                all_list = [set() for _ in range(batch_size)]
                for i, (h, r1, r2) in enumerate(real_h_r1_r2_list):
                    assert (h, r1) in atomic_dict
                    for head in h2rt_train:
                        if (head, r1) not in atomic_dict:
                            all_list[i].add(head)
                            continue
                        bridge = atomic_dict[(head, r1)]
                        # tail = atomic_dict[(real_b_list[i], r2)]
                        if (bridge, r2) not in atomic_dict or atomic_dict[(bridge, r2)] != real_t_list[i]:
                            all_list[i].add(head)
                query_list = []
                for i in range(batch_size):
                    query_list.append(f'<{random.choice(sorted(list(all_list[i]), key=lambda x : int(x.strip("><").split("_")[-1])))}>')
                
                tokenizer_output = tokenizer(query_list, return_tensors="pt", padding=True)
                input_ids_, attention_mask = tokenizer_output["input_ids"], tokenizer_output["attention_mask"]
                input_ids_, attention_mask = input_ids_.to(device), attention_mask.to(device)
                
                with torch.no_grad():
                    outputs_ctft = model(
                        input_ids=input_ids_,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                all_hidden_states_ctft = outputs_ctft['hidden_states']
                
                for layer_to_intervene in range(1, target_layer):
                    hidden_states = all_hidden_states[layer_to_intervene].clone()
                    hidden_states_ctft = all_hidden_states_ctft[layer_to_intervene]
                    # intervene
                    hidden_states[:, 0, :] = hidden_states_ctft[:, 0, :]

                    with torch.no_grad():
                        for i in range(layer_to_intervene, target_layer):
                            f_layer = model.transformer.h[i]
                            # attn
                            residual = hidden_states
                            hidden_states = f_layer.ln_1(hidden_states)
                            attn_output = f_layer.attn(hidden_states)[0] 
                            hidden_states = attn_output + residual
                            # mlp
                            residual = hidden_states
                            hidden_states = f_layer.ln_2(hidden_states)
                            feed_forward_hidden_states = f_layer.mlp.c_proj(f_layer.mlp.act(f_layer.mlp.c_fc(hidden_states)))
                            hidden_states = residual + feed_forward_hidden_states
                        # final ln
                        hidden_states = model.transformer.ln_f(hidden_states)
                    # print("--------")
                    rank_after = return_rank(hidden_states, word_embedding, tokenizer([f"<{target}>" for target in real_t_list])["input_ids"])[:, -1].tolist()
                    temp_dict['h_'+str(layer_to_intervene)] = rank_after
                
                # perturb the 1st relation
                all_list = [set() for _ in range(batch_size)]
                query_list = []
                for i, (h, r1, r2) in enumerate(real_h_r1_r2_list):
                    rt_list = h2rt_train[h]
                    for (relation, bridge) in rt_list:
                        if (bridge, r2) not in atomic_dict or atomic_dict[(bridge, r2)] != real_t_list[i]:
                            all_list[i].add(relation)
                    query_list.append(f"<{h}><{random.choice(list(all_list[i]))}>")

                tokenizer_output = tokenizer(query_list, return_tensors="pt", padding=True)
                input_ids_, attention_mask = tokenizer_output["input_ids"], tokenizer_output["attention_mask"]
                input_ids_, attention_mask = input_ids_.to(device), attention_mask.to(device)
                with torch.no_grad():
                    outputs_ctft = model(
                        input_ids=input_ids_,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                all_hidden_states_ctft = outputs_ctft['hidden_states']

                for layer_to_intervene in range(1, target_layer):
                    hidden_states = all_hidden_states[layer_to_intervene].clone()
                    hidden_states_ctft = all_hidden_states_ctft[layer_to_intervene]
                    # intervene
                    hidden_states[:, 1, :] = hidden_states_ctft[:, 1, :]

                    with torch.no_grad():
                        for i in range(layer_to_intervene, target_layer):
                            f_layer = model.transformer.h[i]
                            # attn
                            residual = hidden_states
                            hidden_states = f_layer.ln_1(hidden_states)
                            attn_output = f_layer.attn(hidden_states)[0] 
                            hidden_states = attn_output + residual
                            # mlp
                            residual = hidden_states
                            hidden_states = f_layer.ln_2(hidden_states)
                            feed_forward_hidden_states = f_layer.mlp.c_proj(f_layer.mlp.act(f_layer.mlp.c_fc(hidden_states)))
                            hidden_states = residual + feed_forward_hidden_states
                        # final ln
                        hidden_states = model.transformer.ln_f(hidden_states)
                    # print("--------")
                    rank_after = return_rank(hidden_states, word_embedding, tokenizer([f"<{target}>" for target in real_t_list])["input_ids"])[:, -1].tolist()
                    temp_dict['r1_'+str(layer_to_intervene)] = rank_after
                # print(temp_dict)
                
                # perturb the second relation
                all_list = [set() for _ in range(batch_size)]
                query_list = []
                for i, (h, r1, r2) in enumerate(real_h_r1_r2_list):
                    b = real_b_list[i]
                    rt_list = h2rt_train[b]
                    t = real_t_list[i]
                    for (relation, tail) in rt_list:
                        if tail != t:
                            assert relation != r2
                            all_list[i].add(relation)
                    query_list.append(f"<{h}><{r1}><{random.choice(list(all_list[i]))}>")

                decoder_temp = tokenizer(query_list, return_tensors="pt", padding=True)
                decoder_input_ids_, decoder_attention_mask = decoder_temp["input_ids"], decoder_temp["attention_mask"]
                decoder_input_ids_, decoder_attention_mask = decoder_input_ids_.to(device), decoder_attention_mask.to(device)

                with torch.no_grad():
                    outputs_ctft = model(
                        input_ids=decoder_input_ids_,
                        attention_mask=decoder_attention_mask,
                        output_hidden_states=True
                    )
                all_hidden_states_ctft = outputs_ctft['hidden_states']

                for layer_to_intervene in range(1, target_layer):
                    hidden_states = all_hidden_states[layer_to_intervene].clone()
                    hidden_states_ctft = all_hidden_states_ctft[layer_to_intervene]
                    # intervene
                    hidden_states[:, 2, :] = hidden_states_ctft[:, 2, :]

                    with torch.no_grad():
                        for i in range(layer_to_intervene, target_layer):
                            f_layer = model.transformer.h[i]
                            # attn
                            residual = hidden_states
                            hidden_states = f_layer.ln_1(hidden_states)
                            attn_output = f_layer.attn(hidden_states)[0] 
                            hidden_states = attn_output + residual
                            # mlp
                            residual = hidden_states
                            hidden_states = f_layer.ln_2(hidden_states)
                            feed_forward_hidden_states = f_layer.mlp.c_proj(f_layer.mlp.act(f_layer.mlp.c_fc(hidden_states)))
                            hidden_states = residual + feed_forward_hidden_states
                        # final ln
                        hidden_states = model.transformer.ln_f(hidden_states)
                    rank_after = return_rank(hidden_states, word_embedding, tokenizer([f"<{target}>" for target in real_t_list])["input_ids"])[:, -1].tolist()
                    temp_dict['r2_'+str(layer_to_intervene)] = rank_after
            
            result_dict_list = [dict() for _ in range(batch_size)]
            for key, value_list in temp_dict.items():
                for i in range(batch_size):
                    result_dict_list[i][key] = value_list[i]
                    
            full_list = full_list + result_dict_list
            
        results[checkpoint] = full_list

    with open(os.path.join(args.save_path, f"{args.model_dir.split('/')[-1]}-{args.data_type}.json"), "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()