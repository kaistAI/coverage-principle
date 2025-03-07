import argparse
import torch
import json
from tqdm import tqdm
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging
from collections import defaultdict
import re

import sys
# sys.path.append(os.path.dirname(__file__))

from utils import parse_tokens, load_atomic_facts_3hop, setup_logging, deduplicate_grouped_data

###############################################################################
# 2) Helpers for 3-Hop Input & Grouping
###############################################################################

def group_data_by_b1(examples, f1_dict):
    """
    Returns a dictionary mapping b1 to a list of examples.
    """
    group_dict = defaultdict(list)
    for ex in examples:
        inp_tokens = parse_tokens(ex["input_text"])
        assert len(inp_tokens) == 4
        t1, t2, _, _ = inp_tokens
        b1 = f1_dict.get((t1, t2))
        group_dict[b1].append(ex)
    return dict(group_dict)

def group_data_by_b2(examples, f1_dict, f2_dict):
    """
    Returns a dictionary mapping b2 to a list of examples.
    """
    group_dict = defaultdict(list)
    for ex in examples:
        inp_tokens = parse_tokens(ex["input_text"])
        assert len(inp_tokens) == 4
        t1, t2, t3, _ = inp_tokens
        b1 = f1_dict.get((t1, t2))
        b2 = f2_dict.get((b1, t3))
        group_dict[b2].append(ex)
    return dict(group_dict)

def group_data_by_t_final(examples, f1_dict, f2_dict, f3_dict):
    """
    Returns a dictionary mapping t_final to a list of examples.
    """
    group_dict = defaultdict(list)
    for ex in examples:
        inp_tokens = parse_tokens(ex["input_text"])
        assert len(inp_tokens) == 4
        t1, t2, t3, t4 = inp_tokens
        b1 = f1_dict.get((t1, t2))
        b2 = f2_dict.get((b1, t3))
        t_final = f3_dict.get((b2, t4))
        group_dict[t_final].append(ex)
    return dict(group_dict)


###############################################################################
# 3) Deduplication and Hooking Logic (Improved Deduplication)
###############################################################################
def deduplicate_vectors(results):
    """
    Deduplicate vectors within each target group and track removal statistics.
    Improvement: Quantize each hidden state vector (by rounding to a fixed precision)
    to a hashable tuple and use set membership for fast duplicate checks.
    """
    import numpy as np
    from collections import defaultdict
    
    dedup_stats = defaultdict(lambda: defaultdict(int))
    deduplicated_results = defaultdict(list)
    
    # 확인 필요 : 이렇게 해서 결과가 달라지진 않을까?
    def quantize_vector(vec, precision=7):
        # Round each element in the vector to the specified precision and convert to tuple
        return tuple(round(x, precision) for x in vec)
    
    def get_vector_key(hidden_state, precision=7):
        """
        Create a combined hashable key for a hidden state by quantizing available vectors
        (e.g., embedding, post_attention, post_mlp) and combining them into one tuple.
        """
        keys = []
        if 'embedding' in hidden_state:
            keys.append(quantize_vector(hidden_state['embedding'], precision))
        if 'post_attention' in hidden_state and hidden_state['post_attention'] is not None:
            keys.append(quantize_vector(hidden_state['post_attention'], precision))
        if 'post_mlp' in hidden_state and hidden_state['post_mlp'] is not None:
            keys.append(quantize_vector(hidden_state['post_mlp'], precision))
        return tuple(keys)
    
    for target, instances in results.items():
        seen_vectors = defaultdict(set)  # (layer, pos) -> set of vector keys
        logging.info(f"Performing deduplication for target {target}")
        if target == 'unknown':
            continue
        
        for instance in instances:
            is_duplicate = False
            for hidden_state in instance['hidden_states']:
                layer = hidden_state['layer']
                pos = hidden_state['position']
                vector_key = get_vector_key(hidden_state)
                if vector_key in seen_vectors[(layer, pos)]:
                    dedup_stats[target][f"layer{layer}_pos{pos}"] += 1
                    is_duplicate = True
                    break
                else:
                    seen_vectors[(layer, pos)].add(vector_key)
            if not is_duplicate:
                deduplicated_results[target].append(instance)
    
    final_stats = {k: dict(v) for k, v in dedup_stats.items()}
    return dict(deduplicated_results), final_stats


def load_and_preprocess_data(f1_dict, f2_dict, f3_dict, test_path, idx):
    """
    Parse test.json, filter examples by type, and group them using atomic facts
    """
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    id_train_data = []
    id_test_data = []
    ood_test_data = []
    
    for d in test_data:
        if d['type'] == 'train_inferred':
            id_train_data.append(d)
        elif d['type'] == 'type_0':
            id_test_data.append(d)
        elif d['type'] in set([f"type_{i}" for i in range(1, 8)]):
            if idx == 1:
                if d['type'] in ['type_1', 'type_2', 'type_3', 'type_4']:
                    ood_test_data.append(d)
            elif idx == 2:
                if d['type'] in ['type_1', 'type_2', 'type_5', 'type_6']:
                    ood_test_data.append(d)
            elif idx == 3:
                if d['type'] in ['type_1', 'type_3', 'type_5', 'type_7']:
                    ood_test_data.append(d)
            else:
                raise NotImplementedError(f"Invalid idx value: {idx}")
        else:
            raise NotImplementedError("Invalid coverage type")
            
    if idx == 1:
        grouped_id_train_data = group_data_by_b1(id_train_data, f1_dict)
        grouped_id_test_data = group_data_by_b1(id_test_data, f1_dict)
        grouped_ood_test_data = group_data_by_b1(ood_test_data, f1_dict)
    
    elif idx == 2:
        grouped_id_train_data = group_data_by_b2(id_train_data, f1_dict, f2_dict)
        grouped_id_test_data = group_data_by_b2(id_test_data, f1_dict, f2_dict)
        grouped_ood_test_data = group_data_by_b2(ood_test_data, f1_dict, f2_dict)
        
    elif idx == 3:
        grouped_id_train_data = group_data_by_t_final(id_train_data, f1_dict, f2_dict, f3_dict)
        grouped_id_test_data = group_data_by_t_final(id_test_data, f1_dict, f2_dict, f3_dict)
        grouped_ood_test_data = group_data_by_t_final(ood_test_data, f1_dict, f2_dict, f3_dict)
    else:
        raise NotImplementedError
    
    return grouped_id_train_data, grouped_id_test_data, grouped_ood_test_data


###############################################################################
# Batch Processing Functions
###############################################################################
def get_hidden_states_residual(model, input_texts, layer_pos_pairs, tokenizer, device):
    """
    Batch processing for residual stream hidden states.
    Tokenizes a list of input texts at once, performs model inference,
    and extracts the hidden state at the specified token positions for each instance.
    """
    inputs = tokenizer(input_texts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    all_hidden_states = outputs["hidden_states"]
    
    batch_hidden_states = []
    for i in range(len(input_texts)):
        instance_hidden_states = []
        for layer, pos in layer_pos_pairs:
            try:
                hs = all_hidden_states[layer]
                if hs.dim() == 3:
                    token_vec = hs[i, pos, :].detach().cpu().numpy()
                elif hs.dim() == 2:
                    token_vec = hs[pos, :].detach().cpu().numpy()
                else:
                    logging.warning(f"Unexpected shape for hidden state at layer {layer}: {hs.shape}")
                    token_vec = None
                instance_hidden_states.append({
                    'layer': layer,
                    'position': pos,
                    'post_attention': None,
                    'post_mlp': token_vec.tolist() if token_vec is not None else None
                })
            except Exception as e:
                logging.error(f"Error processing layer {layer}, position {pos} for batch instance {i}: {str(e)}")
                instance_hidden_states.append({
                    'layer': layer,
                    'position': pos,
                    'error': str(e)
                })
        batch_hidden_states.append(instance_hidden_states)
    return batch_hidden_states

def get_hidden_states_mlp(model, input_texts, layer_pos_pairs, tokenizer, device):
    """
    Batch processing for hook-based (MLP) mode.
    Registers hooks to capture activations for the entire batch,
    then extracts the output at the specified token positions for each instance.
    """
    inputs = tokenizer(input_texts, padding=True, return_tensors="pt").to(device)
    activation = {}
    
    def get_activation(name):
        def hook(module, inp, out):
            activation[name] = out
        return hook
    
    hooks = []
    for layer, pos in layer_pos_pairs:
        if layer > 0:
            hooks.append(model.transformer.h[layer-1].attn.register_forward_hook(get_activation(f'layer{layer}_attn')))
            hooks.append(model.transformer.h[layer-1].mlp.register_forward_hook(get_activation(f'layer{layer}_mlp')))
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    for h in hooks:
        h.remove()
        
    batch_hidden_states = []
    for i in range(len(input_texts)):
        instance_hidden_states = []
        for layer, pos in layer_pos_pairs:
            try:
                if layer == 0:
                    word_embeddings = model.transformer.wte(inputs['input_ids'])
                    token_vec = word_embeddings[i, pos, :].detach().cpu().numpy()
                    instance_hidden_states.append({
                        'layer': layer,
                        'position': pos,
                        'embedding': token_vec.tolist()
                    })
                else:
                    post_attn = activation[f'layer{layer}_attn']
                    post_mlp = activation[f'layer{layer}_mlp']
                    if isinstance(post_attn, tuple):
                        post_attn = post_attn[0]
                    if isinstance(post_mlp, tuple):
                        post_mlp = post_mlp[0]

                    if post_attn.dim() == 3:
                        token_attn = post_attn[i, pos, :].detach().cpu().numpy()
                    elif post_attn.dim() == 2:
                        token_attn = post_attn[pos, :].detach().cpu().numpy()
                    else:
                        logging.warning(f"Unexpected shape for attn at layer {layer}: {post_attn.shape}")
                        token_attn = None
                    if post_mlp.dim() == 3:
                        token_mlp = post_mlp[i, pos, :].detach().cpu().numpy()
                    elif post_mlp.dim() == 2:
                        token_mlp = post_mlp[pos, :].detach().cpu().numpy()
                    else:
                        logging.warning(f"Unexpected shape for mlp at layer {layer}: {post_mlp.shape}")
                        token_mlp = None
                    instance_hidden_states.append({
                        'layer': layer,
                        'position': pos,
                        'post_attention': token_attn.tolist() if token_attn is not None else None,
                        'post_mlp': token_mlp.tolist() if token_mlp is not None else None
                    })
            except Exception as e:
                logging.error(f"Error at layer {layer}, position {pos} for batch instance {i}: {str(e)}")
                instance_hidden_states.append({
                    'layer': layer,
                    'position': pos,
                    'error': str(e)
                })
        batch_hidden_states.append(instance_hidden_states)
    return batch_hidden_states

def process_data_group(model, data_group, layer_pos_pairs, tokenizer, device, mode, batch_size=8):
    """
    For each group (bridge entity), processes examples in batches of size 'batch_size'
    using batch inference.
    """
    results = defaultdict(list)
    for bridge_entity, instances in tqdm(data_group.items(), desc="Processing instances"):
        for i in range(0, len(instances), batch_size):
            batch = instances[i:i+batch_size]
            input_texts = [ex['input_text'] for ex in batch]
            if mode == "residual":
                batch_hidden_states = get_hidden_states_residual(model, input_texts, layer_pos_pairs, tokenizer, device)
            else:
                batch_hidden_states = get_hidden_states_mlp(model, input_texts, layer_pos_pairs, tokenizer, device)
            for ex, hs in zip(batch, batch_hidden_states):
                item = {
                    "input_text": ex['input_text'],
                    "target_text": ex['target_text'],
                    "identified_target": bridge_entity,
                    "type": ex.get('type'),
                    "hidden_states": hs
                }
                results[bridge_entity].append(item)
    return results


###############################################################################
# Main Function with Batch Processing and Improved Deduplication
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Model checkpoint path")
    parser.add_argument("--layer_pos_pairs", required=True, help="(layer, position) tuples")
    parser.add_argument("--save_dir", required=True, help="Directory to store analysis results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--atomic_idx", required=True, type=int, help="Bottleneck function index among f1, f2, and f3 used for collapse evaluation")
    parser.add_argument("--mode", required=True, help="Mode: 'post_mlp' or 'residual'")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size for processing")
    
    args = parser.parse_args()
    assert args.mode in ["post_mlp", "residual"]

    setup_logging(args.debug)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    if args.ckpt.split("/")[-1] == "":
        dataset, step = args.ckpt.split("/")[-3].split("_")[0], args.ckpt.split("/")[-2].split("-")[-1]
    else:
        dataset, step = args.ckpt.split("/")[-2].split("_")[0], args.ckpt.split("/")[-1].split("-")[-1]
    
    logging.info("Loading model and tokenizer...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(os.path.join(base_dir, args.ckpt)).to(device)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(base_dir, args.ckpt))
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    logging.info("Model and tokenizer loaded successfully")
    
    data_dir = os.path.join(base_dir, "data", dataset)
    atomic_file_1 = os.path.join(data_dir, f"atomic_facts_f1.json")
    atomic_file_2 = os.path.join(data_dir, f"atomic_facts_f2.json")
    atomic_file_3 = os.path.join(data_dir, f"atomic_facts_f3.json")
    # (t_N1, t_N2) -> t_N3
    f1_dict, f2_dict, f3_dict = load_atomic_facts_3hop(atomic_file_1, atomic_file_2, atomic_file_3)
    
    grouped_id_train_data, grouped_id_test_data, grouped_ood_test_data = load_and_preprocess_data(
        f1_dict, f2_dict, f3_dict, os.path.join(data_dir, "test.json"), idx=args.atomic_idx
    )
    
    layer_pos_pairs = eval(args.layer_pos_pairs)
    logging.info(f"Layer position pairs: {layer_pos_pairs}")
    
    logging.info(f"ID train targets: {len(grouped_id_train_data)}")
    logging.info(f"ID test targets: {len(grouped_id_test_data)}")
    logging.info(f"OOD test targets: {len(grouped_ood_test_data)}")
    
    torch.manual_seed(0)
    
    logging.info("Processing ID train group with batch processing...")
    id_train_results = process_data_group(model, grouped_id_train_data, layer_pos_pairs, tokenizer, device, args.mode, batch_size=args.batch_size)
    
    logging.info("Processing ID test group with batch processing...")
    id_test_results = process_data_group(model, grouped_id_test_data, layer_pos_pairs, tokenizer, device, args.mode, batch_size=args.batch_size)
    
    logging.info("Processing OOD test group with batch processing...")
    ood_test_results = process_data_group(model, grouped_ood_test_data, layer_pos_pairs, tokenizer, device, args.mode, batch_size=args.batch_size)
    
    # logging.info("Deduplicating ID train results...")
    # id_train_dedup, id_train_stats = deduplicate_vectors(id_train_results)
    
    # logging.info("Deduplicating ID test results...")
    # id_test_dedup, id_test_stats = deduplicate_vectors(id_test_results)
    
    # logging.info("Deduplicating OOD test results...")
    # ood_dedup, ood_stats = deduplicate_vectors(ood_test_results)
    
    logging.info("Deduplicating ID train results...")
    id_train_dedup = deduplicate_grouped_data(id_train_results, args.atomic_idx)
    
    logging.info("Deduplicating ID test results...")
    id_test_dedup = deduplicate_grouped_data(id_test_results, args.atomic_idx)
    
    logging.info("Deduplicating OOD test results...")
    ood_dedup = deduplicate_grouped_data(ood_test_results, args.atomic_idx)
    
    save_dir = os.path.join(args.save_dir, args.mode, dataset, f"f{args.atomic_idx}", f"{str(layer_pos_pairs[0]).replace(' ', '')}", step)
    if os.path.exists(save_dir):
        logging.info(f"{save_dir} already exists!")
        return
    else:
        os.makedirs(save_dir, exist_ok=True)


    id_train_save = os.path.join(save_dir, "id_train_dedup.json")
    with open(id_train_save, "w") as f:
        json.dump(id_train_dedup, f)
    logging.info(f"Saved deduplicated ID train results to {id_train_save}")

    id_test_save = os.path.join(save_dir, "id_test_dedup.json")
    with open(id_test_save, "w") as f:
        json.dump(id_test_dedup, f)
    logging.info(f"Saved deduplicated ID test results to {id_test_save}")
    
    ood_save = os.path.join(save_dir, "ood_dedup.json")
    with open(ood_save, "w") as f:
        json.dump(ood_dedup, f)
    logging.info(f"Saved deduplicated OOD test results to {ood_save}")
    
    # id_train_stats_path = os.path.join(save_dir, "dedup_stats_id_train.json")
    # with open(id_train_stats_path, "w") as f:
    #     json.dump(id_train_stats, f)
    
    # id_test_stats_path = os.path.join(save_dir, "dedup_stats_id_test.json")
    # with open(id_test_stats_path, "w") as f:
    #     json.dump(id_test_stats, f)
    
    # ood_stats_path = os.path.join(save_dir, "dedup_stats_ood.json")
    # with open(ood_stats_path, "w") as f:
    #     json.dump(ood_stats, f)
    
    logging.info("Finished all. Final deduplication stats:")
    logging.info(f"ID train: {len(id_train_dedup)} groups, ID test: {len(id_test_dedup)} groups, OOD: {len(ood_dedup)} groups")
    logging.info("Done.")

if __name__ == "__main__":
    main()