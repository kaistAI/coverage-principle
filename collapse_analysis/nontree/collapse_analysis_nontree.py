"""
This script analyzes the collapse behavior of a transformer model in nontree DAG task.
It processes test data, extracts hidden states, and analyzes model behavior across different data groups.
"""

import argparse
import torch
import json
from tqdm import tqdm
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging
from collections import defaultdict
import re


def setup_logging(debug_mode):
    """Set up logging with DEBUG level if debug_mode is True, otherwise INFO level."""
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s - %(message)s')
    

def parse_tokens(text):
    """Parse tokens from text format <t1><t2><t3></a>."""
    return text.replace("</a>", "").strip("><").split("><")


def load_atomic_facts_nontree(f1_path, f2_path):
    """
    For the non-tree DAG problem, we parse:
      f1: (x1, x2) -> b1
      f2: (b, x2, x3) -> y
    Return two dicts: f1_dict, f2_dict
    """
    def parse_atomic_facts_f1(file_path):
        """
        Format for f1:
          "input_text": "<t_N1><t_N2>"
          "target_text": "<t_N1><t_N2><t_N3></a>"
        This corresponds to the mapping (t_N1, t_N2) -> t_N3
        """
        with open(file_path, "r") as f:
            facts = json.load(f)
        out_dict = {}
        for item in facts:
            inp_tokens = parse_tokens(item["input_text"])
            assert len(inp_tokens) == 2
            tgt_tokens = parse_tokens(item["target_text"])
            assert len(tgt_tokens) == 3 and inp_tokens == tgt_tokens[:2]
            out_dict[(tgt_tokens[0], tgt_tokens[1])] = tgt_tokens[-1]
        return out_dict

    def parse_atomic_facts_f2(file_path):
        """
        Format for f2:
          "input_text": "<b_N1><t_N2><t_N3>",
          "target_text": "<b_N1><t_N2><t_N3><y></a>"
        => (b_1, t_1, t_2) -> y
        """
        with open(file_path, "r") as f:
            facts = json.load(f)
        out_dict = {}
        for item in facts:
            inp_tokens = parse_tokens(item["input_text"])
            assert len(inp_tokens) == 3
            tgt_tokens = parse_tokens(item["target_text"])
            assert len(tgt_tokens) == 4 and inp_tokens == tgt_tokens[:3]
            out_dict[(tgt_tokens[0], tgt_tokens[1], tgt_tokens[2])] = tgt_tokens[-1]
        return out_dict

    return parse_atomic_facts_f1(f1_path), parse_atomic_facts_f2(f2_path)


###############################################################################
# 3) Helpers for parsing 3-token inputs and grouping by b1 or y
###############################################################################

def group_data_by_b(examples, f1_dict):
    """Group examples by their bridge entity (b)."""
    group_dict = defaultdict(list)
    for ex in examples:
        t1, t2, _ = parse_tokens(ex["input_text"])
        b1 = f1_dict.get((t1, t2))
        group_dict[b1].append(ex)
    return dict(group_dict)

def group_data_by_t_final(examples, f1_dict, f2_dict):
    """Group examples by their final target (t_final)."""
    group_dict = defaultdict(list)
    for ex in examples:
        t1, t2, t3 = parse_tokens(ex["input_text"])
        b1 = f1_dict.get((t1, t2))
        t_final = f2_dict.get((b1, t2, t3))
        group_dict[t_final].append(ex)
    return dict(group_dict)

def group_data_by_t2(examples):
    """Group examples by their second token (t2)."""
    group_dict = defaultdict(list)
    for ex in examples:
        _, t2, _ = parse_tokens(ex["input_text"])
        group_dict[t2].append(ex)
    return dict(group_dict)

def group_data_by_b_t2(examples, f1_dict, f2_dict):
    """Group examples by their bridge entity (b) and second token (t2)."""
    group_dict = defaultdict(list)
    for ex in examples:
        t1, t2, _ = parse_tokens(ex["input_text"])
        b1 = f1_dict.get((t1, t2))
        group_dict[f"{b1},{t2}"].append(ex)
    return dict(group_dict)

def group_data_by_b_t1(examples, f1_dict, f2_dict):
    """Group examples by their bridge entity (b) and first token (t1)."""
    group_dict = defaultdict(list)
    for ex in examples:
        t1, t2, _ = parse_tokens(ex["input_text"])
        b1 = f1_dict.get((t1, t2))
        group_dict[f"{b1},{t1}"].append(ex)
    return dict(group_dict)


###############################################################################
# 4) Splitting data into ID train, ID test, OOD test
###############################################################################
def load_and_preprocess_data(f1_dict, f2_dict, test_path, idx):
    """
    Load and preprocess test data, grouping by different criteria.
    
    Args:
        f1_dict, f2_dict: Atomic facts dictionaries
        test_path: Path to test data
        idx: Atomic function index (1, 2, 3, 4 or 5)
    """
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    id_train_data = []
    id_test_data = []
    ood_test_data = []
    
    # Categorize data
    for d in test_data:
        if d['type'] == 'train_inferred':
            id_train_data.append(d)
        elif d['type'] == 'type_0':
            id_test_data.append(d)
        elif d['type'] in set([f"type_{i}" for i in range(1, 4)]):
            if idx == 1:
                if d['type'] in ['type_1', 'type_2']:
                    ood_test_data.append(d)
            elif idx == 2:
                if d['type'] in ['type_1', 'type_3']:
                    ood_test_data.append(d)
            elif idx == 3:  # kind of weird...
                ood_test_data.append(d)
            elif idx == 4:
                if d['type'] in ['type_1', 'type_2']:
                    ood_test_data.append(d)
            elif idx == 5:
                if d['type'] in ['type_1', 'type_2']:
                    ood_test_data.append(d)
            else:
                raise NotImplementedError(f"Invalid idx value: {idx}")
        else:
            raise NotImplementedError("Invalid coverage type")
            
    if idx == 1:
        grouped_id_train = group_data_by_b(id_train_data, f1_dict)
        grouped_id_test = group_data_by_b(id_test_data, f1_dict)
        grouped_ood = group_data_by_b(ood_test_data, f1_dict)
    elif idx == 2:
        grouped_id_train = group_data_by_t_final(id_train_data, f1_dict, f2_dict)
        grouped_id_test = group_data_by_t_final(id_test_data, f1_dict, f2_dict)
        grouped_ood = group_data_by_t_final(ood_test_data, f1_dict, f2_dict)
    elif idx == 3:
        grouped_id_train = group_data_by_t2(id_train_data)
        grouped_id_test = group_data_by_t2(id_test_data)
        grouped_ood = group_data_by_t2(ood_test_data)
    elif idx == 4:
        grouped_id_train = group_data_by_b_t2(id_train_data, f1_dict, f2_dict)
        grouped_id_test = group_data_by_b_t2(id_test_data, f1_dict, f2_dict)
        grouped_ood = group_data_by_b_t2(ood_test_data, f1_dict, f2_dict)
    elif idx == 5:
        grouped_id_train = group_data_by_b_t1(id_train_data, f1_dict, f2_dict)
        grouped_id_test = group_data_by_b_t1(id_test_data, f1_dict, f2_dict)
        grouped_ood = group_data_by_b_t1(ood_test_data, f1_dict, f2_dict)
    else:
        raise NotImplementedError("atomic_idx must be 1 (group by b), 2 (group by y), 3 (group by t2), 4 (group by b+t2), or 5 (group by b+t1)")

    return {
        'id_train': grouped_id_train,
        'id_test': grouped_id_test,
        'ood': grouped_ood
    }

###############################################################################
# Extracting hidden states
# (Same approach: 'residual' uses model outputs; 'post_mlp' uses hooks)
###############################################################################
def get_hidden_states_residual(model, input_texts, layer_pos_pairs, tokenizer, device):
    """
    Extract hidden states from residual stream for a batch of inputs.
    
    Args:
        model: GPT2 model
        input_texts: List of input texts
        layer_pos_pairs: List of (layer, position) tuples to extract
        tokenizer: GPT2 tokenizer
        device: Device to run model on
    
    Returns:
        List of hidden states for each input
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
                if layer == "logit":
                    hs = outputs.logits
                elif layer == "prob":
                    hs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                else:
                    hs = all_hidden_states[layer]
                
                if hs.dim() != 3 and hs.dim() != 2:
                    raise ValueError(f"Invalid hidden state dimension: {hs.dim()}")
                token_vec = hs[i, pos, :].detach().cpu().numpy() if hs.dim() == 3 else hs[pos, :].detach().cpu().numpy()
                instance_hidden_states.append({
                    'layer': layer,
                    'position': pos,
                    'post_attention': None,
                    'post_mlp': token_vec.tolist()
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
    Extract hidden states from MLP layers using hooks.
    
    Args:
        model: GPT2 model
        input_texts: List of input texts
        layer_pos_pairs: List of (layer, position) tuples to extract
        tokenizer: GPT2 tokenizer
        device: Device to run model on
    
    Returns:
        List of hidden states for each input
    """
    inputs = tokenizer(input_texts, padding=True, return_tensors="pt").to(device)
    activation = {}
    
    def get_activation(name):
        def hook(module, inp, out):
            activation[name] = out
        return hook
    
    # Register hooks for attention and MLP layers
    hooks = []
    for layer, pos in layer_pos_pairs:
        if isinstance(layer, int) and layer > 0:
            hooks.append(model.transformer.h[layer-1].attn.register_forward_hook(get_activation(f'layer{layer}_attn')))
            hooks.append(model.transformer.h[layer-1].mlp.register_forward_hook(get_activation(f'layer{layer}_mlp')))
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    for h in hooks:
        h.remove()

    hidden_states = []
    for layer, pos in layer_pos_pairs:
        try:
            if layer == 0:
                word_embeddings = model.transformer.wte(inputs['input_ids'])
                vec = word_embeddings[0,pos,:].detach().cpu().numpy()
                hidden_states.append({
                    'layer': layer,
                    'position': pos,
                    'embedding': vec.tolist()
                })
            else:
                post_attn = activation.get(f'layer{layer}_attn', None)
                post_mlp  = activation.get(f'layer{layer}_mlp', None)
                if isinstance(post_attn, tuple): post_attn= post_attn[0]
                if isinstance(post_mlp, tuple):  post_mlp=  post_mlp[0]

                post_attn = post_attn[0,pos,:].detach().cpu().numpy() if post_attn is not None else None
                post_mlp  = post_mlp[0,pos,:].detach().cpu().numpy() if post_mlp is not None else None

                hidden_states.append({
                    'layer': layer,
                    'position': pos,
                    'post_attention': post_attn.tolist() if post_attn is not None else None,
                    'post_mlp': post_mlp.tolist() if post_mlp is not None else None
                })
        except Exception as e:
            logging.error(f"Error @ layer {layer}, pos {pos}: {str(e)}")
            hidden_states.append({
                'layer': layer,
                'position': pos,
                'error': str(e)
            })
    return hidden_states

def process_data_group(model, data_group, layer_pos_pairs, tokenizer, device, mode, batch_size=8):
    """
    Process a group of data instances in batches.
    
    Args:
        model: GPT2 model
        data_group: Dictionary of data instances grouped by target
        layer_pos_pairs: List of (layer, position) tuples to extract
        tokenizer: GPT2 tokenizer
        device: Device to run model on
        mode: 'post_mlp' or 'residual'
        batch_size: Batch size for processing
    
    Returns:
        Dictionary of processed results grouped by target
    """
    results = defaultdict(list)
    for bridge_entity, instances in tqdm(data_group.items(), desc="Processing instances"):
        for i in range(0, len(instances), batch_size):
            batch = instances[i:i+batch_size]
            input_texts = [ex['target_text'] for ex in batch]
            
            # Get hidden states based on mode
            batch_hidden_states = (get_hidden_states_mlp if mode == "mlp" else get_hidden_states_residual)(
                model, input_texts, layer_pos_pairs, tokenizer, device
            )
            
            # Store results
            for ex, hs in zip(batch, batch_hidden_states):
                results[bridge_entity].append({
                    "input_text": ex['input_text'],
                    "target_text": ex['target_text'],
                    "identified_target": bridge_entity,
                    "type": ex.get('type'),
                    "hidden_states": hs
                })
    return results


# def deduplicate_vectors(results):
#     """
#     Deduplicate vectors within each target group and track removal statistics.
    
#     Args:
#         results (dict): Dictionary containing results grouped by targets
        
#     Returns:
#         tuple: (deduplicated_results, dedup_stats)
#     """
#     import numpy as np
#     from collections import defaultdict
    
#     dedup_stats = defaultdict(lambda: defaultdict(int))
#     deduplicated_results = defaultdict(list)
    
#     def vectors_equal(v1, v2):
#         """Check close equality of two arrays."""
#         if v1 is None or v2 is None:
#             return (v1 is None) and (v2 is None)
#         return np.allclose(np.array(v1), np.array(v2), rtol=1e-5, atol=1e-8)
    
#     def get_vector_key(hidden_state):
#         """Combine the relevant vectors into a single tuple key."""
#         vectors = []
#         if 'embedding' in hidden_state:
#             vectors.append(tuple(hidden_state['embedding']))
#         if 'post_attention' in hidden_state and hidden_state['post_attention'] is not None:
#             vectors.append(tuple(hidden_state['post_attention']))
#         if 'post_mlp' in hidden_state and hidden_state['post_mlp'] is not None:
#             vectors.append(tuple(hidden_state['post_mlp']))
#         return tuple(vectors)
    
#     for target, instances in results.items():
#         seen_vectors = defaultdict(set)
#         for instance in instances:
#             is_duplicate = False
#             for hidden_state in instance['hidden_states']:
#                 layer = hidden_state['layer']
#                 pos = hidden_state['position']
#                 vector_key = get_vector_key(hidden_state)
                
#                 # check if we've seen this vector key
#                 is_vec_dup = False
#                 for seen_key in seen_vectors[(layer, pos)]:
#                     if all(vectors_equal(a, b) for a, b in zip(vector_key, seen_key)):
#                         is_vec_dup = True
#                         dedup_stats[target][f"layer{layer}_pos{pos}"] += 1
#                         break
#                 if is_vec_dup:
#                     is_duplicate = True
#                     break
#                 else:
#                     seen_vectors[(layer, pos)].add(vector_key)

#             if not is_duplicate:
#                 deduplicated_results[target].append(instance)
#     final_stats = {k: dict(v) for k,v in dedup_stats.items()}
#     return dict(deduplicated_results), final_stats

def deduplicate_grouped_data(grouped_data, atomic_idx):
    """
    Remove duplicate entries from grouped data based on atomic function index.
    
    Args:
        grouped_data: Dictionary of grouped data entries
        atomic_idx: Index determining deduplication criteria (1, 2, 3, 4 or 5)
    
    Returns:
        Dictionary with deduplicated entries
    """
    output = {}
    for group_key, entries in grouped_data.items():
        deduped = {}
        for entry in entries:
            tokens = parse_tokens(entry["target_text"])
            if atomic_idx == 1:
                dedup_key = tuple(tokens[:2])  # (t1, t2)
            elif atomic_idx == 2:
                dedup_key = tuple(tokens[:3])  # (t1, t2, t3)
            elif atomic_idx == 3:
                dedup_key = tuple(tokens[:3])  # (t1, t2, t3)
            elif atomic_idx == 4:
                dedup_key = tuple(tokens[:2])  # (t1, t2)
            elif atomic_idx == 5:
                dedup_key = tuple(tokens[:2])  # (t1, t2)
            else:
                raise ValueError("atomic_idx must be 1, 2, 3, 4 or 5")

            if dedup_key not in deduped:
                deduped[dedup_key] = entry
        output[group_key] = list(deduped.values())
    return output


def parse_layer_pos_pairs(layer_pos_str):
    """
    Parse layer position pairs from string format.
    Handles both tuple format like "(0,0),(1,1)" and special formats like "(logit,0)" or "(prob,0)".
    
    Args:
        layer_pos_str: String containing layer position pairs
        
    Returns:
        List of (layer, position) tuples
    """
    if "logit" in layer_pos_str or "prob" in layer_pos_str:
        # Handle special formats like "(logit,0)" or "(prob,0)"
        pos_match = re.search(r"\((logit|prob|\d+),(\d+)\)", layer_pos_str)
        if pos_match:
            layer_type = pos_match.group(1)
            pos = int(pos_match.group(2))
            return [(layer_type, pos)]
        else:
            raise ValueError(f"Invalid layer_pos_pairs format: {layer_pos_str}")
    else:
        # Handle tuple format like "(0,0),(1,1)"
        try:
            return eval(layer_pos_str)
        except:
            raise ValueError(f"Invalid layer_pos_pairs format: {layer_pos_str}")


def main():
    """Main function to run the collapse analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to the model checkpoint")
    parser.add_argument("--layer_pos_pairs", required=True, help="List of (layer, position) tuples to evaluate")
    parser.add_argument("--data_dir", default=None, help="directory for dataset")
    parser.add_argument("--save_dir", required=True, help="Directory to save the analysis results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for verbose output")
    parser.add_argument("--atomic_idx", required=True, type=int, choices=[1,2,3,4,5], help="which function's bridging entity to group by: 1 => b1, 2 => final y, 3 => t2, 4 => b1 + t2, 5 => b1 + t1")
    parser.add_argument("--mode", required=True, choices=["post_mlp","residual"], help="whether to save the hidden representation of post_mlp or the residual stream")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size for processing")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing save directory if it exists")
    
    args = parser.parse_args()

    setup_logging(args.debug)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if args.data_dir:
        assert os.path.isdir(args.data_dir)
        data_dir = args.data_dir
    else:
        data_dir = base_dir

    logging.debug(f"base_dir: {base_dir}")
    logging.debug(f"data_dir: {data_dir}")

    if args.ckpt.split("/")[-1] == "":
        dataset, step = args.ckpt.split("/")[-3].split("_")[0], "final_checkpoint" if args.ckpt.split("/")[-2] == "final_checkpoint" else args.ckpt.split("/")[-2].split("-")[-1]
    else:
        dataset, step = args.ckpt.split("/")[-2].split("_")[0], "final_checkpoint" if args.ckpt.split("/")[-1] == "final_checkpoint" else args.ckpt.split("/")[-1].split("-")[-1]
    
    logging.debug(f"dataset: {dataset}\nstep: {step}")
    data_dir = os.path.join(data_dir, "data", dataset)
    
    logging.info("Loading model and tokenizer...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(args.ckpt).to(device)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(args.ckpt)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    logging.info("Model and tokenizer loaded successfully")
    
    # Load atomic facts and process data
    f1_dict, f2_dict = load_atomic_facts_nontree(
        os.path.join(data_dir, "atomic_facts_f1.json"),
        os.path.join(data_dir, "atomic_facts_f2.json")
    )
    

    grouped_data = load_and_preprocess_data(
        f1_dict, f2_dict, 
        os.path.join(data_dir, "test.json"),
        idx=args.atomic_idx
    )
    
    # Process data and save results
    results = {}
    layer_pos_pairs = parse_layer_pos_pairs(args.layer_pos_pairs)
    logging.info(f"Layer position pairs: {layer_pos_pairs}")
    
    torch.manual_seed(0)
    for data_type, data_group in grouped_data.items():
        results[data_type] = process_data_group(
            model, data_group, layer_pos_pairs, tokenizer, device, args.mode, args.batch_size
        )
    
    # Deduplicate and save results
    save_dir = os.path.join(args.save_dir, args.mode, f"{dataset}",
                           f"f{args.atomic_idx}", str(layer_pos_pairs[0]).replace("'", "").replace(" ", ""),
                           step)
    
    os.makedirs(save_dir, exist_ok=True)
    
    for data_type, data_results in results.items():
        dedup_results = deduplicate_grouped_data(data_results, args.atomic_idx)
        with open(os.path.join(save_dir, f"{data_type}_dedup.json"), "w") as f:
            json.dump(dedup_results, f)

if __name__ == "__main__":
    main()