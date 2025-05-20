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
    """Configure logging level based on debug mode."""
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s - %(message)s')
    

def parse_tokens(text):
    """Parse tokens from text format <t1><t2><t3></a>."""
    return text.replace("</a>", "").strip("><").split("><")


def load_atomic_facts_3hop(f1_path, f2_path, f3_path):
    """
    For 3-hop logic, parse the atomic facts files:
      - Subcomponent 1: (t1, t2) -> b1
      - Subcomponent 2: (b1, t3) -> b2
      - Subcomponent 3: (b2, t4) -> t_final
    Returns three dictionaries: f1_dict, f2_dict, f3_dict.
    """
    f1_dict, f2_dict, f3_dict = {}, {}, {}

    def parse_atomic_facts(file_path):
        """Parse atomic facts from JSON file."""
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
    
    return parse_atomic_facts(f1_path), parse_atomic_facts(f2_path), parse_atomic_facts(f3_path)

###############################################################################
# 2) Helpers for 3-Hop Input & Grouping
###############################################################################

def group_data_by_b1(examples, f1_dict):
    """
    Returns a dictionary mapping b1 to a list of examples.
    """
    group_dict = defaultdict(list)
    for ex in examples:
        t1, t2, _, _ = parse_tokens(ex["input_text"])
        b1 = f1_dict.get((t1, t2))
        group_dict[b1].append(ex)
    return dict(group_dict)

def group_data_by_b2(examples, f1_dict, f2_dict):
    """
    Returns a dictionary mapping b2 to a list of examples.
    """
    group_dict = defaultdict(list)
    for ex in examples:
        t1, t2, t3, _ = parse_tokens(ex["input_text"])
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
        t1, t2, t3, t4 = parse_tokens(ex["input_text"])
        b1 = f1_dict.get((t1, t2))
        b2 = f2_dict.get((b1, t3))
        t_final = f3_dict.get((b2, t4))
        group_dict[t_final].append(ex)
    return dict(group_dict)

def group_data_by_b2_e3(examples, f1_dict, f2_dict):
    """
    Returns a dictionary mapping (b2, e3) pair to a list of examples.
    """
    group_dict = defaultdict(list)
    for ex in examples:
        t1, t2, t3, t4 = parse_tokens(ex["input_text"])
        b1 = f1_dict.get((t1, t2))
        b2 = f2_dict.get((b1, t3))
        group_dict[f"{t3},{b2}"].append(ex)
    return dict(group_dict)

def group_data_by_b1_b2(examples, f1_dict, f2_dict):
    """
    Returns a dictionary mapping (b1, b2) pair to a list of examples.
    """
    group_dict = defaultdict(list)
    for ex in examples:
        t1, t2, t3, _ = parse_tokens(ex["input_text"])
        b1 = f1_dict.get((t1, t2))
        b2 = f2_dict.get((b1, t3))
        group_dict[f"{b1},{b2}"].append(ex)
    return dict(group_dict)


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
            elif idx == 4:
                if d['type'] in ['type_1', 'type_2', 'type_5', 'type_6']:
                    ood_test_data.append(d)
            elif idx == 5:
                if d['type'] in ['type_1', 'type_2']:
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
    elif idx == 4:
        grouped_id_train_data = group_data_by_b2_e3(id_train_data, f1_dict, f2_dict)
        grouped_id_test_data = group_data_by_b2_e3(id_test_data, f1_dict, f2_dict)
        grouped_ood_test_data = group_data_by_b2_e3(ood_test_data, f1_dict, f2_dict)
    elif idx == 5:
        grouped_id_train_data = group_data_by_b1_b2(id_train_data, f1_dict, f2_dict)
        grouped_id_test_data = group_data_by_b1_b2(id_test_data, f1_dict, f2_dict)
        grouped_ood_test_data = group_data_by_b1_b2(ood_test_data, f1_dict, f2_dict)
    else:
        raise NotImplementedError
    
    return {
        'id_train': grouped_id_train_data,
        'id_test': grouped_id_test_data,
        'ood': grouped_ood_test_data
    }


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
                if layer == "logit":
                    hs = outputs.logits
                elif layer == "prob":
                    hs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                else:
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
        assert type(layer) == int
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
#     Improvement: Quantize each hidden state vector (by rounding to a fixed precision)
#     to a hashable tuple and use set membership for fast duplicate checks.
#     """
#     import numpy as np
#     from collections import defaultdict

#     dedup_stats = defaultdict(lambda: defaultdict(int))
#     deduplicated_results = defaultdict(list)

#     def vectors_equal(v1, v2):
#         """Compare two vectors for equality with numerical tolerance."""
#         if v1 is None or v2 is None:
#             return v1 is None and v2 is None
#         return np.allclose(np.array(v1), np.array(v2), rtol=1e-5, atol=1e-8)

#     def get_vector_key(hidden_state, precision=7):
#         """
#         Create a combined hashable key for a hidden state by quantizing available vectors
#         (e.g., embedding, post_attention, post_mlp) and combining them into one tuple.
#         """
#         keys = []
#         if 'embedding' in hidden_state:
#             keys.append(quantize_vector(hidden_state['embedding'], precision))
#         if 'post_attention' in hidden_state and hidden_state['post_attention'] is not None:
#             keys.append(quantize_vector(hidden_state['post_attention'], precision))
#         if 'post_mlp' in hidden_state and hidden_state['post_mlp'] is not None:
#             keys.append(quantize_vector(hidden_state['post_mlp'], precision))
#         return tuple(keys)
    
#     for target, instances in results.items():
#         seen_vectors = defaultdict(set)  # (layer, pos) -> set of vector keys
#         logging.info(f"Performing deduplication for target {target}")
#         if target == 'unknown':
#             continue
        
#         for instance in instances:
#             is_duplicate = False
#             for hidden_state in instance['hidden_states']:
#                 layer = hidden_state['layer']
#                 pos = hidden_state['position']
#                 vector_key = get_vector_key(hidden_state)
#                 if vector_key in seen_vectors[(layer, pos)]:
#                     dedup_stats[target][f"layer{layer}_pos{pos}"] += 1
#                     is_duplicate = True
#                     break
#                 else:
#                     seen_vectors[(layer, pos)].add(vector_key)
#             if not is_duplicate:
#                 deduplicated_results[target].append(instance)
    
#     final_stats = {k: dict(v) for k, v in dedup_stats.items()}
#     return dict(deduplicated_results), final_stats


def deduplicate_grouped_data(grouped_data, atomic_idx):
    """
    grouped_data: 그룹핑된 데이터. 형식은 { group_key: [entry, entry, ...] }이며,
                  각 entry는 "input_text"와 "target_text"를 포함하는 dict입니다.
    atomic_idx: deduplication 기준을 결정하는 인덱스
                - 1이면, target_text의 처음 두 토큰(t1, t2) 기준 deduplication
                - 2이면, 처음 세 토큰(t1, t2, t3) 기준 deduplication
                - 3이면, 처음 네 토큰(t1, t2, t3, t4) 기준 deduplication

    Returns:
        중복 제거된 entry들의 리스트. 동일한 deduplication 키를 가진 entry들은 하나만 남게 됩니다.
    """
    output = {}
    for group_key, entries in grouped_data.items():
        deduped = {}
        for entry in entries:
            tokens = parse_tokens(entry["target_text"])
            if atomic_idx == 1:
                dedup_key = tuple(tokens[:2])  # (t1, t2)
            elif atomic_idx == 2 or atomic_idx == 4 or atomic_idx == 5:
                dedup_key = tuple(tokens[:3])  # (t1, t2, t3)
            elif atomic_idx == 3:
                dedup_key = tuple(tokens[:4])  # (t1, t2, t3, t4)
            else:
                raise ValueError("atomic_idx must be 1, 2, or 3")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to the model checkpoint")
    parser.add_argument("--layer_pos_pairs", required=True, help="List of (layer, position) tuples to evaluate")
    parser.add_argument("--data_dir", default=None, help="directory for dataset")
    parser.add_argument("--save_dir", required=True, help="Directory to save the analysis results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for verbose output")
    parser.add_argument("--atomic_idx", required=True, type=int, choices=[1,2,3], help="Bottleneck function index among f1, f2, and f3 used for collapse evaluation")
    parser.add_argument("--mode", required=True, choices=["post_mlp", "residual"], help="Mode: 'post_mlp' or 'residual'")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size for processing")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing save directory if it exists")
    args = parser.parse_args()

    setup_logging(args.debug)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
    
    data_dir = os.path.join(data_dir, "data", dataset)

    # (t_N1, t_N2) -> t_N3
    f1_dict, f2_dict, f3_dict = load_atomic_facts_3hop(
        os.path.join(data_dir, f"atomic_facts_f1.json"),
        os.path.join(data_dir, f"atomic_facts_f2.json"),
        os.path.join(data_dir, f"atomic_facts_f3.json")
    )

    grouped_data = load_and_preprocess_data(
        f1_dict, f2_dict, f3_dict, os.path.join(data_dir, "test.json"), idx=args.atomic_idx
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