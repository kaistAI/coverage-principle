import argparse
import torch
import json
from tqdm import tqdm
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging
from collections import defaultdict
import re

###############################################################################
# 1) Logging setup
###############################################################################
def setup_logging(debug_mode):
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s - %(message)s')

###############################################################################
# 2) Loading atomic facts for the "non-tree DAG": f1(t0, t1)->b1; f2(b1, t1, t2)->y
###############################################################################
def load_atomic_facts_nontree(f1_path, f2_path):
    """
    For the non-tree DAG problem, we parse:
      f1: (t0, t1) -> b1
      f2: (b1, t1, t2) -> y
    Return two dicts: f1_dict, f2_dict
    """

    def parse_atomic_facts_f1(file_path):
        """
        Format for f1:
          "input_text": "<t_0><t_1>",
          "target_text": "<t_0><t_1><b_1></a>"
        => (t_0, t_1) -> b_1
        """
        with open(file_path, "r") as f:
            facts = json.load(f)
        out_dict = {}
        for item in facts:
            inp = item["input_text"].strip("<>").split("><")
            if len(inp) != 2:
                continue
            t0, t1 = inp
            tgt = item["target_text"].replace("</a>", "").strip("<>").split("><")
            # Expect 3 tokens in the target: t0, t1, b1
            if len(tgt) == 3:
                b1 = tgt[-1]
                out_dict[(t0, t1)] = b1
        return out_dict

    def parse_atomic_facts_f2(file_path):
        """
        Format for f2:
          "input_text": "<b_1><t_1><t_2>",
          "target_text": "<b_1><t_1><t_2><y></a>"
        => (b_1, t_1, t_2) -> y
        """
        with open(file_path, "r") as f:
            facts = json.load(f)
        out_dict = {}
        for item in facts:
            inp = item["input_text"].strip("<>").split("><")
            if len(inp) != 3:
                continue
            b1, t1, t2 = inp
            tgt = item["target_text"].replace("</a>", "").strip("<>").split("><")
            # Expect 4 tokens in the target: b1, t1, t2, y
            if len(tgt) == 4:
                y = tgt[-1]
                out_dict[(b1, t1, t2)] = y
        return out_dict

    return parse_atomic_facts_f1(f1_path), parse_atomic_facts_f2(f2_path)


###############################################################################
# 3) Helpers for parsing 3-token inputs and grouping by b1 or y
###############################################################################
def parse_3token_input(input_text):
    """
    e.g. "<t_7><t_11><t_99>" => ["t_7","t_11","t_99"]
    returns None if not 3 tokens
    """
    tokens = input_text.strip("<>").split("><")
    if len(tokens) != 3:
        return None
    return tokens

def group_data_by_b1(examples, f1_dict):
    """
    For each example: parse (t0,t1,t2),
      b1 = f1_dict.get((t0,t1),'unknown')
    group by b1 => group_dict[b1] = list of examples
    """
    group_dict = defaultdict(list)
    for ex in examples:
        inp_tokens = parse_3token_input(ex["input_text"])
        if not inp_tokens:
            continue
        t0, t1, _ = inp_tokens
        b1 = f1_dict.get((t0, t1), "unknown")
        if b1 == "unknown":
            continue
        group_dict[b1].append(ex)
    return dict(group_dict)

def group_data_by_y(examples, f1_dict, f2_dict):
    """
    For each example: parse (t0,t1,t2),
    b1 = f1_dict.get((t0,t1),'unknown')
    y  = f2_dict.get((b1,t1,t2),'unknown') if b1 != 'unknown'
    group by y => group_dict[y] = list of examples
    """
    group_dict = defaultdict(list)
    for ex in examples:
        inp_tokens = parse_3token_input(ex["input_text"])
        if not inp_tokens:
            continue
        t0, t1, t2 = inp_tokens
        b1 = f1_dict.get((t0, t1), "unknown")
        if b1 == "unknown":
            continue
        y = f2_dict.get((b1, t1, t2), "unknown")
        if y == "unknown":
            continue
        group_dict[y].append(ex)
    return dict(group_dict)


###############################################################################
# 4) Splitting data into ID train, ID test, OOD test
###############################################################################
def load_and_preprocess_data(f1_dict, f2_dict, test_path, idx):
    """
    - read test.json => separate examples by coverage 'type'
    - 'train_inferred' => in-domain train
    - 'type_0' => in-domain test
    - everything else => OOD test
    - group by b1 if idx=1, or by y if idx=2
    """
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    id_train_data = []
    id_test_data = []
    ood_test_data = []

    for d in test_data:
        typ = d['type']
        if typ == 'train_inferred':
            id_train_data.append(d)
        elif typ == 'type_0':
            id_test_data.append(d)
        else:
            # assume type_1..3 => OOD
            ood_test_data.append(d)

    if idx == 1:
        grouped_id_train_data = group_data_by_b1(id_train_data, f1_dict)
        grouped_id_test_data = group_data_by_b1(id_test_data, f1_dict)
        grouped_ood_test_data = group_data_by_b1(ood_test_data, f1_dict)
    elif idx == 2:
        grouped_id_train_data = group_data_by_y(id_train_data, f1_dict, f2_dict)
        grouped_id_test_data = group_data_by_y(id_test_data, f1_dict, f2_dict)
        grouped_ood_test_data = group_data_by_y(ood_test_data, f1_dict, f2_dict)
    else:
        raise NotImplementedError("atomic_idx must be 1 (group by b1) or 2 (group by y).")

    return grouped_id_train_data, grouped_id_test_data, grouped_ood_test_data


###############################################################################
# 5) Extracting hidden states
#    (Same approach: 'residual' uses model outputs; 'post_mlp' uses hooks)
###############################################################################
def get_hidden_states_residual(model, input_text, layer_pos_pairs, tokenizer, device):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    all_hidden_states = outputs["hidden_states"]
    
    hidden_states = []
    for layer, pos in layer_pos_pairs:
        try:
            post_block = all_hidden_states[layer]
            # post_block shape: (batch, seq_len, hidden_dim)
            post_block = post_block[0, pos, :].detach().cpu().numpy()
            hidden_states.append({
                'layer': layer,
                'position': pos,
                'post_attention': None,
                'post_mlp': post_block.tolist() if post_block is not None else None
            })
        except Exception as e:
            logging.error(f"Error processing layer {layer}, position {pos}: {str(e)}")
            hidden_states.append({
                'layer': layer,
                'position': pos,
                'error': str(e)
            })
    return hidden_states

def get_hidden_states_mlp(model, input_text, layer_pos_pairs, tokenizer, device):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    activation = {}
    def get_activation(name):
        def hook(model, inp, out):
            activation[name] = out
        return hook
    
    # register hooks for each layer & position
    hooks = []
    for layer, pos in layer_pos_pairs:
        if layer > 0:
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

def process_data_group(model, data_group, layer_pos_pairs, tokenizer, device, mode):
    results = defaultdict(list)
    for bridge_entity, instances in tqdm(data_group.items(), desc="Processing instances"):
        for instance in instances:
            inp_txt = instance['input_text']
            if mode == "residual":
                hs = get_hidden_states_residual(model, inp_txt, layer_pos_pairs, tokenizer, device)
            else:
                hs = get_hidden_states_mlp(model, inp_txt, layer_pos_pairs, tokenizer, device)
            item={
                "input_text": inp_txt,
                "target_text": instance['target_text'],
                "identified_target": bridge_entity,
                "type": instance.get('type'),
                "hidden_states": hs
            }
            results[bridge_entity].append(item)
    return results

###############################################################################
# 6) Deduplication
###############################################################################
def deduplicate_vectors(results):
    import numpy as np
    from collections import defaultdict
    
    dedup_stats = defaultdict(lambda: defaultdict(int))
    deduplicated_results = defaultdict(list)
    
    def vectors_equal(v1, v2):
        """Check close equality of two arrays."""
        if v1 is None or v2 is None:
            return (v1 is None) and (v2 is None)
        return np.allclose(np.array(v1), np.array(v2), rtol=1e-5, atol=1e-8)
    
    def get_vector_key(hidden_state):
        """Combine the relevant vectors into a single tuple key."""
        vectors = []
        if 'embedding' in hidden_state:
            vectors.append(tuple(hidden_state['embedding']))
        if 'post_attention' in hidden_state and hidden_state['post_attention'] is not None:
            vectors.append(tuple(hidden_state['post_attention']))
        if 'post_mlp' in hidden_state and hidden_state['post_mlp'] is not None:
            vectors.append(tuple(hidden_state['post_mlp']))
        return tuple(vectors)
    
    for target, instances in results.items():
        seen_vectors = defaultdict(set)
        for instance in instances:
            is_duplicate = False
            for hidden_state in instance['hidden_states']:
                layer = hidden_state['layer']
                pos = hidden_state['position']
                vector_key = get_vector_key(hidden_state)
                
                # check if we've seen this vector key
                is_vec_dup = False
                for seen_key in seen_vectors[(layer, pos)]:
                    if all(vectors_equal(a, b) for a, b in zip(vector_key, seen_key)):
                        is_vec_dup = True
                        dedup_stats[target][f"layer{layer}_pos{pos}"] += 1
                        break
                if is_vec_dup:
                    is_duplicate = True
                    break
                else:
                    seen_vectors[(layer, pos)].add(vector_key)

            if not is_duplicate:
                deduplicated_results[target].append(instance)
    final_stats = {k: dict(v) for k,v in dedup_stats.items()}
    return dict(deduplicated_results), final_stats

###############################################################################
# 7) Main
###############################################################################
def main():
    parser= argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="model checkpoint path")
    parser.add_argument("--layer_pos_pairs", required=True, help="(layer, position) tuples, e.g. '[(0,0),(1,0)]'")
    parser.add_argument("--save_dir", required=True, help="dir to store analysis results")
    parser.add_argument("--base_dir", required=True, help="base dir")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--atomic_idx", required=True, type=int,
                        help="which function's bridging entity to group by: 1 => b1, 2 => final y")
    parser.add_argument("--mode", required=True, choices=["post_mlp","residual"],
                        help="whether to save the hidden representation of post_mlp or the residual stream")

    args = parser.parse_args()
    setup_logging(args.debug)

    base_dir = args.base_dir
    logging.info(f"base_dir: {base_dir}")

    # for demonstration, parse dataset & step from path:
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

    data_dir = os.path.join(base_dir, "data", dataset)
    logging.info(f"data_dir: {data_dir}")

    # Load atomic facts for f1, f2
    atomic_file_1 = os.path.join(data_dir, "atomic_facts_f1.json")
    atomic_file_2 = os.path.join(data_dir, "atomic_facts_f2.json")
    f1_dict, f2_dict = load_atomic_facts_nontree(atomic_file_1, atomic_file_2)

    # We only have train.json or test.json for final data
    test_path = os.path.join(data_dir, "test.json")
    grouped_id_train_data, grouped_id_test_data, grouped_ood_test_data = load_and_preprocess_data(
        f1_dict, f2_dict, test_path, idx=args.atomic_idx
    )
    
    layer_pos_pairs = eval(args.layer_pos_pairs)
    logging.info(f"Layer position pairs: {layer_pos_pairs}")
    logging.info(f"ID train groups: {len(grouped_id_train_data)}")
    logging.info(f"ID test groups: {len(grouped_id_test_data)}")
    logging.info(f"OOD test groups: {len(grouped_ood_test_data)}")

    save_dir = os.path.join(args.save_dir, args.mode, dataset, f"f{args.atomic_idx}", str(layer_pos_pairs[0]).replace(" ", ""), step)
    if os.path.exists(save_dir):
        logging.info(f"{save_dir} already exists!")
        return
    else:
        os.makedirs(save_dir, exist_ok=True)

    logging.info("Process ID train group...")
    id_train_results = process_data_group(model, grouped_id_train_data, layer_pos_pairs, tokenizer, device, args.mode)

    logging.info("Process ID test group...")
    id_test_results = process_data_group(model, grouped_id_test_data, layer_pos_pairs, tokenizer, device, args.mode)

    logging.info("Process OOD test group...")
    ood_test_results = process_data_group(model, grouped_ood_test_data, layer_pos_pairs, tokenizer, device, args.mode)

    logging.info("Deduplicate ID train...")
    id_train_dedup, id_train_stats = deduplicate_vectors(id_train_results)

    logging.info("Deduplicate ID test...")
    id_test_dedup, id_test_stats = deduplicate_vectors(id_test_results)

    logging.info("Deduplicate OOD test...")
    ood_dedup, ood_stats = deduplicate_vectors(ood_test_results)

    # Save deduplicated results
    with open(os.path.join(save_dir, "id_train_dedup.json"), "w") as f:
        json.dump(id_train_dedup, f)
    with open(os.path.join(save_dir, "id_test_dedup.json"), "w") as f:
        json.dump(id_test_dedup, f)
    with open(os.path.join(save_dir, "ood_dedup.json"), "w") as f:
        json.dump(ood_dedup, f)

    # Save stats
    with open(os.path.join(save_dir, "dedup_stats_id_train.json"), "w") as f:
        json.dump(id_train_stats, f)
    with open(os.path.join(save_dir, "dedup_stats_id_test.json"), "w") as f:
        json.dump(id_test_stats, f)
    with open(os.path.join(save_dir, "dedup_stats_ood.json"), "w") as f:
        json.dump(ood_stats, f)

    logging.info("Finished. Final dedup =>")
    logging.info(f"ID train => {len(id_train_dedup)} groups, ID test => {len(id_test_dedup)} groups, OOD => {len(ood_dedup)}")
    logging.info("Done.")


if __name__=="__main__":
    main()
