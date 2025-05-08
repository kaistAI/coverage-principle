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
    tokens = text.replace("</a>", "").strip("><").split("><")
    return tokens

###############################################################################
# 2) Loading atomic facts for the "non-tree DAG": f1(t0, t1)->b1; f2(b1, t1, t2)->y
###############################################################################
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

def group_data_by_b1(examples, f1_dict):
    """
    For each example: parse (t0,t1,t2),
      b1 = f1_dict.get((t1, t2))
    group by b1 => group_dict[b1] = list of examples
    """
    group_dict = defaultdict(list)
    for ex in examples:
        inp_tokens = parse_tokens(ex["input_text"])
        assert len(inp_tokens) == 3
        t1, t2, _ = inp_tokens
        b1 = f1_dict.get((t1, t2))
        group_dict[b1].append(ex)
    return dict(group_dict)

def group_data_by_t_final(examples, f1_dict, f2_dict):
    """
    For each example: parse (t0,t1,t2),
    b1 = f1_dict.get((t0,t1),'unknown')
    y  = f2_dict.get((b1,t1,t2),'unknown') if b1 != 'unknown'
    group by y => group_dict[y] = list of examples
    """
    group_dict = defaultdict(list)
    for ex in examples:
        inp_tokens = parse_tokens(ex["input_text"])
        assert len(inp_tokens) == 3
        t1, t2, t3 = inp_tokens
        b1 = f1_dict.get((t1, t2))
        t_final = f2_dict.get((b1, t2, t3))
        group_dict[t_final].append(ex)
    return dict(group_dict)

def group_data_by_t2(examples):
    """
    For each example: parse (t0,t1,t2),
    group by t2 => group_dict[t2] = list of examples
    """
    group_dict = defaultdict(list)
    for ex in examples:
        inp_tokens = parse_tokens(ex["input_text"])
        assert len(inp_tokens) == 3
        t1, t2, t3 = inp_tokens
        group_dict[t2].append(ex)
    return dict(group_dict)

def group_data_by_b1_t2(examples, f1_dict, f2_dict):
    """
    Returns a dictionary mapping (b1, t2) pair to a list of examples.
    """
    group_dict = defaultdict(list)
    for ex in examples:
        inp_tokens = parse_tokens(ex["input_text"])
        assert len(inp_tokens) == 3
        t1, t2, t3 = inp_tokens
        b1 = f1_dict.get((t1, t2))
        group_dict[f"{b1},{t2}"].append(ex)
    return dict(group_dict)


###############################################################################
# 4) Splitting data into ID train, ID test, OOD test
###############################################################################
def load_and_preprocess_data(f1_dict, f2_dict, test_path, idx):
    """
    데이터를 로드하고 전처리합니다.
    
    Args:
        f1_dict: 첫 번째 함수의 매핑 딕셔너리
        f2_dict: 두 번째 함수의 매핑 딕셔너리
        test_path: 테스트 데이터 파일 경로
        idx: 그룹핑 방식 (1: b1 기준, 2: y 기준, 3: t2 기준, 4: b1+t2 기준)
    
    Returns:
        grouped_id_train_data: ID 학습 데이터 그룹
        grouped_id_test_data: ID 테스트 데이터 그룹
        grouped_ood_test_data: OOD 테스트 데이터 그룹
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
            else:
                raise NotImplementedError(f"Invalid idx value: {idx}")
        else:
            raise NotImplementedError("Invalid coverage type")
            
    if idx == 1:
        grouped_id_train_data = group_data_by_b1(id_train_data, f1_dict)
        grouped_id_test_data = group_data_by_b1(id_test_data, f1_dict)
        grouped_ood_test_data = group_data_by_b1(ood_test_data, f1_dict)
    elif idx == 2:
        grouped_id_train_data = group_data_by_t_final(id_train_data, f1_dict, f2_dict)
        grouped_id_test_data = group_data_by_t_final(id_test_data, f1_dict, f2_dict)
        grouped_ood_test_data = group_data_by_t_final(ood_test_data, f1_dict, f2_dict)
    elif idx == 3:
        grouped_id_train_data = group_data_by_t2(id_train_data)
        grouped_id_test_data = group_data_by_t2(id_test_data)
        grouped_ood_test_data = group_data_by_t2(ood_test_data)
    elif idx == 4:
        grouped_id_train_data = group_data_by_b1_t2(id_train_data, f1_dict, f2_dict)
        grouped_id_test_data = group_data_by_b1_t2(id_test_data, f1_dict, f2_dict)
        grouped_ood_test_data = group_data_by_b1_t2(ood_test_data, f1_dict, f2_dict)
    else:
        raise NotImplementedError("atomic_idx must be 1 (group by b1), 2 (group by y), 3 (group by t2), or 4 (group by b1+t2)")

    return grouped_id_train_data, grouped_id_test_data, grouped_ood_test_data


###############################################################################
# 5) Extracting hidden states
#    (Same approach: 'residual' uses model outputs; 'post_mlp' uses hooks)
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
    For each group, processes examples in batches of size 'batch_size'
    using batch inference.
    """
    results = defaultdict(list)
    for bridge_entity, instances in tqdm(data_group.items(), desc="Processing instances"):
        for i in range(0, len(instances), batch_size):
            batch = instances[i:i+batch_size]
            input_texts = [ex['target_text'] for ex in batch]
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
# 6) Deduplication
###############################################################################
# def deduplicate_vectors(results):
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
    grouped_data: 그룹핑된 데이터. 형식은 { group_key: [entry, entry, ...] }이며,
                  각 entry는 "input_text"와 "target_text"를 포함하는 dict입니다.
    atomic_idx: deduplication 기준을 결정하는 인덱스
                - 1이면, target_text의 처음 두 토큰(t1, t2) 기준 deduplication
                - 2이면, 처음 세 토큰(t1, t2, t3) 기준 deduplication

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
            elif atomic_idx == 2:
                dedup_key = tuple(tokens[:3])  # (t1, t2, t3)
            elif atomic_idx == 3:
                dedup_key = tuple(tokens[:3])  # (t1, t2, t3)
            elif atomic_idx == 4:
                dedup_key = tuple(tokens[:2])  # (t1, t2)
            else:
                raise ValueError("atomic_idx must be 1, 2, 3 or 4")

            if dedup_key not in deduped:
                deduped[dedup_key] = entry
        output[group_key] = list(deduped.values())

    return output

def save_results(results, save_path):
    """결과를 JSON 파일로 저장합니다."""
    with open(save_path, "w") as f:
        json.dump(results, f)
    logging.info(f"Saved results to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to the model checkpoint")
    parser.add_argument("--layer_pos_pairs", required=True, help="List of (layer, position) tuples to evaluate")
    parser.add_argument("--data_dir", default=None, help="directory for dataset")
    parser.add_argument("--save_dir", required=True, help="Directory to save the analysis results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for verbose output")
    parser.add_argument("--atomic_idx", required=True, type=int, choices=[1,2,3,4], help="which function's bridging entity to group by: 1 => b1, 2 => final y, 3 => t2, 4 => b1 + t2")
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
    atomic_file_1 = os.path.join(data_dir, "atomic_facts_f1.json")
    atomic_file_2 = os.path.join(data_dir, "atomic_facts_f2.json")
    f1_dict, f2_dict = load_atomic_facts_nontree(atomic_file_1, atomic_file_2)

    grouped_id_train_data, grouped_id_test_data, grouped_ood_test_data = load_and_preprocess_data(
        f1_dict, f2_dict, os.path.join(data_dir, "test.json"), idx=args.atomic_idx
    )
    
    # 정규식을 사용한 파싱 로직
    if "logit" in args.layer_pos_pairs or "prob" in args.layer_pos_pairs:
        pos_match = re.search(r"\((logit|prob|\d+),(\d+)\)", args.layer_pos_pairs)
        if pos_match:
            layer_type = pos_match.group(1)
            pos = int(pos_match.group(2))
            layer_pos_pairs = [(layer_type, pos)]
        else:
            raise ValueError("Invalid layer_pos_pairs format")
    else:
        layer_pos_pairs = eval(args.layer_pos_pairs)
    
    logging.info(f"Layer position pairs: {layer_pos_pairs}")
    
    logging.info(f"ID train groups: {len(grouped_id_train_data)}")
    logging.info(f"ID test groups: {len(grouped_id_test_data)}")
    logging.info(f"OOD test groups: {len(grouped_ood_test_data)}")
    
    torch.manual_seed(0)
    
    logging.info("Processing ID train group with batch processing...")
    id_train_results = process_data_group(model, grouped_id_train_data, layer_pos_pairs, tokenizer, device, args.mode, batch_size=args.batch_size)
    
    logging.info("Processing ID test group with batch processing...")
    id_test_results = process_data_group(model, grouped_id_test_data, layer_pos_pairs, tokenizer, device, args.mode, batch_size=args.batch_size)
    
    logging.info("Processing OOD test group with batch processing...")
    ood_test_results = process_data_group(model, grouped_ood_test_data, layer_pos_pairs, tokenizer, device, args.mode, batch_size=args.batch_size)
    
    # Perform deduplication
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
    
    # 작은따옴표를 제거한 경로 생성
    layer_pos_str = str(layer_pos_pairs[0]).replace("'", "").replace(" ", "")
    save_dir = os.path.join(args.save_dir, args.mode, dataset, f"f{args.atomic_idx}", layer_pos_str, step)
    
    # 저장 디렉토리 처리
    if os.path.exists(save_dir):
        if args.overwrite:
            logging.info(f"Overwriting existing directory: {save_dir}")
            import shutil
            shutil.rmtree(save_dir)
            os.makedirs(save_dir)
        else:
            logging.warning(f"Directory already exists: {save_dir}")
            logging.warning("Use --overwrite flag to overwrite existing directory")
            return
    else:
        os.makedirs(save_dir)
        logging.info(f"Created new directory: {save_dir}")

    # 결과 저장
    save_results(id_train_dedup, os.path.join(save_dir, "id_train_dedup.json"))
    save_results(id_test_dedup, os.path.join(save_dir, "id_test_dedup.json"))
    save_results(ood_dedup, os.path.join(save_dir, "ood_dedup.json"))

    logging.info("Finished all. Final deduplication stats:")
    logging.info(f"ID train: {len(id_train_dedup)} groups, ID test: {len(id_test_dedup)} groups, OOD: {len(ood_dedup)} groups")
    logging.info("Done.")

if __name__ == "__main__":
    main()