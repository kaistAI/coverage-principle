import argparse
import logging
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
from collections import defaultdict


def setup_logging(debug_mode):
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')


def return_rank(hd, word_embedding_, token_ids_list, metric='dot'):
    if metric == 'dot':
        word_embedding = word_embedding_
    elif metric == 'cos':
        word_embedding = F.normalize(word_embedding_, p=2, dim=1)
    else:
        assert False

    logits_ = torch.matmul(hd, word_embedding.T)
    batch_size, seq_len, vocab_size = logits_.shape
    token_ids_list = torch.tensor(token_ids_list).view(batch_size, 1, 1).expand(batch_size, seq_len, vocab_size).to(logits_.device)
    _, sorted_indices = logits_.sort(dim=-1, descending=True)
    rank = (sorted_indices == token_ids_list).nonzero(as_tuple=True)[-1].view(batch_size, seq_len).cpu()
    return rank

def parse_tokens(text):
    """
    Separates the given string based on '<' and '>' to generate a token list.
    Example: "<t_5><t_23><t_17><t_42></a>" -> ["t_5", "t_23", "t_17", "t_42"]
    """
    tokens = text.replace("</a>", "").strip("><").split("><")
    return tokens

def parse_atomic_fact(atomic_facts_f1, atomic_facts_f2):
    
    def _func(atomic_facts):
        result = {}
        for item in atomic_facts:
            tokens_in = parse_tokens(item["input_text"])
            tokens_out = parse_tokens(item["target_text"])
            assert len(tokens_in) == 2 and len(tokens_out) == 3 and tokens_out[:2] == tokens_in
            result[(tokens_out[0], tokens_out[1])] = tokens_out[2]
        return result

    f1_dict = _func(atomic_facts_f1)
    f2_dict = _func(atomic_facts_f2)
    
    return f1_dict, f2_dict


def build_seen_atomic_dicts(f1_dict, f2_dict, train_data):
    '''
    f1_dict : (inp_token1, inp_token2) -> out_token
    '''
    seen_f1_dict, seen_f2_dict = {}, {}
    for data in train_data:
        tokens = parse_tokens(data["target_text"])
        assert len(tokens) == 4
        t1, t2, t3, t = tokens
        b1 = f1_dict[(t1, t2)]
        seen_f1_dict[(t1, t2)] = b1
        assert f2_dict[(b1, t3)] == t
        seen_f2_dict[(b1, t3)] = t
    return seen_f1_dict, seen_f2_dict


def group_by_target(data, f1_dict = None):
    """
    data: train.json (or test.json) entry list. Each entry has "input_text" and "target_text".
    f1_dict : (inp_token1, inp_token2) -> out_token
    """
    grouped = {}
    for entry in data:
        tokens = parse_tokens(entry["target_text"])
        t1, t2, t3, t = tokens
        assert f1_dict != None
        key = f1_dict[(t1, t2)]
        grouped.setdefault(key, []).append(entry)
    return grouped


def deduplicate_grouped_data(grouped_data):
    """
    grouped_data: Grouped data. Format is { group_key: [entry, entry, ...] },
                  where each entry is a dict containing "input_text" and "target_text".

    Returns:
        Deduplicated list of entries. Only one entry remains for entries with the same deduplication key.
    """
    output = {}
    for group_key, entries in grouped_data.items():
        deduped = {}
        for entry in entries:
            tokens = parse_tokens(entry["target_text"])
            dedup_key = tuple(tokens[:2])  # (t1, t2)

            if dedup_key not in deduped:
                deduped[dedup_key] = entry
        output[group_key] = list(deduped.values())

    return output

def intervene_and_measure(original_data, 
                          model, 
                          tokenizer, 
                          seen_b1_to_t1t2,
                          seen_f2_dict,
                          device, 
                          batch_size=32,
                          metric_type="rank"):
    """
    original_data: { bridge_entity: [entry, ...], ... }
    model: Trained Decoder-only Transformer
    tokenizer: Tokenizer
    device: Execution device (e.g., "cuda")
    batch_size: Mini-batch size
    metric_type: "rank" or "prob" - measurement method
    """
    results = []
    skipped_data = 0

    all_original_inputs = []
    all_real_t = []
    all_changed_t = []
    all_query = []
    all_injection_pos = []  # List to store injection positions

    # Each entry's target_text is assumed to be in the form [t1, t2, t3, t]
    for bridge_entity, entries in original_data.items():
        for entry in entries:
            original_input = entry['input_text']
            tokens = parse_tokens(entry['target_text'])
            assert len(tokens) == 4
            t1, t2, t3, t = tokens

            # Generate candidate query and changed t (first relation perturb)
            candidate_set = set()
            for b1_candidate in seen_b1_to_t1t2.keys():
                if b1_candidate == bridge_entity:
                    continue
                # f2 mapping: (b1_candidate, t3) -> t must exist
                if (b1_candidate, t3) not in seen_f2_dict:
                    continue
                candidate_t = seen_f2_dict[(b1_candidate, t3)]
                if candidate_t != t:
                    candidate_set.add((b1_candidate, candidate_t))
            candidate_hr_set = set()
            for b1_candidate, cand_t in candidate_set:
                for (t1_candidate, t2_candidate) in seen_b1_to_t1t2[b1_candidate]:
                    if t1_candidate == t1 and t2_candidate != t2:
                        candidate_hr_set.add((t1, t2_candidate, t3, cand_t))
            if len(candidate_hr_set) == 0:
                skipped_data += 1
                continue
            candidate_list = sorted(list(candidate_hr_set))
            selected_candidate = candidate_list[np.random.randint(0, len(candidate_list))]
            # query_text is perterbed input: "<t1><t2>'<t3>"
            query_text = ''.join([f"<{token}>" for token in selected_candidate[:3]])
            changed_t = selected_candidate[-1]
            
            for injection_pos in range(3):  # injection for all input positions
                all_original_inputs.append(original_input)
                all_real_t.append(t)
                all_changed_t.append(changed_t)
                all_query.append(query_text)
                all_injection_pos.append(injection_pos)
    
    print(f"The number of data skipped because appropriate intervene data was not seen during training: {skipped_data}")
    if len(all_original_inputs) == 0:
        print(f"All data did not find intervene data")
        return results

    num_samples = len(all_original_inputs)
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        current_original_inputs = all_original_inputs[start:end]
        current_real_t = all_real_t[start:end]
        current_changed_t = all_changed_t[start:end]
        current_query = all_query[start:end]
        current_injection_pos = all_injection_pos[start:end]

        # 1. Model forward for original input
        tokenizer_output = tokenizer(current_original_inputs, return_tensors="pt", padding=True)
        input_ids = tokenizer_output["input_ids"].to(device)
        attention_mask = tokenizer_output["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        original_hidden_states = outputs['hidden_states']
        word_embedding = model.lm_head.weight.data

        # Target token processing (actual target, changed target)
        tokenized_real_t = tokenizer([f"<{t}>" for t in current_real_t], return_tensors="pt", padding=True)
        tokenized_changed_t = tokenizer([f"<{t}>" for t in current_changed_t], return_tensors="pt", padding=True)
        input_ids_real = tokenized_real_t["input_ids"]
        input_ids_changed = tokenized_changed_t["input_ids"]

        if metric_type == "rank":
            rank_before_real = return_rank(original_hidden_states[-1], word_embedding, input_ids_real)[:, -1].tolist()
            rank_before_changed = return_rank(original_hidden_states[-1], word_embedding, input_ids_changed)[:, -1].tolist()
        else:  # metric_type == "prob"
            logits = torch.matmul(original_hidden_states[-1], word_embedding.T)
            probs = F.softmax(logits, dim=-1)
            batch_size, seq_len, vocab_size = probs.shape
            input_ids_changed_expanded = input_ids_changed.view(batch_size, 1, 1).expand(batch_size, seq_len, vocab_size).to(device)
            prob_before_changed = probs.gather(-1, input_ids_changed_expanded)[:, -1, 0].tolist()

        # Free unnecessary memory
        del outputs
        torch.cuda.empty_cache()

        # 2. Model forward for query input (for intervention)
        tokenizer_output_query = tokenizer(current_query, return_tensors="pt", padding=True)
        input_ids_query = tokenizer_output_query["input_ids"].to(device)
        attention_mask_query = tokenizer_output_query["attention_mask"].to(device)
        with torch.no_grad():
            outputs_query = model(
                input_ids=input_ids_query,
                attention_mask=attention_mask_query,
                output_hidden_states=True
            )
        query_hidden_states = outputs_query['hidden_states']

        if metric_type == "prob":
            logits_query = torch.matmul(query_hidden_states[-1], word_embedding.T)
            probs_query = F.softmax(logits_query, dim=-1)
            batch_size, seq_len, vocab_size = probs_query.shape
            input_ids_changed_expanded = input_ids_changed.view(batch_size, 1, 1).expand(batch_size, seq_len, vocab_size).to(device)
            prob_query_changed = probs_query.gather(-1, input_ids_changed_expanded)[:, -1, 0].tolist()

        # Free unnecessary memory
        del outputs_query
        torch.cuda.empty_cache()

        # 3. intervention: perform for each sample at specified injection_pos
        if metric_type == "rank":
            intervened_ranks_real = {}
            intervened_ranks_changed = {}
        else:  # metric_type == "prob"
            intervened_metrics_changed = {}

        for layer_to_intervene in range(1, 8):
            hidden_states = original_hidden_states[layer_to_intervene].clone()
            
            # Perform intervention for each sample at specified injection_pos
            for i, pos in enumerate(current_injection_pos):
                hidden_states[i, pos, :] = query_hidden_states[layer_to_intervene][i, pos, :]
            
            intervened_hidden = hidden_states
            for i in range(layer_to_intervene, 8):
                f_layer = model.transformer.h[i]
                # Attention block
                residual = intervened_hidden
                intervened_hidden = f_layer.ln_1(intervened_hidden)
                attn_output = f_layer.attn(intervened_hidden)[0]
                intervened_hidden = attn_output + residual
                # MLP block
                residual = intervened_hidden
                intervened_hidden = f_layer.ln_2(intervened_hidden)
                feed_forward_hidden = f_layer.mlp.c_proj(f_layer.mlp.act(f_layer.mlp.c_fc(intervened_hidden)))
                intervened_hidden = residual + feed_forward_hidden
            # Final layer norm
            intervened_hidden = model.transformer.ln_f(intervened_hidden)

            if metric_type == "rank":
                rank_after_real = return_rank(intervened_hidden, word_embedding, input_ids_real)[:, -1].tolist()
                rank_after_changed = return_rank(intervened_hidden, word_embedding, input_ids_changed)[:, -1].tolist()
                intervened_ranks_real[layer_to_intervene] = rank_after_real
                intervened_ranks_changed[layer_to_intervene] = rank_after_changed
            else:  # metric_type == "prob"
                logits = torch.matmul(intervened_hidden, word_embedding.T)
                probs = F.softmax(logits, dim=-1)
                batch_size, seq_len, vocab_size = probs.shape
                input_ids_changed_expanded = input_ids_changed.view(batch_size, 1, 1).expand(batch_size, seq_len, vocab_size).to(device)
                prob_after_changed = probs.gather(-1, input_ids_changed_expanded)[:, -1, 0].tolist()
                intervened_metrics_changed[layer_to_intervene] = prob_after_changed

            # Free unnecessary memory
            del intervened_hidden
            torch.cuda.empty_cache()

        # 4. Generate result dict for each sample in mini-batch and save to results list
        for i in range(len(current_original_inputs)):
            result_dict = {}
            result_dict["injection_pos"] = current_injection_pos[i]  # Add injection position information
            
            if metric_type == "rank":
                result_dict["rank_before_real_t"] = rank_before_real[i]
                result_dict["rank_before_changed_t"] = rank_before_changed[i]
                for layer in range(1, 8):
                    result_dict[f"rank_after_{layer}_real_t"] = intervened_ranks_real[layer][i]
                    result_dict[f"rank_after_{layer}_changed_t"] = intervened_ranks_changed[layer][i]
            else:  # metric_type == "prob"
                result_dict["prob_before_changed_t"] = prob_before_changed[i]
                result_dict["prob_query_changed_t"] = prob_query_changed[i]
                for layer in range(1, 8):
                    result_dict[f"prob_after_{layer}_changed_t"] = intervened_metrics_changed[layer][i]
            
            results.append(result_dict)

        # Free unnecessary memory
        del query_hidden_states
        torch.cuda.empty_cache()
        
    assert len(results) == num_samples

    return results

def load_and_preprocess_data(f1_dict, test_data, detailed_grouping=2):
    """
    Parse test.json, filter examples by type, and group them using atomic facts
    
    Args:
        f1_dict: Atomic facts dictionary
        test_data: Test data
        detailed_grouping: Grouping level (1: k<3/k>=3, 2: k<3/k=3/k>3, 3: each covered_N)
    """
    id_train_data = []
    id_test_data = []
    ood_test_data = []
    
    # Dictionary to store detailed groups according to covered_ values
    covered_groups = defaultdict(list)
    
    for d in test_data:
        if d['type'] == 'train_inferred':
            id_train_data.append(d)
        elif d['type'] == 'type_0':
            id_test_data.append(d)
        elif d['type'].startswith("covered_"):
            n = int(d['type'].split("_")[-1])
            covered_groups[n].append(d)
        elif d['type'] in ['type_1', 'type_2']:
            ood_test_data.append(d)
        elif d['type'] in ['type_3']:
            pass
        else:
            raise NotImplementedError("Invalid coverage type")
            
    grouped_id_train_data = group_by_target(id_train_data, f1_dict)
    grouped_id_test_data = group_by_target(id_test_data, f1_dict)
    grouped_ood_test_data = group_by_target(ood_test_data, f1_dict)
    
    if detailed_grouping == 3:
        # Group by each covered_N
        grouped_covered_data = {n: group_by_target(group, f1_dict) for n, group in covered_groups.items()}
    elif detailed_grouping == 2:
        # Divide into three groups: k<3, k=3, k>3
        low_cutoff = []
        mid_cutoff = []
        high_cutoff = []
        for n, group in covered_groups.items():
            if n < 3:
                low_cutoff.extend(group)
            elif n == 3:
                mid_cutoff.extend(group)
            else:
                high_cutoff.extend(group)
        grouped_covered_data = {
            'low_cutoff': group_by_target(low_cutoff, f1_dict),
            'mid_cutoff': group_by_target(mid_cutoff, f1_dict),
            'high_cutoff': group_by_target(high_cutoff, f1_dict)
        }
    elif detailed_grouping == 1:
        # Divide into two groups: k<3, k>=3
        low_cutoff = []
        high_cutoff = []
        for n, group in covered_groups.items():
            if n < 3:
                low_cutoff.extend(group)
            else:
                high_cutoff.extend(group)
        grouped_covered_data = {
            'low_cutoff': group_by_target(low_cutoff, f1_dict),
            'high_cutoff': group_by_target(high_cutoff, f1_dict)
        }
    else:
        raise NotImplementedError(f"Invalid detailed_grouping value: {detailed_grouping}")
    
    return {
        'id_train': grouped_id_train_data,
        'id_test': grouped_id_test_data,
        'covered': grouped_covered_data,
        'ood': grouped_ood_test_data
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to the model checkpoint")
    parser.add_argument("--step_list", default=None, nargs="+", help="checkpoint's steps to check causal strength")
    parser.add_argument("--data_dir", default=None, help="directory for dataset")
    parser.add_argument("--device", default="cuda", help="Device to run the model on")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--detailed_grouping", type=int, required=True, choices=[1,2,3], help="Grouping level for covered data: 1=k<3/k>=3, 2=k<3/k=3/k>3, 3=each covered_N")
    parser.add_argument("--metric_type", type=str, choices=["rank", "prob"], default="rank",
                         help="Measurement method: rank or probability")
    
    args = parser.parse_args()
    setup_logging(args.debug)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = base_dir
    
    if args.model_dir.split("/")[-1] == "":
        dataset = args.model_dir.split("/")[-2].split("_")[0]
    else:
        dataset = args.model_dir.split("/")[-1].split("_")[0]

    logging.debug(f"base_dir: {base_dir}")
    logging.debug(f"data_dir: {data_dir}")
    logging.debug(f"dataset: {dataset}")
    
    # Load atomic facts
    with open(os.path.join(data_dir, "data", dataset, "atomic_facts_f1.json"), "r") as f:
        atomic_facts_f1 = json.load(f)
    with open(os.path.join(data_dir, "data", dataset, "atomic_facts_f2.json"), "r") as f:
        atomic_facts_f2 = json.load(f)

    # (inp_token1, inp_token2) -> out_token
    f1_dict, f2_dict = parse_atomic_fact(atomic_facts_f1, atomic_facts_f2)
    
    # Consisting only of atomic facts shown during training
    with open(os.path.join(data_dir, "data", dataset, "train.json"), "r") as f:
        train_data = json.load(f)
    # (inp_token1, inp_token2) -> out_token
    seen_f1_dict, seen_f2_dict = build_seen_atomic_dicts(f1_dict, f2_dict, train_data)

    # for convenience
    # b1 -> set((t1, t2))
    seen_b1_to_t1t2 = {}
    for (t1, t2), b1 in seen_f1_dict.items():
        seen_b1_to_t1t2.setdefault(b1, set()).add((t1, t2))
    
    with open(os.path.join(data_dir, "data", dataset, "test_annotated.json"), "r") as f:
        test_data = json.load(f)
    
    grouped_data = load_and_preprocess_data(f1_dict, test_data, detailed_grouping=args.detailed_grouping)
    
    # deduplicate data according to atomic_idx
    original_train_dedup = deduplicate_grouped_data(grouped_data['id_train'])
    original_test_id_dedup = deduplicate_grouped_data(grouped_data['id_test'])
    
    if args.detailed_grouping == 3:
        original_covered_dedup = {
            f"covered_{n}": deduplicate_grouped_data(group) 
            for n, group in grouped_data['covered'].items()
        }
    elif args.detailed_grouping == 2:
        original_covered_dedup = {
            'covered_low_cutoff': deduplicate_grouped_data(grouped_data['covered']['low_cutoff']),
            'covered_mid_cutoff': deduplicate_grouped_data(grouped_data['covered']['mid_cutoff']),
            'covered_high_cutoff': deduplicate_grouped_data(grouped_data['covered']['high_cutoff'])
        }
    elif args.detailed_grouping == 1:
        original_covered_dedup = {
            'covered_low_cutoff': deduplicate_grouped_data(grouped_data['covered']['low_cutoff']),
            'covered_high_cutoff': deduplicate_grouped_data(grouped_data['covered']['high_cutoff'])
        }
    else:
        raise NotImplementedError(f"Invalid detailed_grouping value: {args.detailed_grouping}")

    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    all_checkpoints = [
        checkpoint_name for step in args.step_list
        for checkpoint_name in (["final_checkpoint"] if step == "final_checkpoint" else [f"checkpoint-{step}"])
        if os.path.exists(os.path.join(args.model_dir, checkpoint_name))
    ]
    
    all_checkpoints.sort(key=lambda x: float('inf') if x == "final_checkpoint" else int(x.split("-")[1]))
    logging.info(f"Found checkpoints: {all_checkpoints}")
    
    results = {}
    
    for checkpoint in all_checkpoints:
        result_ckpt = {}
        logging.info(f"Now checkpoint {checkpoint}")
        step = checkpoint.split("-")[-1]
        model_path = os.path.join(args.model_dir, checkpoint)
        model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        model.eval()
        
        logging.info("Train data intervene exp...")
        train_results = intervene_and_measure(original_data=original_train_dedup, 
                                              model=model, 
                                              tokenizer=tokenizer, 
                                              seen_b1_to_t1t2=seen_b1_to_t1t2, 
                                              seen_f2_dict=seen_f2_dict,
                                              device=device, 
                                              batch_size=args.batch_size,
                                              metric_type=args.metric_type)
        logging.info("Test data intervene exp...")
        test_results = intervene_and_measure(original_data=original_test_id_dedup, 
                                              model=model, 
                                              tokenizer=tokenizer, 
                                              seen_b1_to_t1t2=seen_b1_to_t1t2, 
                                              seen_f2_dict=seen_f2_dict,
                                              device=device, 
                                              batch_size=args.batch_size,
                                              metric_type=args.metric_type)
        
        result_ckpt["train_inferred"] = train_results
        result_ckpt["test_inferred"] = test_results
        
        for coverage, group in original_covered_dedup.items():
            logging.info(f"Test data with {coverage} intervene exp...")
            coverage_results = intervene_and_measure(original_data=group, 
                                                model=model, 
                                                tokenizer=tokenizer, 
                                                seen_b1_to_t1t2=seen_b1_to_t1t2, 
                                                seen_f2_dict=seen_f2_dict,
                                                device=device, 
                                                batch_size=args.batch_size,
                                                metric_type=args.metric_type)
            result_ckpt[f"test_inferred_{coverage}"] = coverage_results
        results[checkpoint] = result_ckpt
    
    save_file_name = f"{args.model_dir.split('/')[-1]}_residual_diff_f1_{args.metric_type}"
    if args.detailed_grouping:
        save_file_name += f"_detailed_grouping_{args.detailed_grouping}"
    out_dir = os.path.join(base_dir, "circuit_analysis", "tracing_results", "2-hop")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{save_file_name}.json"), "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    # Post-processing: Summarize intervention before/after changes for each checkpoint
    logging.info(f"Summarizing Intervention Exp. result...")
    refined_results = {}
    all_ckpts = list(results.keys())
    all_ckpts.sort(key=lambda x: float('inf') if x == "final_checkpoint" else int(x.split("-")[1]))

    for checkpoint in all_ckpts:
        results_4_ckpt = {}
        for data_type, entries in results[checkpoint].items():
            # Save results for each position
            results_by_pos = {0: {}, 1: {}, 2: {}}
            for pos in range(3):
                sample_num = 0
                result_4_type = dict()
                
                # Filter only entries for the corresponding position
                pos_entries = [entry for entry in entries if entry["injection_pos"] == pos]
                
                if args.metric_type == "rank":
                    for i in range(1,8):
                        result_4_type[f"real_t>changed_t_layer{i}"] = 0
                        result_4_type[f"rank(changed_t)<=5_layer{i}"] = 0
                        result_4_type[f"both_layer{i}"] = 0
                    
                    for entry in pos_entries:
                        # Exclude cases where the actual t is originally lower than changed t (intervention method is meaningless)
                        if entry["rank_before_real_t"] > entry["rank_before_changed_t"]:
                            continue
                        sample_num += 1
                        for i in range(1,8):
                            if entry[f"rank_after_{i}_changed_t"] <= 5:
                                result_4_type[f"rank(changed_t)<=5_layer{i}"] += 1
                            if entry[f"rank_after_{i}_real_t"] > entry[f"rank_after_{i}_changed_t"]:
                                result_4_type[f"real_t>changed_t_layer{i}"] += 1
                                if entry[f"rank_after_{i}_changed_t"] <= 5:
                                    result_4_type[f"both_layer{i}"] += 1
                    
                    result_4_type["sample_num"] = sample_num
                    result_4_type["passed_data_num"] = len(pos_entries) - sample_num
                else:  # metric_type == "prob"
                    for i in range(1,8):
                        result_4_type[f"relative_prob_change_layer{i}"] = 0.0
                    
                    valid_sample_num = 0  # Number of samples with valid denominator
                    for entry in pos_entries:
                        # Check if prob_after_changed_t is greater than prob_query_changed_t for all layers
                        skip_entry = False
                        for i in range(1,8):
                            if entry[f"prob_after_{i}_changed_t"] > entry["prob_query_changed_t"]:
                                skip_entry = True
                                break
                        if skip_entry:
                            continue
                        if entry["prob_query_changed_t"] < entry["prob_before_changed_t"]:
                            continue
                        denominator = entry["prob_query_changed_t"] - entry["prob_before_changed_t"]
                        if abs(denominator) < 1e-6:  # Skip if denominator is too small
                            continue
                        valid_sample_num += 1
                        for i in range(1,8):
                            numerator = entry[f"prob_after_{i}_changed_t"] - entry["prob_before_changed_t"]
                            if numerator / denominator > 1:
                                print(f"layer: {i}\n{entry}")
                            result_4_type[f"relative_prob_change_layer{i}"] += numerator / denominator
                    
                    # Calculate average
                    if valid_sample_num > 0:
                        for i in range(1,8):
                            result_4_type[f"relative_prob_change_layer{i}"] /= valid_sample_num
                    
                    result_4_type["sample_num"] = valid_sample_num
                    result_4_type["total_sample_num"] = len(pos_entries)  # Also store total sample count
                
                results_by_pos[pos] = result_4_type
            
            results_4_ckpt[data_type] = results_by_pos
        refined_results[checkpoint] = results_4_ckpt
    
    with open(os.path.join(out_dir, f"{save_file_name}_refined.json"), "w", encoding='utf-8') as f:
        json.dump(refined_results, f, indent=4)
        
if __name__=="__main__":
    main()