import argparse
import logging
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json


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
    f1_dict : (inp_token1, inp_token2) -> bridge_token1
    f2_dict : (bridge_token1, inp_token3) -> out_token
    '''
    seen_f1_dict, seen_f2_dict = {}, {}
    for data in train_data:
        tokens = parse_tokens(data["target_text"])
        assert len(tokens) == 5
        t1, t2, t3, b1, t = tokens
        assert f1_dict[(t1, t2)] == b1
        seen_f1_dict[(t1, t2)] = b1
        assert f2_dict[(b1, t3)] == t
        seen_f2_dict[(b1, t3)] = t
    return seen_f1_dict, seen_f2_dict



def group_by_target(data, f1_dict):
    """
    data: train.json (or test.json) entry list. Each entry has "input_text" and "target_text".
    f1_dict : (inp_token1, inp_token2) -> out_token
    """
    grouped = {}
    for entry in data:
        tokens = parse_tokens(entry["target_text"])
        t1, t2, t3, b1, t = tokens
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
                          batch_size=32):
    """
    original_data: { bridge_entity: [entry, ...], ... }
    model: Trained Decoder-only Transformer
    tokenizer: Tokenizer
    device: Execution device (e.g., "cuda:0")
    batch_size: Mini-batch size
    """
    results = []
    skipped_data = 0

    all_original_inputs = []
    all_real_t = []
    all_changed_t = []
    all_query = []

    # Each entry's target_text is assumed to be in the form [t1, t2, t3, b1, t]
    for bridge_entity, entries in original_data.items():
        for entry in entries:
            original_input = entry['input_text']
            tokens = parse_tokens(entry['target_text'])
            assert len(tokens) == 5
            t1, t2, t3, b1, t = tokens
            assert b1 == bridge_entity

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
                        candidate_hr_set.add((t1_candidate, t2_candidate, cand_t))
            if len(candidate_hr_set) == 0:
                skipped_data += 1
                continue
            candidate_list = sorted(list(candidate_hr_set))
            selected_candidate = candidate_list[np.random.randint(0, len(candidate_list))]
            # query_text is f1 input: "<t1><t2>"
            query_text = ''.join([f"<{token}>" for token in selected_candidate[:2]])
            changed_t = selected_candidate[-1]
            injection_pos = 1  # During intervention, replace the second token position

            all_original_inputs.append(original_input)
            all_real_t.append(t)
            all_changed_t.append(changed_t)
            all_query.append(query_text)
    
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

        rank_before_real = return_rank(original_hidden_states[-1], word_embedding, input_ids_real)[:, -1].tolist()
        rank_before_changed = return_rank(original_hidden_states[-1], word_embedding, input_ids_changed)[:, -1].tolist()

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

        # Free unnecessary memory
        del outputs_query
        torch.cuda.empty_cache()

        # 3. intervention: token position to inject is injection_pos (varies by atomic_idx)
        intervened_ranks_real = {}
        intervened_ranks_changed = {}
        for layer_to_intervene in range(1, 8):
            hidden_states = original_hidden_states[layer_to_intervene].clone()
            hidden_states[:, injection_pos, :] = query_hidden_states[layer_to_intervene][:, injection_pos, :]
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

            rank_after_real = return_rank(intervened_hidden, word_embedding, input_ids_real)[:, -1].tolist()
            rank_after_changed = return_rank(intervened_hidden, word_embedding, input_ids_changed)[:, -1].tolist()

            intervened_ranks_real[layer_to_intervene] = rank_after_real
            intervened_ranks_changed[layer_to_intervene] = rank_after_changed

            # Free unnecessary memory
            del intervened_hidden
            torch.cuda.empty_cache()

        # 4. Generate result dict for each sample in mini-batch and save to results list
        for i in range(len(current_original_inputs)):
            result_dict = {}
            result_dict["rank_before_real_t"] = rank_before_real[i]
            result_dict["rank_before_changed_t"] = rank_before_changed[i]
            for layer in range(1, 8):
                result_dict[f"rank_after_{layer}_real_t"] = intervened_ranks_real[layer][i]
                result_dict[f"rank_after_{layer}_changed_t"] = intervened_ranks_changed[layer][i]
            results.append(result_dict)

        # Free unnecessary memory
        del query_hidden_states
        torch.cuda.empty_cache()

    return results


def get_total_list_length(json_data):
    total_length = 0
    for key, value in json_data.items():
        if isinstance(value, list):
            total_length += len(value)
    return total_length



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to the model checkpoint")
    parser.add_argument("--step_list", default=None, nargs="+", help="checkpoint's steps to check causal strength")
    parser.add_argument("--device", default="cuda", help="Device to run the model on")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    setup_logging(args.debug)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    if args.model_dir.split("/")[-1] == "":
        dataset = args.model_dir.split("/")[-2].split("_")[0]
    else:
        dataset = args.model_dir.split("/")[-1].split("_")[0]

    # Load atomic facts
    with open(os.path.join(base_dir, "data", dataset, "atomic_facts_f1.json"), "r") as f:
        atomic_facts_f1 = json.load(f)
    with open(os.path.join(base_dir, "data", dataset, "atomic_facts_f2.json"), "r") as f:
        atomic_facts_f2 = json.load(f)

    # (inp_token1, inp_token2) -> out_token
    f1_dict, f2_dict = parse_atomic_fact(atomic_facts_f1, atomic_facts_f2)
    
    # Consisting only of atomic facts shown during training
    with open(os.path.join(base_dir, "data", dataset, "train.json"), "r") as f:
        train_data = json.load(f)
    # (inp_token1, inp_token2) -> out_token
    seen_f1_dict, seen_f2_dict = build_seen_atomic_dicts(f1_dict, f2_dict, train_data)

    # for convenience
    # b1 -> set((t1, t2))
    seen_b1_to_t1t2 = {}
    for (t1, t2), b1 in seen_f1_dict.items():
        seen_b1_to_t1t2.setdefault(b1, set()).add((t1, t2))
    
    with open(os.path.join(base_dir, "data", dataset, "test.json"), "r") as f:
        test_data = json.load(f)
    
    test_id_data = [entry for entry in test_data if entry.get("type") == "type_0"]

    original_train = group_by_target(train_data, f1_dict=f1_dict)
    original_test_id  = group_by_target(test_id_data, f1_dict=f1_dict)
    
    # deduplicate data according to atomic_idx
    original_train_dedup = deduplicate_grouped_data(original_train)
    original_test_id_dedup = deduplicate_grouped_data(original_test_id)

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
        logging.info(f"\nNow checkpoint {checkpoint}")
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
                                              batch_size=args.batch_size)
        print("test data intervene exp...")
        test_results = intervene_and_measure(original_data=original_test_id_dedup, 
                                              model=model, 
                                              tokenizer=tokenizer, 
                                              seen_b1_to_t1t2=seen_b1_to_t1t2, 
                                              seen_f2_dict=seen_f2_dict,
                                              device=device, 
                                              batch_size=args.batch_size)
        result_ckpt["train_inferred"] = train_results
        result_ckpt["test_inferred"] = test_results
        results[checkpoint] = result_ckpt
    
    save_file_name = f"{args.model_dir.split('/')[-1]}_residual_diff_f1"
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
            sample_num = 0
            
            result_4_type = dict()
            for i in range(1,8):
                result_4_type[f"real_t>changed_t_layer{i}"] = 0
                result_4_type[f"rank(changed_t)<=5_layer{i}"] = 0
                result_4_type[f"both_layer{i}"] = 0
            
            for entry in entries:
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
            result_4_type["passed_data_num"] = len(entries) - sample_num
            results_4_ckpt[data_type] = result_4_type
        refined_results[checkpoint] = results_4_ckpt
    
    with open(os.path.join(out_dir, f"{save_file_name}_refined.json"), "w", encoding='utf-8') as f:
        json.dump(refined_results, f, indent=4)
        
if __name__=="__main__":
    main()