import argparse
import logging
import os
import json
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F


def setup_logging(debug_mode):
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    

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


def load_and_preprocess_data(f1_dict, test_data):
    """
    Parse test.json, filter examples by type, and group them using atomic facts
    
    Args:
        f1_dict: Atomic facts dictionary
        test_data: Test data
    """
    id_train_data = []
    id_test_data = []
    ood_test_data = []
    id_ood_test_data = []
    
    for d in test_data:
        if d['type'] == 'train_inferred':
            id_train_data.append(d)
        elif d['type'] == 'type_0':
            id_test_data.append(d)
        elif d['type'] in ['type_1', 'type_2']:
            ood_test_data.append(d)
        elif d['type'] in ['type_3']:
            id_ood_test_data.append(d)
        else:
            raise NotImplementedError("Invalid coverage type")
            
    grouped_id_train_data = group_by_target(id_train_data, f1_dict)
    grouped_id_test_data = group_by_target(id_test_data, f1_dict)
    grouped_ood_test_data = group_by_target(ood_test_data, f1_dict)
    grouped_id_ood_test_data = group_by_target(id_ood_test_data, f1_dict)
    
    return {
        'id_train': grouped_id_train_data,
        'id_test': grouped_id_test_data,
        'ood': grouped_ood_test_data,
        'id_ood': grouped_id_ood_test_data
    }
    

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

def logit_lens(data, model, tokenizer, device, batch_size):
    """
    data: deduplicated data
    """
    all_original_inputs = []
    all_bridge_entities = []
    
    # 1. Store all inputs and bridge entities in lists
    for bridge_entity, entries in data.items():
        for entry in entries:
            all_original_inputs.append(entry['input_text'])
            all_bridge_entities.append(bridge_entity)
    
    results = {}
    
    num_samples = len(all_original_inputs)
    # 2. Process in batches
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        batch_inputs = all_original_inputs[start:end]
        batch_bridges = all_bridge_entities[start:end]
        temp_dict = dict()

        # Tokenization
        tokenizer_output = tokenizer(batch_inputs, return_tensors="pt", padding=True)
        input_ids = tokenizer_output["input_ids"].to(device)
        attention_mask = tokenizer_output["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        # Get all hidden states
        all_hidden_states = outputs['hidden_states']
        word_embedding = model.lm_head.weight.data
        
        # 4. Apply logit lens for each layer and position
        for layer_idx in range(1, len(all_hidden_states)):
            hidden_states = all_hidden_states[layer_idx]
            with torch.no_grad():
                temp = model.transformer.ln_f(hidden_states)
                
            b_logit_lens = return_rank(temp, word_embedding, tokenizer([f"<{bridge}>" for bridge in batch_bridges])["input_ids"])
            
            # Store results for each position
            for pos in range(b_logit_lens.size(1)):
                if f"layer_{layer_idx}" not in temp_dict:
                    temp_dict[f"layer_{layer_idx}"] = {}
                if pos not in temp_dict[f"layer_{layer_idx}"]:
                    temp_dict[f"layer_{layer_idx}"][pos] = []
                temp_dict[f"layer_{layer_idx}"][pos].extend(b_logit_lens[:, pos].tolist())
        
        # Merge batch results into final results
        for layer_idx in temp_dict:
            if layer_idx not in results:
                results[layer_idx] = {}
            for pos in temp_dict[layer_idx]:
                if pos not in results[layer_idx]:
                    results[layer_idx][pos] = []
                results[layer_idx][pos].extend(temp_dict[layer_idx][pos])
    
    for layer_idx in results:
        for pos in results[layer_idx]:
            assert len(results[layer_idx][pos]) == num_samples
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to the model checkpoint")
    parser.add_argument("--step_list", default=None, nargs="+", help="checkpoint's steps to check causal strength")
    parser.add_argument("--data_dir", default=None, help="directory for dataset")
    parser.add_argument("--device", default="cuda", help="Device to run the model on")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
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

    # Data preprocessing
    # (inp_token1, inp_token2) -> out_token
    f1_dict, f2_dict = parse_atomic_fact(atomic_facts_f1, atomic_facts_f2)
    
    with open(os.path.join(data_dir, "data", dataset, "test.json"), "r") as f:
        test_data = json.load(f)
    
    grouped_data = load_and_preprocess_data(f1_dict, test_data)
    
    # deduplicate data according to atomic_idx
    original_train_dedup = deduplicate_grouped_data(grouped_data['id_train'])
    original_test_id_dedup = deduplicate_grouped_data(grouped_data['id_test'])
    original_ood_dedup = deduplicate_grouped_data(grouped_data['ood'])
    original_id_ood_dedup = deduplicate_grouped_data(grouped_data['id_ood'])
    
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
        
        logging.info("Train data logit-lens exp...")
        train_results = logit_lens(original_train_dedup, 
                                   model, 
                                   tokenizer, 
                                   device, 
                                   args.batch_size)
        
        logging.info("Test data logit-lens exp...")
        test_results = logit_lens(original_test_id_dedup, 
                                   model, 
                                   tokenizer, 
                                   device, 
                                   args.batch_size)
        result_ckpt["train_inferred"] = train_results
        result_ckpt["test_inferred"] = test_results
        results[checkpoint] = result_ckpt
        
    save_file_name = f"{args.model_dir.split('/')[-1]}"
    out_dir = os.path.join(base_dir, "circuit_analysis", "logit-lens_results", "2-hop")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{save_file_name}.json"), "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    # Calculate average values for each data type, layer, and position
    avg_results = {}
    for checkpoint in results:
        avg_results[checkpoint] = {}
        for data_type in results[checkpoint]:
            avg_results[checkpoint][data_type] = {}
            for layer in results[checkpoint][data_type]:
                avg_results[checkpoint][data_type][layer] = {}
                for pos in results[checkpoint][data_type][layer]:
                    values = results[checkpoint][data_type][layer][pos]
                    avg_results[checkpoint][data_type][layer][pos] = sum(values) / len(values)

    # Save averaged results
    avg_save_path = os.path.join(out_dir, f"{save_file_name}_refined.json")
    with open(avg_save_path, "w", encoding='utf-8') as f:
        json.dump(avg_results, f, indent=4)
        
    
if __name__=="__main__":
    main()