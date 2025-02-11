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


def intervene_and_measure(original_data, intervene_data, model, tokenizer, device, method="original", batch_size=10):
    """
    original_data: { bridge_entity: [entry, ...], ... }
    intervene_data: intervene data used when method=="original"
    model: Trained Decoder-only Transformer
    tokenizer: 
    device: 
    method: intervention method
         "original": data with the same bridge entity from intervene_data
         "gaussian": Gaussian random noise vector
         "query": random entity input
    """
    # Todo : Data Batch processing
    results = []
    skipped_data = 0
    word_embedding = model.lm_head.weight.data

    # 1. change original_data to flat list
    original_data_list = []
    for bridge_entity, entries in original_data.items():
        # If the same bridge entity has never been used in intervene_data, it cannot be intervened
        if bridge_entity in intervene_data:
            for entry in entries:
                # sanity check
                assert entry["identified_target"] == bridge_entity
                original_data_list.append(entry)
        else:
            continue

    for batch_start in tqdm(range(0, len(original_data_list), batch_size), desc="Processing batches"):
        batch_entries = original_data_list[batch_start : batch_start + batch_size]
        original_input_list = [entry["input_text"] for entry in batch_entries]
        target_text_list = [entry["target_text"] for entry in batch_entries]
        logging.debug(f"\noriginal_input: {original_input_list}")
        logging.debug(f"target_text: {target_text_list}")
        temp_dict = dict()

        real_h_r1_r2_t_list = [target.strip("><").split("><") for target in target_text_list]
        assert all([len(tokens) == 5 for tokens in real_h_r1_r2_t_list])

        # perturb the 1st relation
        filtered_original_input_list = []
        filtered_real_t_list = []
        query_list = []
        for i, entry in enumerate(batch_entries):
            bridge_entity = entry["identified_target"]
            real_tokens = real_h_r1_r2_t_list[i]
            original_input = entry["input_text"]
            if method == "original":
                # Select data with different combinations of (h, r1) for current bridge_entity from intervene_data
                candidate_entries = [
                    e for e in intervene_data[bridge_entity]
                    if e["input_text"].strip("><").split("><")[:2] != real_tokens[:2]
                ]
                if len(candidate_entries) != 0:
                    filtered_original_input_list.append(original_input)
                    filtered_real_t_list.append(real_tokens[3])
                    selected_intervene_data = candidate_entries[np.random.randint(0, len(candidate_entries))]
                    query_list.append(''.join([f"<{token}>" for token in selected_intervene_data["input_text"].strip("><").split("><")[:2]]))
                else:
                    skipped_data += 1
                    continue
                # if len(candidate_entries) == 0:
                #     query_list[i] = None
                #     if i in valid_q_indices:
                #         valid_q_indices.remove(i)
                # else:
                #     selected = candidate_entries[np.random.randint(0, len(candidate_entries))]
                #     q = ''.join([f"<{token}>" for token in selected["input_text"].strip("><").split("><")[:2]])
                #     query_list[i] = q
            elif method == "query":
                n1 = np.random.randint(0, 2000)
                n2 = np.random.randint(0, 2000)
                query_list[i] = f"<e_{n1}><e_{n2}>"
        
        logging.debug(f"query_list: {query_list}")
        if len(query_list) == 0:
            continue
        
        # 3. perform batch inference on original_input
        tokenizer_output = tokenizer(filtered_original_input_list, return_tensors="pt", padding=True)
        input_ids, attention_mask = tokenizer_output["input_ids"].to(device), tokenizer_output["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        all_hidden_states = outputs['hidden_states']
        rank_before = return_rank(all_hidden_states[8], word_embedding, tokenizer([f"<{target}>" for target in filtered_real_t_list])["input_ids"])[:, -1].tolist()
        temp_dict['rank_before'] = rank_before

        # batch_query_list = [query_list[i] for i in valid_q_indices]
        tokenizer_output = tokenizer(query_list, return_tensors="pt", padding=True)
        input_ids_, attention_mask_ = tokenizer_output["input_ids"].to(device), tokenizer_output["attention_mask"].to(device)
        with torch.no_grad():
            outputs_ctft = model(
                input_ids=input_ids_,
                attention_mask=attention_mask_,
                output_hidden_states=True
            )
        all_hidden_states_ctft = outputs_ctft["hidden_states"]

        for layer_to_intervene in range(1, 8):
            hidden_states = all_hidden_states[layer_to_intervene].clone()
            hidden_states_query_layer = all_hidden_states_ctft[layer_to_intervene]
            hidden_states[:, 1, :] = hidden_states_query_layer[:, 1, :]

            with torch.no_grad():
                for i in range(layer_to_intervene, 8):
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
            rank_after = return_rank(hidden_states, word_embedding, tokenizer([f"<{t}>" for t in filtered_real_t_list])["input_ids"])[:, -1].tolist()
            temp_dict['r1_'+str(layer_to_intervene)] = rank_after
        logging.debug(f"temp_dict: {temp_dict}")
        
        result_dict_list = [dict() for _ in range(len(filtered_original_input_list))]
        for key, value_list in temp_dict.items():
                for i in range(len(filtered_original_input_list)):
                    result_dict_list[i][key] = value_list[i]
        results.extend(result_dict_list)
    logging.info(f"skipped_data: {skipped_data}")
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
    # parser.add_argument("--layer_pos_pairs", required=True, help="List of (layer, position) tuples to evaluate")
    parser.add_argument("--step_list", default=None, nargs="+", help="checkpoint's steps to check causal strength")
    parser.add_argument("--device", default="cuda:0", help="Device to run the model on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for verbose output")

    args = parser.parse_args()
    setup_logging(args.debug)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args.model_dir.split("/")[-1] == "":
        dataset = args.model_dir.split("/")[-2].split("_")[0]
    else:
        dataset = args.model_dir.split("/")[-1].split("_")[0]
    
    # load all trained checkpoints
    all_checkpoints = [
        checkpoint for checkpoint in os.listdir(args.model_dir) 
        if checkpoint.startswith("checkpoint") and checkpoint.split("-")[-1] in args.step_list
    ]
    assert all(os.path.isdir(os.path.join(args.model_dir, checkpoint)) for checkpoint in all_checkpoints)
    all_checkpoints.sort(key=lambda var: int(var.split("-")[1]))
    print(all_checkpoints)
    
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    BATCH_SIZE = 4096
    
    results = dict()
    for checkpoint in all_checkpoints:
        result_ckpt = {}
        print("\nnow checkpoint", checkpoint)
        step = checkpoint.split("-")[-1]
        model_path = os.path.join(args.model_dir, checkpoint)
        model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        model.eval()
        
        # Todo : use layer_pos_pairs parameter
        # Load already deduplicated hidden representation results file
        with open(f"/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/{dataset}/(5,1)/{step}/id_train_dedup.json") as f:
            id_train_dedup = json.load(f)
        print(f"Total data number of id_train_dedup.json : {get_total_list_length(id_train_dedup)}")
        with open(f"/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/{dataset}/(5,1)/{step}/id_test_dedup.json") as f:
            id_test_dedup = json.load(f)
        print(f"Total data number of id_test_dedup.json : {get_total_list_length(id_test_dedup)}")
        with open(f"/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/{dataset}/(5,1)/{step}/ood_dedup.json") as f:
            ood_dedup = json.load(f)
        print(f"Total data number of ood_dedup.json : {get_total_list_length(ood_dedup)}")
        with open(f"/mnt/nas/jinho/GrokkedTransformer/collapse_analysis/{dataset}/(5,1)/{step}/nonsense_dedup.json") as f:
            nonsense_dedup = json.load(f)
        print(f"Total data number of nonsense_dedup.json : {get_total_list_length(nonsense_dedup)}")

        id_train_test_results = intervene_and_measure(id_train_dedup, id_test_dedup, model, tokenizer, device, batch_size=BATCH_SIZE)
        result_ckpt["train_inferred-test_inferred_id"] = id_train_test_results
        
        id_train_ood_results = intervene_and_measure(id_train_dedup, ood_dedup, model, tokenizer, device, batch_size=BATCH_SIZE)
        result_ckpt["train_inferred-test_inferred_ood"] = id_train_ood_results
        
        id_test_ood_results = intervene_and_measure(id_test_dedup, ood_dedup, model, tokenizer, device, batch_size=BATCH_SIZE)
        result_ckpt["test_inferred_id-test_inferred_ood"] = id_test_ood_results
        
        id_train_results = intervene_and_measure(id_train_dedup, id_train_dedup, model, tokenizer, device, batch_size=BATCH_SIZE)
        result_ckpt["train_inferred"] = id_train_results
        
        id_test_results = intervene_and_measure(id_test_dedup, id_test_dedup, model, tokenizer, device, batch_size=BATCH_SIZE)
        result_ckpt["test_inferred_id"] = id_test_results
        
        ood_results = intervene_and_measure(ood_dedup, ood_dedup, model, tokenizer, device, batch_size=BATCH_SIZE)
        result_ckpt["test_inferred_ood"] = ood_results
        
        results[checkpoint] = result_ckpt
    
    # save results for each data
    save_file_name = f"{args.model_dir.split('/')[-1]}_residual_same"
    with open(os.path.join(base_dir, "collapse_analysis", "tracing_results", f"{save_file_name}.json"), "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    refined_results = {}
    all_checkpoints = list(results.keys())
    all_checkpoints.sort(key=lambda var: int(var.split("-")[1]))
    
    for checkpoint in all_checkpoints:
        print("\nnow checkpoint", checkpoint)
        results_4_ckpt = {}
        for data_type, entries in tqdm(results[checkpoint].items()):
            result_4_type = {}
            result_4_type["total_num"] = len(entries)
            for i in range(1,8):
                result_4_type[f"r1_{i}"] = 0
            for entry in entries:
                for i in range(1,8):
                    if entry["rank_before"] != entry[f"r1_{i}"]:
                        result_4_type[f"r1_{i}"] += 1
            results_4_ckpt[data_type] = result_4_type
        refined_results[checkpoint] = results_4_ckpt

    with open(os.path.join(base_dir, "collapse_analysis", "tracing_results", f"{save_file_name}_refined.json"), "w", encoding='utf-8') as f:
        json.dump(refined_results, f, indent=4)
        
if __name__=="__main__":
    main()