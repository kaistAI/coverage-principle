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


def intervene_and_measure(original_data, model, tokenizer, b2hr1_train, br2t_train_dict, device, batch_size=32):
    """
    original_data: { bridge_entity: [entry, ...], ... }
    model: Trained Decoder-only Transformer
    tokenizer: Tokenizer
    device: 실행 디바이스 (예: "cuda:0")
    batch_size: 모델 추론 시 사용할 미니배치 크기
    """
    results = []
    skipped_data = 0

    # 전체 데이터를 flatten하여 리스트로 저장
    all_original_inputs = []
    all_real_t = []
    all_changed_t = []
    all_query = []

    for bridge_entity, entries in original_data.items():
        for entry in entries:
            if entry['identified_target'] != bridge_entity:
                continue

            original_input = entry['input_text']
            target_text = entry['target_text']
            tokens = target_text.strip("><").split("><")
            if len(tokens) != 5:
                continue
            real_t = tokens[3]

            # 후보 query 및 변경된 t 생성 (첫 번째 relation perturb)
            candidate_set = set()
            for b_prime, r2t_dict in br2t_train_dict.items():
                if bridge_entity == b_prime or tokens[2] not in r2t_dict:
                    continue
                if tokens[3] != r2t_dict[tokens[2]]:
                    candidate_set.add((b_prime, r2t_dict[tokens[2]]))
            hrt_prime_candidate_set = set()
            for b_prime, t_prime in candidate_set:
                if b_prime not in b2hr1_train:
                    continue
                for h, r1 in b2hr1_train[b_prime]:
                    if h == tokens[0] and r1 != tokens[1]:
                        hrt_prime_candidate_set.add((h, r1, t_prime))
            if len(hrt_prime_candidate_set) == 0:
                skipped_data += 1
                continue

            hrt_prime_candidate_list = sorted(list(hrt_prime_candidate_set))
            selected_hrt_prime = hrt_prime_candidate_list[np.random.randint(0, len(hrt_prime_candidate_list))]
            query_text = ''.join([f"<{token}>" for token in selected_hrt_prime[:2]])
            changed_t = selected_hrt_prime[2]

            all_original_inputs.append(original_input)
            all_real_t.append(real_t)
            all_changed_t.append(changed_t)
            all_query.append(query_text)

    if len(all_original_inputs) == 0:
        print(f"skipped_data: {skipped_data}")
        return results

    num_samples = len(all_original_inputs)
    # 전체 데이터를 batch_size 단위로 처리
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        current_original_inputs = all_original_inputs[start:end]
        current_real_t = all_real_t[start:end]
        current_changed_t = all_changed_t[start:end]
        current_query = all_query[start:end]

        # 1. 원본 입력에 대해 모델 forward
        tokenizer_output = tokenizer(current_original_inputs, return_tensors="pt", padding=True)
        input_ids = tokenizer_output["input_ids"].to(device)
        attention_mask = tokenizer_output["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        original_hidden_states = outputs['hidden_states']  # 각 레이어의 hidden states 리스트
        word_embedding = model.lm_head.weight.data

        # target 토큰들 (real_t, changed_t) 배치 처리
        tokenized_real_t = tokenizer([f"<{t}>" for t in current_real_t], return_tensors="pt", padding=True)
        tokenized_changed_t = tokenizer([f"<{t}>" for t in current_changed_t], return_tensors="pt", padding=True)
        input_ids_real = tokenized_real_t["input_ids"]
        input_ids_changed = tokenized_changed_t["input_ids"]
        # input_ids_real = tokenized_real_t["input_ids"].to(device)
        # input_ids_changed = tokenized_changed_t["input_ids"].to(device)

        rank_before_real = return_rank(original_hidden_states[8], word_embedding, input_ids_real)[:, -1].tolist()
        rank_before_changed = return_rank(original_hidden_states[8], word_embedding, input_ids_changed)[:, -1].tolist()

        # 2. query 입력에 대해 모델 forward (intervention용)
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

        # 3. 각 layer (1~7)별로 intervention 수행 후 rank 계산
        intervened_ranks_real = {}
        intervened_ranks_changed = {}
        for layer_to_intervene in range(1, 8):
            # 원본 hidden state 복사 후 query의 hidden state (position 1)로 대체
            hidden_states = original_hidden_states[layer_to_intervene].clone()
            hidden_states[:, 1, :] = query_hidden_states[layer_to_intervene][:, 1, :]

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
            # 최종 layer norm
            intervened_hidden = model.transformer.ln_f(intervened_hidden)

            rank_after_real = return_rank(intervened_hidden, word_embedding, input_ids_real)[:, -1].tolist()
            rank_after_changed = return_rank(intervened_hidden, word_embedding, input_ids_changed)[:, -1].tolist()

            intervened_ranks_real[layer_to_intervene] = rank_after_real
            intervened_ranks_changed[layer_to_intervene] = rank_after_changed

        # 4. 미니배치 내 각 샘플별 결과 dict 생성 및 결과 리스트에 저장
        for i in range(len(current_original_inputs)):
            result_dict = {}
            result_dict["rank_before_real_t"] = rank_before_real[i]
            result_dict["rank_before_changed_t"] = rank_before_changed[i]
            for layer in range(1, 8):
                result_dict[f"r1_{layer}_real_t"] = intervened_ranks_real[layer][i]
                result_dict[f"r1_{layer}_changed_t"] = intervened_ranks_changed[layer][i]
            results.append(result_dict)

    print(f"skipped_data: {skipped_data}")
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
    
    # Make additional dictionary for data
    if "inf" in dataset:
        with open(os.path.join(base_dir, "data", dataset, "atomic_facts.json")) as f:
            atomic_items = json.load(f)
    else:
        with open(os.path.join(base_dir, "data", dataset, "train.json")) as f:
            atomic_items = json.load(f)

    all_atomic = set()     # (h,r,t)
    all_atomic_dict = dict()    # (h, r) -> t
    for item in tqdm(atomic_items):
        temp = item['target_text'].strip("><").split("><")
        # ignore inferred facts
        if len(temp) != 4:
            continue
        h, r, t = temp[:3]
        all_atomic_dict[(h,r)] = t
        all_atomic.add((h,r,t))

    with open(os.path.join(base_dir, "data", dataset, "train.json")) as f:
        train_data = json.load(f)

    b2hr1_train = dict()    # b -> (h, r1)
    br2t_train_dict = dict()  # b -> r2 -> t
    for item in tqdm(train_data):
        temp = item['target_text'].strip("><").split("><")
        # ignore atomic_ood
        if len(temp) == 4:
            continue
        h, r1, r2, t = temp[:4]
        b = all_atomic_dict[(h, r1)]
        assert all_atomic_dict[(b, r2)] == t
        if b not in b2hr1_train:
            b2hr1_train[b] = set()
        b2hr1_train[b].add((h, r1))
        if b not in br2t_train_dict:
            br2t_train_dict[b] = dict()
        br2t_train_dict[b][r2] = t

    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
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

        # id_train_test_results = intervene_and_measure(id_train_dedup, id_test_dedup, model, tokenizer, device)
        # result_ckpt["train_inferred-test_inferred_id"] = id_train_test_results
        
        # id_train_ood_results = intervene_and_measure(id_train_dedup, ood_dedup, model, tokenizer, device)
        # result_ckpt["train_inferred-test_inferred_ood"] = id_train_ood_results
        
        # id_test_ood_results = intervene_and_measure(id_test_dedup, ood_dedup, model, tokenizer, device)
        # result_ckpt["test_inferred_id-test_inferred_ood"] = id_test_ood_results
        
        id_train_results = intervene_and_measure(id_train_dedup, model, tokenizer, b2hr1_train, br2t_train_dict, device, batch_size=BATCH_SIZE)
        result_ckpt["train_inferred"] = id_train_results
        
        id_test_results = intervene_and_measure(id_test_dedup, model, tokenizer, b2hr1_train, br2t_train_dict, device, batch_size=BATCH_SIZE)
        result_ckpt["test_inferred_id"] = id_test_results
        
        # ood_results = intervene_and_measure(ood_dedup, ood_dedup, model, tokenizer, device)
        # result_ckpt["test_inferred_ood"] = ood_results
        
        results[checkpoint] = result_ckpt
    
    # save results for each data
    save_file_name = f"{args.model_dir.split('/')[-1]}_residual_diff"
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
                    if entry[f"r1_{i}_real_t"] < entry[f"r1_{i}_changed_t"]:
                        result_4_type[f"r1_{i}"] += 1
                    # if entry["rank_before"] != entry[f"r1_{i}"]:
                    #     result_4_type[f"r1_{i}"] += 1
            results_4_ckpt[data_type] = result_4_type
        refined_results[checkpoint] = results_4_ckpt

    with open(os.path.join(base_dir, "collapse_analysis", "tracing_results", f"{save_file_name}_refined.json"), "w", encoding='utf-8') as f:
        json.dump(refined_results, f, indent=4)
        
if __name__=="__main__":
    main()