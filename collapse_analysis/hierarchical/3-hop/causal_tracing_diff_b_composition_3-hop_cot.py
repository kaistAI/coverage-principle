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
    주어진 문자열을 "<"와 ">"를 기준으로 분리하여 토큰 리스트를 생성합니다.
    예: "<t_5><t_23><t_17><t_42><t_33></a>" -> ["t_5", "t_23", "t_17", "t_42", "t_33"]
    """
    tokens = text.replace("</a>", "").strip("><").split("><")
    return tokens

def parse_atomic_fact(atomic_facts_f1, atomic_facts_f2, atomic_facts_f3):
    
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
    f3_dict = _func(atomic_facts_f3)
    
    return f1_dict, f2_dict, f3_dict


def build_seen_atomic_dicts(f1_dict, f2_dict, f3_dict, train_data):
    seen_f1_dict, seen_f2_dict, seen_f3_dict = {}, {}, {}
    for data in train_data:
        tokens = parse_tokens(data["target_text"])
        assert len(tokens) == 7
        t1, t2, t3, t4, b1, b2, t = tokens
        assert f1_dict[(t1, t2)] == b1
        seen_f1_dict[(t1, t2)] = b1
        assert f2_dict[(b1, t3)] == b2
        seen_f2_dict[(b1, t3)] = b2
        assert f3_dict[(b2, t4)] == t
        seen_f3_dict[(b2, t4)] = t
    return seen_f1_dict, seen_f2_dict, seen_f3_dict


# atomic_idx에 따라 grouping: 
# - atomic_idx==1: group by f1 (identified_target = b1)
# - atomic_idx==2: group by f2 (identified_target = b2)
# - atomic_idx==3: group by f3 (identified_target = t_final)
def group_by_target(data, atomic_idx):
    """
    data: train.json (또는 test.json)의 entry 리스트. 각 entry는 "input_text"와 "target_text"를 가짐.
    atomic_idx:
    - 1이면 f1 기준 → 그룹키는 (h1, r1)에 대해 hr_to_b1로 계산한 b1.
    - 2이면 f2 기준 → 먼저 f1을 통해 b1을 구한 뒤, (b1, h3)를 통해 b2 (b1h3_to_b2 사전 사용).
    - 3이면 f3 기준 → 그룹키는 target_text의 다섯 번째 토큰 t.
    hr_to_b1: atomic_facts_f1를 이용해 (h1, r1) -> b1로 매핑한 dict.
    b1h3_to_b2: atomic_facts_f2_map와 동일한 (b1, h3) -> b2 매핑 dict.
    """
    grouped = {}
    for entry in data:
        tokens = parse_tokens(entry["target_text"])
        t1, t2, t3, t4, b1, b2, t = tokens
        if atomic_idx == 1:
            key = b1
        elif atomic_idx == 2:
            key = b2
        elif atomic_idx == 3:
            key = t
        else:
            continue
        grouped.setdefault(key, []).append(entry)
    return grouped


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
            elif atomic_idx == 2:
                dedup_key = tuple(tokens[:3])  # (t1, t2, t3)
            elif atomic_idx == 3:
                dedup_key = tuple(tokens[:4])  # (t1, t2, t3, t4)
            else:
                raise ValueError("atomic_idx must be 1, 2, or 3")

            if dedup_key not in deduped:
                deduped[dedup_key] = entry
        output[group_key] = list(deduped.values())

    return output


def intervene_and_measure(original_data, 
                          model, 
                          tokenizer, 
                          atomic_idx, 
                          seen_b1_to_t1t2, 
                          seen_f2_dict, 
                          seen_f3_dict,
                          extra_dict, device, batch_size=32):
    """
    original_data: dict, 그룹화 기준은 atomic_idx에 따라 달라짐.
      - atomic_idx==1: key는 f1의 출력(b1)
      - atomic_idx==2: key는 f2의 출력(b2)
      - atomic_idx==3: key는 최종 target t
    model, tokenizer, device: 모델 관련 객체
    seen_b1_to_t1t2: b1 -> set((t1, t2), ...)
    seen_f2_dict: (b1, t3) -> b2
    seen_f3_dict: (b2, t4) -> t
    extra_dict: 
      - if atomic_idx==2: seen_b2_to_b1t3 (b2 -> set((b1, h3)) )
      - if atomic_idx==3: seen_t4_to_b2t (t4 -> set((b2, t)) )
      - if atomic_idx==1: None
    batch_size: 미니배치 크기
    """
    results = []
    skipped_data = 0

    all_original_inputs = []
    all_real_t = []
    all_changed_t = []
    all_query = []

    # 각 entry의 target_text는 [h1, r1, h3, h4, t] 형태라고 가정
    for bridge_entity, entries in original_data.items():
        for entry in entries:
            original_input = entry['input_text']
            tokens = parse_tokens(entry['target_text'])
            assert len(tokens)== 5
            t1, t2, t3, t4, t = tokens

            # atomic_idx에 따라 후보 선택 로직 분기
            if atomic_idx == 1:
                # f1 기반: base_entity = b1
                candidate_set = set()
                for b1_candidate in seen_b1_to_t1t2.keys():
                    if b1_candidate == bridge_entity:
                        continue
                    # f2 mapping: (b1_prime, h3) -> b2가 있어야 함
                    if (b1_candidate, t3) not in seen_f2_dict:
                        continue
                    b2_candidate = seen_f2_dict[(b1_candidate, t3)]
                    # f3 mapping: (candidate_b2, h4) -> candidate_t가 있어야 함
                    if (b2_candidate, t4) not in seen_f3_dict:
                        continue
                    candidate_t = seen_f3_dict[(b2_candidate, t4)]
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
                # query_text는 f1의 입력: "<t1><t2>"
                query_text = ''.join([f"<{token}>" for token in selected_candidate[:2]])
                changed_t = selected_candidate[-1]
                injection_pos = 1  # intervention 시, 두 번째 토큰 위치 교체

            elif atomic_idx == 2:
                # extra_dict: seen_b2_to_b1t3, mapping b2 -> set((b1, h3))
                candidate_set = set()
                for b2_candidate in extra_dict.keys():
                    if b2_candidate == bridge_entity:
                        continue
                    if (b2_candidate, t4) not in seen_f3_dict:
                        continue
                    candidate_t = seen_f3_dict[(b2_candidate, t4)]
                    if candidate_t != t:
                        candidate_set.add((b2_candidate, candidate_t))
                candidate_input_set = set()
                # 후보로 얻은 b2_prime에서, extra_dict[b2_prime]는 set((b1, h3))
                for b2_candidate, cand_t in candidate_set:
                    for (b1_candidate, t3_candidate) in extra_dict[b2_candidate]:
                        for (t1_candidate, t2_candidate) in seen_b1_to_t1t2[b1_candidate]:
                            if t1_candidate == t1 and t2_candidate == t2 and t3_candidate != t3:
                                candidate_input_set.add((t1_candidate, t2_candidate, t3_candidate, cand_t))
                if len(candidate_input_set) == 0:
                    skipped_data += 1
                    continue
                candidate_list = sorted(list(candidate_input_set))
                selected_candidate = candidate_list[np.random.randint(0, len(candidate_list))]
                # query_text는 f2의 입력: "<t1><t2><t3>"
                query_text = ''.join([f"<{token}>" for token in selected_candidate[:3]])
                changed_t = selected_candidate[-1]
                injection_pos = 2  # intervention 시, 세 번째 토큰 위치 교체

            # elif atomic_idx == 3:
            #     # f3 기반: base_entity = t_final (최종 target)
            #     # extra_dict: h4_to_candidates, mapping h4 -> list of (b2, candidate_t)
            #     candidate_set = set()
            #     for (b2_candidate, cand_t) in extra_dict[t4]:
            #         if cand_t != t:
            #             candidate_set.add((b2_candidate, cand_t))
            #     if len(candidate_set) == 0:
            #         skipped_data += 1
            #         continue
            #     candidate_list = sorted(list(candidate_set))
            #     selected_candidate = candidate_list[np.random.randint(0, len(candidate_list))]
            #     # query_text는 f3의 입력: "<t1><t2><t3><t4>"
            #     query_text = ''.join([f"<{token}>" for token in [selected_candidate[0], t4]])
            #     changed_t = selected_candidate[1]
            #     injection_pos = 3  # intervention 시, 네 번째 토큰 위치 교체
            else:
                raise ValueError("atomic_idx must be 1 or 2")
            
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

        # 1. 원본 입력에 대한 모델 forward
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

        # target token 처리 (실제 target, 변경된 target)
        tokenized_real_t = tokenizer([f"<{t}>" for t in current_real_t], return_tensors="pt", padding=True)
        tokenized_changed_t = tokenizer([f"<{t}>" for t in current_changed_t], return_tensors="pt", padding=True)
        input_ids_real = tokenized_real_t["input_ids"]
        input_ids_changed = tokenized_changed_t["input_ids"]

        rank_before_real = return_rank(original_hidden_states[-1], word_embedding, input_ids_real)[:, -1].tolist()
        rank_before_changed = return_rank(original_hidden_states[-1], word_embedding, input_ids_changed)[:, -1].tolist()

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

        # 3. intervention: injection할 토큰 위치는 injection_pos (atomic_idx별로 다름)
        intervened_ranks_real = {}
        intervened_ranks_changed = {}
        for layer_to_intervene in range(1, 8):
            hidden_states = original_hidden_states[layer_to_intervene].clone()
            hidden_states[:, injection_pos, :] = query_hidden_states[layer_to_intervene][:, injection_pos, :]
            intervened_hidden = hidden_states
            for i in range(layer_to_intervene, 8):
                f_layer = model.transformer.h[i]
                residual = intervened_hidden
                intervened_hidden = f_layer.ln_1(intervened_hidden)
                attn_output = f_layer.attn(intervened_hidden)[0]
                intervened_hidden = attn_output + residual
                residual = intervened_hidden
                intervened_hidden = f_layer.ln_2(intervened_hidden)
                feed_forward_hidden = f_layer.mlp.c_proj(f_layer.mlp.act(f_layer.mlp.c_fc(intervened_hidden)))
                intervened_hidden = residual + feed_forward_hidden
            intervened_hidden = model.transformer.ln_f(intervened_hidden)

            rank_after_real = return_rank(intervened_hidden, word_embedding, input_ids_real)[:, -1].tolist()
            rank_after_changed = return_rank(intervened_hidden, word_embedding, input_ids_changed)[:, -1].tolist()

            intervened_ranks_real[layer_to_intervene] = rank_after_real
            intervened_ranks_changed[layer_to_intervene] = rank_after_changed

        for i in range(len(current_original_inputs)):
            result_dict = {}
            result_dict["rank_before_real_t"] = rank_before_real[i]
            result_dict["rank_before_changed_t"] = rank_before_changed[i]
            for layer in range(1, 8):
                result_dict[f"rank_after_{layer}_real_t"] = intervened_ranks_real[layer][i]
                result_dict[f"rank_after_{layer}_changed_t"] = intervened_ranks_changed[layer][i]
            results.append(result_dict)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to the model checkpoint")
    parser.add_argument("--step_list", default=None, nargs="+", help="Checkpoint steps to evaluate")
    parser.add_argument("--device", default="cuda:0", help="Device to run the model on")
    parser.add_argument("--atomic_idx", type=int, choices=[1, 2], required=True,
                        help="기준 atomic index: 1이면 f1, 2이면 f2, 3이면 f3 기준 intervention")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    setup_logging(args.debug)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    dataset = args.model_dir.split("/")[-1].split("_")[0]
    
    logging.info(f"base_dir: {base_dir}")
    logging.info(f"dataset: {dataset}")
    
    # atomic facts 불러오기
    with open(os.path.join(base_dir, "data", dataset, "atomic_facts_f1.json"), "r") as f:
        atomic_facts_f1 = json.load(f)
    with open(os.path.join(base_dir, "data", dataset, "atomic_facts_f2.json"), "r") as f:
        atomic_facts_f2 = json.load(f)
    with open(os.path.join(base_dir, "data", dataset, "atomic_facts_f3.json"), "r") as f:
        atomic_facts_f3 = json.load(f)

    # (inp_token1, inp_token2) -> out_token 
    f1_dict, f2_dict, f3_dict = parse_atomic_fact(atomic_facts_f1, atomic_facts_f2, atomic_facts_f3)
    
    # Consisting only of atomic facts shown during training
    with open(os.path.join(base_dir, "data", dataset, "train.json"), "r") as f:
        train_data = json.load(f)
    seen_f1_dict, seen_f2_dict, seen_f3_dict = build_seen_atomic_dicts(f1_dict, f2_dict, f3_dict, train_data)
    
    # for convenience
    # b1 -> set((t1, t2))
    seen_b1_to_t1t2 = {}
    for (t1, t2), b1 in seen_f1_dict.items():
        seen_b1_to_t1t2.setdefault(b1, set()).add((t1, t2))
    
    # b2 -> set((b1, t3))
    seen_b2_to_b1t3 = {}
    for (b1, t3), b2 in seen_f2_dict.items():
        seen_b2_to_b1t3.setdefault(b2, set()).add((b1, t3))
    
    # seen_t4_to_b2t = {}
    # for (b2, t4), t in seen_f3_dict.items():
    #     seen_t4_to_b2t.setdefault(t4, set()).add((b2, t))
    
    with open(os.path.join(base_dir, "data", dataset, "test.json"), "r") as f:
        test_data = json.load(f)
    
    test_id_data = [entry for entry in test_data if entry.get("type") == "type_0"]
    # test_ood_data = [entry for entry in test_data if entry.get("type") != "type_0"]

    if args.atomic_idx == 1:
        original_train = group_by_target(train_data, atomic_idx=1)
        original_test_id  = group_by_target(test_id_data, atomic_idx=1)
        extra_dict = None  # f1-based는 별도 extra dict 필요 없음
    elif args.atomic_idx == 2:
        original_train = group_by_target(train_data, atomic_idx=2)
        original_test_id  = group_by_target(test_id_data, atomic_idx=2)
        extra_dict = seen_b2_to_b1t3  # f2-based: extra dict
    # elif args.atomic_idx == 3:
    #     original_train = group_by_target(train_data, atomic_idx=3)
    #     original_test_id  = group_by_target(test_id_data, atomic_idx=3)
    #     extra_dict = seen_t4_to_b2t  # f3-based: extra dict
    
    # deduplicate data according to atomic_idx
    original_train_dedup = deduplicate_grouped_data(original_train, args.atomic_idx)
    original_test_id_dedup = deduplicate_grouped_data(original_test_id, args.atomic_idx)

    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    all_checkpoints = [
        checkpoint for checkpoint in os.listdir(args.model_dir)
        if checkpoint.startswith("checkpoint") and checkpoint.split("-")[-1] in args.step_list
    ]
    all_checkpoints.sort(key=lambda var: int(var.split("-")[1]))
    results = {}
    
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
        
        print("train data intervene exp...")
        train_results = intervene_and_measure(original_data=original_train_dedup, 
                                              model=model, 
                                              tokenizer=tokenizer, 
                                              atomic_idx=args.atomic_idx,
                                              seen_b1_to_t1t2=seen_b1_to_t1t2, 
                                              seen_f2_dict=seen_f2_dict, 
                                              seen_f3_dict=seen_f3_dict,
                                              extra_dict=extra_dict, 
                                              device=device, 
                                              batch_size=args.batch_size)
        print("test data intervene exp...")
        test_results = intervene_and_measure(original_data=original_test_id_dedup, 
                                             model=model, 
                                             tokenizer=tokenizer, 
                                             atomic_idx=args.atomic_idx,
                                             seen_b1_to_t1t2=seen_b1_to_t1t2, 
                                             seen_f2_dict=seen_f2_dict, 
                                             seen_f3_dict=seen_f3_dict,
                                             extra_dict=extra_dict, 
                                             device=device, 
                                             batch_size=args.batch_size)
        result_ckpt["train_inferred"] = train_results
        result_ckpt["test_inferred"] = test_results
        results[checkpoint] = result_ckpt
    
    save_file_name = f"{args.model_dir.split('/')[-1]}_residual_diff_f{args.atomic_idx}"
    out_dir = os.path.join(base_dir, "collapse_analysis", "tracing_results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{save_file_name}.json"), "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    # 후처리: 각 checkpoint별 intervention 전후 변화 요약
    logging.info(f"Summarizing Intervention Exp. result...")
    refined_results = {}
    all_ckpts = list(results.keys())
    all_ckpts.sort(key=lambda var: int(var.split("-")[1]))
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
                # 애초부터 실제 t가 changed t보다 낮은 경우는 제외 (intervene 방식의 의미가 없음)
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
