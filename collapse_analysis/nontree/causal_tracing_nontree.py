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
    예: "<t_5><t_23><t_17><t_42></a>" -> ["t_5", "t_23", "t_17", "t_42"]
    """
    tokens = text.replace("</a>", "").strip("><").split("><")
    return tokens

def parse_atomic_fact(atomic_facts_f1, atomic_facts_f2):
    
    def _func_f1(atomic_facts):
        result = {}
        for item in atomic_facts:
            tokens_in = parse_tokens(item["input_text"])
            tokens_out = parse_tokens(item["target_text"])
            assert len(tokens_in) == 2 and len(tokens_out) == 3 and tokens_out[:2] == tokens_in
            result[(tokens_out[0], tokens_out[1])] = tokens_out[2]
        return result
    
    def _func_f2(atomic_facts):
        result = {}
        for item in atomic_facts:
            tokens_in = parse_tokens(item["input_text"])
            tokens_out = parse_tokens(item["target_text"])
            assert len(tokens_in) == 3 and len(tokens_out) == 4 and tokens_out[:3] == tokens_in
            result[(tokens_out[0], tokens_out[1], tokens_out[2])] = tokens_out[3]
        return result

    f1_dict = _func_f1(atomic_facts_f1)
    f2_dict = _func_f2(atomic_facts_f2)
    
    return f1_dict, f2_dict


def build_seen_atomic_dicts(f1_dict, f2_dict, train_data):
    '''
    f1_dict : (inp_token1, inp_token2) -> out_token
    f2_dict : (b1, t2, t3) -> t
    '''
    seen_f1_dict, seen_f2_dict = {}, {}
    for data in train_data:
        tokens = parse_tokens(data["target_text"])
        assert len(tokens) == 4
        t1, t2, t3, t = tokens
        b1 = f1_dict[(t1, t2)]
        seen_f1_dict[(t1, t2)] = b1
        assert f2_dict[(b1, t2, t3)] == t
        seen_f2_dict[(b1, t2, t3)] = t
    return seen_f1_dict, seen_f2_dict


def group_by_target(data, f1_dict = None, atomic_idx = 1):
    """
    data: train.json (또는 test.json)의 entry 리스트. 각 entry는 "input_text"와 "target_text"를 가짐.
    f1_dict : (inp_token1, inp_token2) -> out_token
    atomic_idx: 1이면 bridge entity로 grouping, 4나 5이면 (bridge entity, t2) pair로 grouping
    """
    grouped = {}
    for entry in data:
        tokens = parse_tokens(entry["target_text"])
        t1, t2, t3, t = tokens
        assert f1_dict != None
        b1 = f1_dict[(t1, t2)]
        if atomic_idx == 1:
            key = b1
        elif atomic_idx == 2:
            key = (b1, t2)
        elif atomic_idx in [4, 5]:
            key = f"{b1},{t2}"
        else:
            raise ValueError("atomic_idx must be 1, 4, or 5")
        grouped.setdefault(key, []).append(entry)
    return grouped


def deduplicate_grouped_data(grouped_data):
    """
    grouped_data: 그룹핑된 데이터. 형식은 { group_key: [entry, entry, ...] }이며,
                  각 entry는 "input_text"와 "target_text"를 포함하는 dict입니다.

    Returns:
        중복 제거된 entry들의 리스트. 동일한 (t1, t2)를 가진 entry들은 하나만 남게 됩니다.
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
                          atomic_idx = 1,
                          batch_size=32):
    """
    original_data: { group_key: [entry, ...], ... }
    model: Trained Decoder-only Transformer
    tokenizer: Tokenizer
    device: 실행 디바이스 (예: "cuda")
    atomic_idx: 1이면 bridge entity 기준 intervention, 
                4이면 (bridge entity, t2) pair 기준으로 t2만 바꿔서 b 유지하면서 t 바꾸기,
                5이면 (bridge entity, t2) pair 기준으로 t2만 바꿔서 b 바꾸면서 t 바꾸기
    batch_size: 미니배치 크기
    """
    results = []
    skipped_data = 0

    all_original_inputs = []
    all_real_t = []
    all_changed_t = []
    all_query = []
    all_injection_pos = []

    # 각 entry의 target_text는 [t1, t2, t3, t] 형태라고 가정
    for group_key, entries in original_data.items():
        for entry in entries:
            original_input = entry['input_text']
            tokens = parse_tokens(entry['target_text'])
            assert len(tokens) == 4
            t1, t2, t3, t = tokens

            # 후보 query 및 변경된 t 생성
            candidate_set = set()
            if atomic_idx == 1:
                # t2, t3는 유지하고 t1만 바꿔서 b1이 바뀌도록
                for b1_candidate in seen_b1_to_t1t2.keys():
                    if b1_candidate == group_key:  # group_key가 bridge entity
                        continue
                    
                    # f2 mapping: (b1_candidate, t2, t3) -> t가 있어야 함
                    if (b1_candidate, t2, t3) not in seen_f2_dict:
                        continue
                    
                    candidate_t = seen_f2_dict[(b1_candidate, t2, t3)]
                    if candidate_t != t:
                        # b1_candidate를 만드는 (t1, t2) pair 중 t2가 현재 t2와 같은 것만 선택
                        for (t1_candidate, t2_candidate) in seen_b1_to_t1t2[b1_candidate]:
                            if t2_candidate == t2 and t1_candidate != t1:
                                candidate_set.add((t1_candidate, t2, t3, candidate_t))
            elif atomic_idx == 4:
                b1, t2 = group_key.split(",")
                # t1과 t3는 유지하고 t2만 바꿔서 원래 b1을 유지하면서 t가 바뀌도록
                # 현재 b1을 만드는 (t1, t2) pair 중에서 t2가 다른 것들을 찾음
                for (t1_candidate, t2_candidate) in seen_b1_to_t1t2[b1]:
                    if t1_candidate == t1 and t2_candidate != t2:
                        # 새로운 t2로 f2 mapping이 있는지 확인
                        if (b1, t2_candidate, t3) in seen_f2_dict:
                            candidate_t = seen_f2_dict[(b1, t2_candidate, t3)]
                            if candidate_t != t:
                                candidate_set.add((t1, t2_candidate, t3, candidate_t))
            elif atomic_idx == 5:
                b1, t2 = group_key.split(",")
                # t1과 t3는 유지하고 t2만 바꿔서 b1이 바뀌도록
                for b1_candidate in seen_b1_to_t1t2.keys():
                    if b1_candidate == b1:  # 같은 b1은 건너뛰기
                        continue
                    
                    # b1_candidate를 만드는 (t1, t2) pair 중 t1이 현재 t1과 같은 것만 선택
                    for (t1_candidate, t2_candidate) in seen_b1_to_t1t2[b1_candidate]:
                        if t1_candidate == t1 and t2_candidate != t2:
                            # 새로운 b1과 t2로 f2 mapping이 있는지 확인
                            if (b1_candidate, t2_candidate, t3) in seen_f2_dict:
                                candidate_t = seen_f2_dict[(b1_candidate, t2_candidate, t3)]
                                if candidate_t != t:
                                    candidate_set.add((t1, t2_candidate, t3, candidate_t))
            else:
                raise ValueError("atomic_idx must be 1, 4, or 5")

            if len(candidate_set) == 0:
                skipped_data += 1
                continue

            candidate_list = sorted(list(candidate_set))
            selected_candidate = candidate_list[np.random.randint(0, len(candidate_list))]
            # query_text는 전체 입력: "<t1><t2><t3>"
            query_text = ''.join([f"<{token}>" for token in selected_candidate[:3]])
            changed_t = selected_candidate[-1]
            
            for injection_pos in range(3):  # 모든 입력 위치에 대해 injection
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

        # 불필요한 메모리 해제
        del outputs
        torch.cuda.empty_cache()

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

        # 불필요한 메모리 해제
        del outputs_query
        torch.cuda.empty_cache()

        # 3. intervention: 모든 position에 대해 수행
        intervened_ranks_real = {}
        intervened_ranks_changed = {}
        for layer_to_intervene in range(1, 8):
            hidden_states = original_hidden_states[layer_to_intervene].clone()
            
            # 각 샘플별로 지정된 injection_pos에 대해 intervention 수행
            for i, injection_pos in enumerate(current_injection_pos):
                hidden_states[i, injection_pos, :] = query_hidden_states[layer_to_intervene][i, injection_pos, :]
            
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

            # 불필요한 메모리 해제
            del intervened_hidden
            torch.cuda.empty_cache()

        # 4. 미니배치 내 각 샘플별 결과 dict 생성 및 결과 리스트에 저장
        for i in range(len(current_original_inputs)):
            result_dict = {}
            result_dict["injection_pos"] = current_injection_pos[i]  # injection 위치 정보 추가
            result_dict["rank_before_real_t"] = rank_before_real[i]
            result_dict["rank_before_changed_t"] = rank_before_changed[i]
            for layer in range(1, 8):
                result_dict[f"rank_after_{layer}_real_t"] = intervened_ranks_real[layer][i]
                result_dict[f"rank_after_{layer}_changed_t"] = intervened_ranks_changed[layer][i]
            results.append(result_dict)

        # 불필요한 메모리 해제
        del query_hidden_states
        torch.cuda.empty_cache()
        
    assert len(results) == num_samples

    return results



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to the model checkpoint")
    parser.add_argument("--step_list", default=None, nargs="+", help="checkpoint's steps to check causal strength")
    parser.add_argument("--data_dir", default=None, help="directory for dataset")
    parser.add_argument("--device", default="cuda", help="Device to run the model on")
    parser.add_argument("--atomic_idx", type=int, choices=[1, 2, 4, 5], required=True,
                         help="기준 atomic index: 1이면 f1, 2이면 f2, 4이면 f1 + x2 기준으로 b 유지하면서 x2 바꾸기, 5이면 f1 + x2 기준으로 b 바꾸면서 x2 바꾸기")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    setup_logging(args.debug)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
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
    
    # atomic facts 불러오기
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
    
    with open(os.path.join(data_dir, "data", dataset, "test.json"), "r") as f:
        test_data = json.load(f)
    
    test_id_data = [entry for entry in test_data if entry.get("type") == "type_0"]
    # test_ood_data = [entry for entry in test_data if entry.get("type") != "type_0"]

    original_train = group_by_target(train_data, f1_dict=f1_dict, atomic_idx=args.atomic_idx)
    original_test_id  = group_by_target(test_id_data, f1_dict=f1_dict, atomic_idx=args.atomic_idx)
    
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
                                              atomic_idx=args.atomic_idx,
                                              batch_size=args.batch_size)
        print("test data intervene exp...")
        test_results = intervene_and_measure(original_data=original_test_id_dedup, 
                                              model=model, 
                                              tokenizer=tokenizer, 
                                              seen_b1_to_t1t2=seen_b1_to_t1t2, 
                                              seen_f2_dict=seen_f2_dict,
                                              device=device,
                                              atomic_idx=args.atomic_idx,
                                              batch_size=args.batch_size)
        result_ckpt["train_inferred"] = train_results
        result_ckpt["test_inferred"] = test_results
        results[checkpoint] = result_ckpt
    
    save_file_name = f"{args.model_dir.split('/')[-1]}_residual_diff_f{args.atomic_idx}"
    out_dir = os.path.join(base_dir, "collapse_analysis", "tracing_results", "nontree")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{save_file_name}.json"), "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    # 후처리: 각 checkpoint별 intervention 전후 변화 요약
    logging.info(f"Summarizing Intervention Exp. result...")
    refined_results = {}
    all_ckpts = list(results.keys())
    all_ckpts.sort(key=lambda x: float('inf') if x == "final_checkpoint" else int(x.split("-")[1]))

    for checkpoint in all_ckpts:
        results_4_ckpt = {}
        for data_type, entries in results[checkpoint].items():
            # 각 position별로 결과를 저장
            results_by_pos = {0: {}, 1: {}, 2: {}}
            for pos in range(3):
                sample_num = 0
                result_4_type = dict()
                for i in range(1,8):
                    result_4_type[f"real_t>changed_t_layer{i}"] = 0
                    result_4_type[f"rank(changed_t)<=5_layer{i}"] = 0
                    result_4_type[f"both_layer{i}"] = 0
                
                for entry in entries:
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
                results_by_pos[pos] = result_4_type
            
            results_4_ckpt[data_type] = results_by_pos
        refined_results[checkpoint] = results_4_ckpt
    
    with open(os.path.join(out_dir, f"{save_file_name}_refined.json"), "w", encoding='utf-8') as f:
        json.dump(refined_results, f, indent=4)
        
if __name__=="__main__":
    main()