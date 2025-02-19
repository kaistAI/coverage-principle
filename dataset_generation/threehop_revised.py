import argparse
import numpy as np
import random
from collections import defaultdict
import logging
import itertools
from typing import List
import multiprocessing as mp
from tqdm import tqdm
import os
import json
from functools import partial


def setup_logging(debug_mode):
    level = logging.DEBUG if debug_mode else logging.INFO
    # logging.basicConfig(filename="a.log", level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=level, format="%(levelname)s - %(message)s")
    

def make_arbitrary_function(domain: List, codomain: List, seen_ratio: int):
    result = dict()
    range = random.choices(codomain, k=len(domain))
    for operand, res in zip(domain, range):
        result[operand] = res
    
    keys = list(result.keys())
    random.shuffle(keys)
    cutoff = int(round(seen_ratio * len(keys)))
    seen_expected_inputs, unseen_inputs = keys[:cutoff], keys[cutoff:]
    
    seen_expected_data_dict = dict()
    for input in seen_expected_inputs:
        seen_expected_data_dict[input] = result[input]
    unseen_data_dict = dict()
    for input in unseen_inputs:
        unseen_data_dict[input] = result[input]
    
    return result, seen_expected_data_dict, unseen_data_dict


def form_item(input_tokens, output_token):
    """
    Create a dict with:
      input_text = concatenation of input_tokens
      target_text = input_text + output_token + '</a>'
    """
    inp = "".join(input_tokens)
    tgt = inp + output_token + "</a>"
    return {"input_text": inp, "target_text": tgt}

def process_item_f1(item, f2_index, f3_index):
    (h1, h2), b1 = item
    partial_set = set()
    assert b1 in f2_index, f"b1 {b1} not in f2_index"
    for h3, b2 in f2_index[b1]:
        assert b2 in f3_index, f"b2 {b2} not in f3_index"
        for h4, t in f3_index[b2]:
            partial_set.add((h1, h2, h3, h4, t))
    return partial_set

def process_item_S_f1(item, S_f2_index, S_f3_index):
    (h1, h2), b1 = item
    partial_set = set()
    assert b1 in S_f2_index, f"b1 {b1} not in S_f2_index"
    for h3, b2 in S_f2_index[b1]:
        assert b2 in S_f3_index, f"b2 {b2} not in S_f3_index"
        for h4, t in S_f3_index[b2]:
            partial_set.add((h1, h2, h3, h4, t))
    return partial_set


def coverage_type(sc1, sc2, sc3):
    bits = (sc1<<2) + (sc2<<1) + sc3
    # bits in [0..7], 7 means (1,1,1).
    if bits==7:
        return 0
    else:
        return bits+1  # map [0..6] => [1..7]
    

def choose(arr, ratio_or_count):
    if type(ratio_or_count) == float:
        num = round(ratio_or_count*len(arr))
    elif type(ratio_or_count) == int:
        num = ratio_or_count
    else:
         assert False
    if num >= len(arr):
        return arr
    rand_inds = np.random.choice(len(arr), num, replace=False).tolist()
    return [arr[i] for i in rand_inds]


def parse_3hop_input(input_text):
    """
    e.g. "<t_7><t_11><t_99><t_25>" => [7, 11, 99, 25]
    returns None if not 4 tokens
    """
    tokens = input_text.strip("<>").split("><")
    tokens = [int(token.split("_")[-1]) for token in tokens]
    if len(tokens) != 4:
        return None
    return tokens


def group_data_by_b1(examples, f1_dict):
    """
    For each example: parse (h1,h2,h3,h4),
    b1= f1_dict.get((h1,h2),'unknown'),
    group by b1
    => group_dict[b1] = list of examples
    """
    group_dict = defaultdict(list)
    for ex in examples:
        inp_tokens = parse_3hop_input(ex["input_text"])
        if not inp_tokens:
            continue
        h1,h2,h3,h4 = inp_tokens
        b1 = f1_dict.get((h1,h2), "unknown")
        group_dict[b1].append(ex)
    return dict(group_dict)


def group_data_by_b2(examples, f1_dict, f2_dict):
    """
    For each example: parse (h1,h2,h3,h4),
    b1= f1_dict.get((h1,h2),'unknown'),
    b2= f2_dict.get((b1,h3),'unknown') if b1 !='unknown'
    group by b2 => group_dict[b2] = list of ex
    """
    group_dict = defaultdict(list)
    for ex in examples:
        inp_tokens = parse_3hop_input(ex["input_text"])
        if not inp_tokens:
            continue
        h1,h2,h3,h4 = inp_tokens
        b1 = f1_dict.get((h1,h2), "unknown")
        if b1 == "unknown":
            b2 = "unknown"
        else:
            b2 = f2_dict.get((b1,h3), "unknown")
        group_dict[b2].append(ex)
    return dict(group_dict)


def group_data_by_t(examples, f1_dict, f2_dict, f3_dict):
    """
    For each example: parse (h1,h2,h3,h4),
    b1= f1_dict.get((h1,h2),'unknown'),
    b2= f2_dict.get((b1,h3),'unknown') if b1 !='unknown'
    group by b2 => group_dict[b2] = list of ex
    """
    group_dict = defaultdict(list)
    for ex in examples:
        inp_tokens = parse_3hop_input(ex["input_text"])
        if not inp_tokens:
            continue
        h1,h2,h3,h4 = inp_tokens
        b1 = f1_dict.get((h1,h2), "unknown")
        if b1 == "unknown":
            b2 = "unknown"
        else:
            b2 = f2_dict.get((b1,h3), "unknown")
            
        if b2 == "unknown":
            t ="unknown"
        else:
            t = f3_dict.get((b2,h4), "unknown")
        group_dict[t].append(ex)
    return dict(group_dict)


def load_and_preprocess_data(f1_dict, f2_dict, f3_dict, test_data, idx):
    """
    Original approach from parallel version:
    - We parse train_path => 'atomic_facts_N.json' style
    - Filter for 2-entity input_text
    - Build a lookup => e.g. train_lookup[input_text] = ...
    - Then group test data by that bridging entity
    """
    # with open(train_path, 'r') as f:
    #     train_data = json.load(f)

    id_train_data = []
    id_test_data = []
    ood_test_data = []
    
    for d in test_data:
        if d['type'] == 'train_inferred':
            id_train_data.append(d)
        elif d['type'] == 'type_0':
            id_test_data.append(d)
        else:
            ood_test_data.append(d)
            
    if idx==1:
        grouped_id_train_data = group_data_by_b1(id_train_data, f1_dict)
        grouped_id_test_data = group_data_by_b1(id_test_data, f1_dict)
        grouped_ood_test_data = group_data_by_b1(ood_test_data, f1_dict)
    
    elif idx==2:
        grouped_id_train_data = group_data_by_b2(id_train_data, f1_dict, f2_dict)
        grouped_id_test_data = group_data_by_b2(id_test_data, f1_dict, f2_dict)
        grouped_ood_test_data = group_data_by_b2(ood_test_data, f1_dict, f2_dict)
        
    elif idx==3:
        grouped_id_train_data = group_data_by_t(id_train_data, f1_dict, f2_dict, f3_dict)
        grouped_id_test_data = group_data_by_t(id_test_data, f1_dict, f2_dict, f3_dict)
        grouped_ood_test_data = group_data_by_t(ood_test_data, f1_dict, f2_dict, f3_dict)
    else:
        raise NotImplementedError
    
    return grouped_id_train_data, grouped_id_test_data, grouped_ood_test_data


def deduplicate_vectors(grouped_results, idx):
    dedup_stats = defaultdict(lambda: defaultdict(int))
    # deduplicated_results = defaultdict(lambda: defaultdict(list))
    deduplicated_results = defaultdict(lambda: defaultdict(list))
    seen_datas = defaultdict(lambda: defaultdict(set))
    
    for bridge, instances in grouped_results.items():
        # logging.info(f"Performing dedup for bridge {bridge}")
        if bridge == 'unknown':
            continue
        for instance in instances:
            ctype = instance["type"]
            tokens = parse_3hop_input(instance["input_text"])
            if idx == 1:
                if (tokens[0], tokens[1]) in seen_datas[bridge][ctype]:
                    dedup_stats[bridge][ctype] += 1
                else:
                    seen_datas[bridge][ctype].add((tokens[0], tokens[1]))
                    deduplicated_results[ctype][bridge].append(instance)
            elif idx == 2:
                if (tokens[0], tokens[1], tokens[2]) in seen_datas[bridge][ctype]:
                    dedup_stats[bridge][ctype] += 1
                else:
                    seen_datas[bridge][ctype].add((tokens[0], tokens[1], tokens[2]))
                    deduplicated_results[ctype][bridge].append(instance)
            elif idx == 3:
                if (tokens[0], tokens[1], tokens[2], tokens[3]) in seen_datas[bridge][ctype]:
                    dedup_stats[bridge][ctype] += 1
                else:
                    seen_datas[bridge][ctype].add((tokens[0], tokens[1], tokens[2], tokens[3]))
                    deduplicated_results[ctype][bridge].append(instance)
            else:
                raise NotImplementedError
    return dict(deduplicated_results), dedup_stats

def main():
    parser= argparse.ArgumentParser()
    parser.add_argument("--num_tokens", required=True, type=int, help="the number of total tokens")
    parser.add_argument("--default_seen_ratio", default=0.7, type=float, help="the ratio of ")
    parser.add_argument("--max_train_data_num", default=382000, type=int, help="the total data of train_inferred")
    parser.add_argument("--test_size_for_type", default=3000, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", default=42, type=int, help="for controlling randomness")
    
    args = parser.parse_args()
    setup_logging(args.debug)
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Set Total tokens, it will be saved as vocab.json
    vocab = [f"<t_{i}>" for i in range(args.num_tokens)]
    vocab = vocab + ["</a>"]
    assert len(vocab) == len(set(vocab))
    logging.debug(f"total vocab: {vocab}")
    
    # Make the total information of all functions
    domain: List = list(itertools.product(*([set([i for i in range(args.num_tokens)]), set([i for i in range(args.num_tokens)])])))     # (NUM_TOKENS X NUM_TOKENS)
    codomain: List = list(set([i for i in range(args.num_tokens)]))
    logging.debug(f"domain of f1: {domain}")
    logging.debug(f"codomain of f2: {codomain}")
    f1_dict, S_f1, OOD_f1 = make_arbitrary_function(domain, codomain, args.default_seen_ratio)
    logging.debug(f"The data expected to be seen for f1 : {len(S_f1)}")
    logging.debug(f"The data not be seen for f1 : {len(OOD_f1)}")

    f2_dict, S_f2, OOD_f2 = make_arbitrary_function(domain, codomain, args.default_seen_ratio)
    logging.debug(f"The data expected to be seen for f2 : {len(S_f2)}")
    logging.debug(f"The data not be seen for f2 : {len(OOD_f2)}")

    f3_dict, S_f3, OOD_f3 = make_arbitrary_function(domain, codomain, args.default_seen_ratio)
    logging.debug(f"The data expected to be seen for f3 : {len(S_f3)}")
    logging.debug(f"The data not be seen for f3 : {len(OOD_f3)}")
    
    # Make all possible inferred facts using f_dict and expected to be seen inferred facts using S_f
    f2_index = {}
    for (b1, h3), b2 in f2_dict.items():
        f2_index.setdefault(b1, []).append((h3, b2))

    f3_index = {}
    for (b2, h4), t in f3_dict.items():
        f3_index.setdefault(b2, []).append((h4, t))

    with mp.Pool(processes=round(mp.cpu_count() * 0.9)) as pool:
        process_f1 = partial(process_item_f1, f2_index=f2_index, f3_index=f3_index)
        results = list(tqdm(pool.imap(process_f1, list(f1_dict.items())), total=len(f1_dict), 
                            desc="Processing all possible inferred facts based on f1, f2, and f3"))

    all_possible_inferred_idx = set().union(*results)
    logging.info(f"len(all_inferred_facts): {len(all_possible_inferred_idx)}")

    S_f2_index = {}
    for (b1, h3), b2 in S_f2.items():
        S_f2_index.setdefault(b1, []).append((h3, b2))
    
    S_f3_index = {}
    for (b2, h4), t in S_f3.items():
        S_f3_index.setdefault(b2, []).append((h4, t))
    
    with mp.Pool(processes=round(mp.cpu_count() * 0.9)) as pool:
        process_S_f1 = partial(process_item_S_f1, S_f2_index=S_f2_index, S_f3_index=S_f3_index)
        results2 = list(tqdm(pool.imap(process_S_f1, list(S_f1.items())), total=len(S_f1),
                             desc="Processing inferred facts based on S_f1, S_f2, S_f3 expected to be shown during training"))
    seen_expected_inferred_idx = set().union(*results2)
    logging.info(f"len(seen_expected_inferred_facts): {len(seen_expected_inferred_idx)}")
    
    # Sample the part of seen_expected_inferred_fact
    if len(seen_expected_inferred_idx) <= args.max_train_data_num:
        raise Exception("\n\n\n###############################################################\nAll covered data is in train_inferred since MAX_TRAIN_DATA_NUM is too large.\nPlease make the value smaller\n###############################################################\n\n\n")
    else:
        train_inferred_idx: List = random.sample(list(seen_expected_inferred_idx), args.max_train_data_num)
        # does not include all cases that can be made with S_f1, S_f2, and S_f3, so the actual data shown may be different and needs to be redefined.
        S_f1, S_f2, S_f3 = dict(), dict(), dict()
        for seen_h1, seen_h2, seen_h3, seen_h4, seen_t in tqdm(train_inferred_idx):
            seen_b1 = f1_dict[(seen_h1, seen_h2)]
            S_f1[(seen_h1, seen_h2)] = seen_b1
            
            seen_b2 = f2_dict[(seen_b1, seen_h3)]
            S_f2[(seen_b1, seen_h3)] = seen_b2
            S_f3[(seen_b2, seen_h4)] = seen_t
        
        OOD_f1 = {key: value for key, value in f1_dict.items() if key not in S_f1}
        OOD_f2 = {key: value for key, value in f1_dict.items() if key not in S_f2}
        OOD_f3 = {key: value for key, value in f1_dict.items() if key not in S_f3}
        
        assert not (set(S_f1.items()) & set(OOD_f1.items()))
        assert not (set(S_f2.items()) & set(OOD_f2.items()))
        assert not (set(S_f3.items()) & set(OOD_f3.items()))

    train_inferred_idx_set = set(train_inferred_idx)

    logging.info(f"real_seen_f1_set len: {len(S_f1)}")
    logging.info(f"real_seen_f2_set len: {len(S_f2)}")
    logging.info(f"real_seen_f3_set len: {len(S_f3)}")

    logging.info(f"real_not_seen_f1_set len: {len(OOD_f1)}")
    logging.info(f"real_not_seen_f2_set len: {len(OOD_f2)}")
    logging.info(f"real_not_seen_f3_set len: {len(OOD_f3)}")
    
    # Make ID_train & ID_test
    train_inferred = []
    for (h1_idx, h2_idx, h3_idx, h4_idx, t_idx) in train_inferred_idx_set:
        inp_tokens = [vocab[h1_idx], vocab[h2_idx], vocab[h3_idx], vocab[h4_idx]]
        out_token = vocab[t_idx]
        train_inferred.append(form_item(inp_tokens, out_token))
        
    logging.info(f"train_inferred:\n    example: {train_inferred[:10]}\n    len: {len(train_inferred)}")
    
    covered_inferred = []
    non_covered_inferred_dict = dict()

    for (h1_idx, h2_idx, h3_idx, h4_idx, t_idx) in tqdm(all_possible_inferred_idx, desc="Classify data by coverage type"):
        if (h1_idx, h2_idx) in S_f1:
            sc1 = True
        else: sc1 = False
        matched_b1_idx = f1_dict[(h1_idx, h2_idx)]
        if (matched_b1_idx, h3_idx) in S_f2:
            sc2 = True
        else: sc2 = False
        matched_b2_idx = f2_dict[(matched_b1_idx, h3_idx)]
        if (matched_b2_idx, h4_idx) in S_f3:
            sc3 = True
        else:
            sc3 = False

        c_type = coverage_type(sc1, sc2, sc3)
        if c_type == 0:
            if not (h1_idx, h2_idx, h3_idx, h4_idx, t_idx) in train_inferred_idx_set:
                item = form_item([vocab[h1_idx], vocab[h2_idx], vocab[h3_idx], vocab[h4_idx]], vocab[t_idx])
                item["type"] = f"type_{c_type}"
                covered_inferred.append(item)
        else:
            item = form_item([vocab[h1_idx], vocab[h2_idx], vocab[h3_idx], vocab[h4_idx]], vocab[t_idx])
            item["type"] = f"type_{c_type}"
            if f"type_{c_type}" not in non_covered_inferred_dict:
                non_covered_inferred_dict[f"type_{c_type}"] = []
            non_covered_inferred_dict[f"type_{c_type}"].append(item)
    
    # Make test data
    test_data = []
    sampled_train_inferred_data = choose(train_inferred, args.test_size_for_type)
    for item in sampled_train_inferred_data:
        item["type"] = "train_inferred"
        test_data.append(item)
    test_data.extend(choose(covered_inferred, args.test_size_for_type))
    for key, value in non_covered_inferred_dict.items():
        test_data.extend(choose(value, args.test_size_for_type))
        
    logging.info(f"test_data len: {len(test_data)}")
    
    # shuffle final sets
    random.shuffle(train_inferred)
    random.shuffle(test_data)

    # build atomic facts for analysis
    atomic_facts_1 = []
    for (h1_idx, h2_idx), b1_idx in f1_dict.items():
        item = form_item([vocab[h1_idx], vocab[h2_idx]], vocab[b1_idx])
        atomic_facts_1.append(item)

    atomic_facts_2 = []
    for (b1_idx, h3_idx), b2_idx in f2_dict.items():
        item = form_item([vocab[b1_idx], vocab[h3_idx]], vocab[b2_idx])
        atomic_facts_2.append(item)

    atomic_facts_3 = []
    for (b2_idx, h4_idx), t_idx in f3_dict.items():
        item = form_item([vocab[b2_idx], vocab[h4_idx]], vocab[t_idx])
        atomic_facts_3.append(item)
        
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    save_dir = os.path.join(base_dir, "data", f"threehop.{args.num_tokens}.inf")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=4)

    with open(os.path.join(save_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_inferred, f, indent=4)

    with open(os.path.join(save_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4)

    with open(os.path.join(save_dir, "atomic_facts_f1.json"), "w", encoding="utf-8") as f:
        json.dump(atomic_facts_1, f, indent=4)
    with open(os.path.join(save_dir, "atomic_facts_f2.json"), "w", encoding="utf-8") as f:
        json.dump(atomic_facts_2, f, indent=4)
    with open(os.path.join(save_dir, "atomic_facts_f3.json"), "w", encoding="utf-8") as f:
        json.dump(atomic_facts_3, f, indent=4)
        
    grouped_id_train_data_1, grouped_id_test_data_1, grouped_ood_test_data_1 = load_and_preprocess_data(
        f1_dict, f2_dict, f3_dict, test_data, idx=1
    )
    grouped_id_train_data_2, grouped_id_test_data_2, grouped_ood_test_data_2 = load_and_preprocess_data(
        f1_dict, f2_dict, f3_dict, test_data, idx=2
    )
    grouped_id_train_data_3, grouped_id_test_data_3, grouped_ood_test_data_3 = load_and_preprocess_data(
        f1_dict, f2_dict, f3_dict, test_data, idx=3
    )
    
    logging.info("Deduplicate ID train...")
    id_train_dedup, id_train_stats = deduplicate_vectors(grouped_id_train_data_1, 1)

    for ctype, b_dict in id_train_dedup.items():
        logging.info(f"ctype: {ctype}")
        for b, items in b_dict.items():
            logging.info(f"     b: {b}, len: {len(items)}")

    logging.info("Deduplicate ID test...")
    id_test_dedup, id_test_stats = deduplicate_vectors(grouped_id_test_data_1, 1)

    for ctype, b_dict in id_test_dedup.items():
        logging.info(f"ctype: {ctype}")
        for b, items in b_dict.items():
            logging.info(f"     b: {b}, len: {len(items)}")

    logging.info("Deduplicate OOD test...")
    ood_dedup, ood_stats = deduplicate_vectors(grouped_ood_test_data_1, 1)

    for ctype, b_dict in ood_dedup.items():
        logging.info(f"ctype: {ctype}")
        logging.info(len(b_dict))
        # for b, items in b_dict.items():
        #     logging.info(f"b: {b}, len: {len(items)}")
        
    logging.info("Deduplicate ID train...")
    id_train_dedup, id_train_stats = deduplicate_vectors(grouped_id_train_data_2, 2)

    for ctype, b_dict in id_train_dedup.items():
        logging.info(f"ctype: {ctype}")
        for b, items in b_dict.items():
            logging.info(f"     b: {b}, len: {len(items)}")

    logging.info("Deduplicate ID test...")
    id_test_dedup, id_test_stats = deduplicate_vectors(grouped_id_test_data_2, 2)

    for ctype, b_dict in id_test_dedup.items():
        logging.info(f"ctype: {ctype}")
        for b, items in b_dict.items():
            logging.info(f"     b: {b}, len: {len(items)}")

    logging.info("Deduplicate OOD test...")
    ood_dedup, ood_stats = deduplicate_vectors(grouped_ood_test_data_2, 2)

    for ctype, b_dict in ood_dedup.items():
        logging.info(f"ctype: {ctype}")
        logging.info(len(b_dict))
        # for b, items in b_dict.items():
        #     logging.info(f"b: {b}, len: {len(items)}")
        
    logging.info("Deduplicate ID train...")
    id_train_dedup, id_train_stats = deduplicate_vectors(grouped_id_train_data_3, 3)

    for ctype, b_dict in id_train_dedup.items():
        logging.info(f"ctype: {ctype}")
        for b, items in b_dict.items():
            logging.info(f"     b: {b}, len: {len(items)}")

    logging.info("Deduplicate ID test...")
    id_test_dedup, id_test_stats = deduplicate_vectors(grouped_id_test_data_3, 3)

    for ctype, b_dict in id_test_dedup.items():
        logging.info(f"ctype: {ctype}")
        for b, items in b_dict.items():
            logging.info(f"     b: {b}, len: {len(items)}")

    logging.info("Deduplicate OOD test...")
    ood_dedup, ood_stats = deduplicate_vectors(grouped_ood_test_data_3, 3)

    for ctype, b_dict in ood_dedup.items():
        logging.info(f"ctype: {ctype}")
        logging.info(len(b_dict))
        # for b, items in b_dict.items():
        #     logging.info(f"b: {b}, len: {len(items)}")

if __name__=="__main__":
    main()