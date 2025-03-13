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


def setup_logging(debug_mode: bool):
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s - %(message)s")


def make_arbitrary_function(domain: List[int], codomain: List[int], seen_ratio: float):
    """
    Create an arbitrary mapping from domain to codomain, then split it into 
    'seen' and 'OOD' subsets according to 'seen_ratio'.
    """
    result = {}
    range_vals = random.choices(codomain, k=len(domain))
    for operand, res in zip(domain, range_vals):
        result[operand] = res

    keys = list(result.keys())
    random.shuffle(keys)
    cutoff = int(round(seen_ratio * len(keys)))
    seen_expected_inputs, unseen_inputs = keys[:cutoff], keys[cutoff:]

    S_dict = {inp: result[inp] for inp in seen_expected_inputs}
    return result, S_dict


def form_item(input_tokens: List[str], output_token: str):
    """
    Convert a list of input tokens plus one output token into a dictionary.
    Example: 4 inputs => "<t_1><t_2><t_3><t_4>", 
             output   => "<t_99>", 
    then "target_text" => "<t_1><t_2><t_3><t_4><t_99></a>"
    """
    inp = "".join(input_tokens)
    tgt = inp + output_token + "</a>"
    return {"input_text": inp, "target_text": tgt}


def coverage_type(sc1: bool, sc2: bool, sc3: bool):
    """
    Determine coverage type based on whether each of the three sub-computations
    (f1, f2, f3) is in the 'seen' set (True) or not (False).

    If sc1=sc2=sc3=True (bits=111 => 7), coverage=0.
    Otherwise coverage=bits+1 in [1..7].
    """
    bits = (sc1 << 2) + (sc2 << 1) + sc3
    if bits == 7:  # 111
        return 0
    else:
        return bits + 1  # 1..7


def choose(arr, n_or_ratio):
    """
    If n_or_ratio is an int, pick that many random samples from arr.
    If it's a float, interpret it as ratio of the array length.
    """
    if isinstance(n_or_ratio, float):
        n = int(round(len(arr) * n_or_ratio))
    else:
        n = n_or_ratio
    if n >= len(arr):
        return arr
    return random.sample(arr, n)


def reservoir_update(reservoir, tup, total_count, capacity):
    """
    Perform a single step of Reservoir Sampling for one coverage bucket.
    - reservoir: current list of items in the reservoir
    - tup: (input_tokens_list, output_token_string)
    - total_count: how many items have been seen so far (for this coverage)
    - capacity: reservoir size limit
    """
    if len(reservoir) < capacity:
        reservoir.append(form_item(tup[0], tup[1]))
    else:
        # choose a random index from [0..total_count-1]
        r = random.randint(0, total_count - 1)
        if r < capacity:
            reservoir[r] = form_item(tup[0], tup[1])



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tokens", type=int, required=True,
                        help="Number of tokens (excluding the final '</a>' token).")
    parser.add_argument("--same_f123", action="store_true",
                        help="If true, f2 and f3 are duplicates of f1 (same mapping) instead of being created independently.")
    parser.add_argument("--default_seen_ratio", type=float, default=0.7,
                        help="Ratio of the domain considered 'seen' for each function (f1, f2, f3).")
    parser.add_argument("--max_train_data_num", type=int, default=382000,
                        help="Maximum number of parallel-2-hop samples in the training set.")
    parser.add_argument("--test_size_for_type", type=int, default=3000,
                        help="Number of final test samples for each coverage type.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug-level logging (more verbose).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--include_atomic", action="store_true",
                        help="If true, include atomic facts in the training set.")
    args = parser.parse_args()

    # 1) Setup
    setup_logging(args.debug)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 2) Create vocab, domain, codomain
    vocab = [f"<t_{i}>" for i in range(args.num_tokens)] + ["</a>"]
    domain = list(itertools.product(range(args.num_tokens), range(args.num_tokens)))
    codomain = list(range(args.num_tokens))

    # 3) Create f1, f2, f3 (either identical or independent)
    f1_dict, S_f1 = make_arbitrary_function(domain, codomain, args.default_seen_ratio)
    if args.same_f123:
        f2_dict = dict(f1_dict)
        S_f2 = dict(S_f1)
        f3_dict = dict(f1_dict)
        S_f3 = dict(S_f1)
    else:
        f2_dict, S_f2 = make_arbitrary_function(domain, codomain, args.default_seen_ratio)
        f3_dict, S_f3 = make_arbitrary_function(domain, codomain, args.default_seen_ratio)

    # 4) Build the "inferred" set for training.
    #
    # We want all valid chains (h1,h2) => b1, (h3,h4) => b2, (b1,b2) => t
    # but only from the "seen" subsets S_f1, S_f2, S_f3 (so fully in-distribution).
    # Then we randomly pick up to `max_train_data_num`.
    #
    # We'll do this in parallel as well, though it is a cross-product approach:
    # For each ( (h1,h2), b1 ) in S_f1, we combine with all ( (h3,h4), b2 ) in S_f2,
    # then see if (b1,b2) in S_f3 => t. That yields (h1,h2,h3,h4,t).

    # a) For quick membership check in S_f3:
    #    S_f3 is a dict of ((b1,b2) -> t).
    #    We can do a direct membership test: if (b1,b2) in S_f3 => t
    # b) We'll flatten S_f2 into a list for easier parallelization.
    S_f2_list = list(S_f2.items())  # [ ( (h3,h4), b2 ), ... ]

    # Define a parallel worker
    def worker_f1_parallel2hop(item_f1, S_f2_list, S_f3):
        """
        item_f1 is ((h1,h2), b1).
        We'll combine with each item in S_f2_list:
        [(((h3,h4), b2), ...)]
        Then check if (b1, b2) in S_f3 => t.
        """
        ((h1, h2), b1) = item_f1
        expansions = []
        for ((h3, h4), b2) in S_f2_list:
            if (b1, b2) in S_f3:
                t = S_f3[(b1, b2)]
                expansions.append((h1, h2, h3, h4, t))
        return expansions

    # c) Map over S_f1 in parallel, collect expansions
    results_f1 = []
    for item_f1 in tqdm(S_f1.items(), desc="Processing parallel-2-hop expansions (S_f1, S_f2, S_f3)"):
        expansions = worker_f1_parallel2hop(item_f1, S_f2_list, S_f3)
        results_f1.append(expansions)
        
    all_inferred_idx = set()
    for sublist in results_f1:
        all_inferred_idx.update(sublist)
    del results_f1  # free memory

    logging.info(f"Number of fully-seen expansions: {len(all_inferred_idx)}")

    # d) Randomly pick up to max_train_data_num from that set
    if len(all_inferred_idx) <= args.max_train_data_num:
        raise Exception(
            "\n\n\n###############################################################\n"
            "All data is in train_inferred since MAX_TRAIN_DATA_NUM is too large.\n"
            "Please choose a smaller value.\n"
            "###############################################################\n\n\n"
        )
    train_inferred_idx_set = set(random.sample(list(all_inferred_idx), args.max_train_data_num))
    del all_inferred_idx

    # e) Re-define S_f1, S_f2, S_f3 to reflect only the actually used mappings
    #    This is analogous to the 3-hop code but we adapt to the parallel structure.
    S_f1_used = {}
    S_f2_used = {}
    S_f3_used = {}

    for (h1, h2, h3, h4, t) in train_inferred_idx_set:
        b1 = f1_dict[(h1, h2)]
        S_f1_used[(h1, h2)] = b1

        b2 = f2_dict[(h3, h4)]
        S_f2_used[(h3, h4)] = b2

        S_f3_used[(b1, b2)] = t

    # OOD definitions
    OOD_f1 = {k: v for k, v in f1_dict.items() if k not in S_f1_used}
    OOD_f2 = {k: v for k, v in f2_dict.items() if k not in S_f2_used}
    OOD_f3 = {k: v for k, v in f3_dict.items() if k not in S_f3_used}

    # logging info
    logging.info(f"real_seen_f1_set len: {len(S_f1_used)}")
    logging.info(f"real_seen_f2_set len: {len(S_f2_used)}")
    logging.info(f"real_seen_f3_set len: {len(S_f3_used)}")
    logging.info(f"real_not_seen_f1_set len: {len(OOD_f1)}")
    logging.info(f"real_not_seen_f2_set len: {len(OOD_f2)}")
    logging.info(f"real_not_seen_f3_set len: {len(OOD_f3)}")

    # f) Build final train_inferred data
    train_inferred = []
    for (h1, h2, h3, h4, t) in train_inferred_idx_set:
        inp_tokens = [vocab[h1], vocab[h2], vocab[h3], vocab[h4]]
        out_token = vocab[t]
        train_inferred.append(form_item(inp_tokens, out_token))

    logging.info(f"train_inferred:\n    example: {train_inferred[:5]}\n    len: {len(train_inferred)}")

    # 5) Build coverage-type test samples
    #
    # We do a triple nested loop:
    #   for each (h1,h2) => b1 in f1_dict
    #   for each (h3,h4) => b2 in f2_dict
    #   if (b1,b2) => t in f3_dict
    # 
    # We apply skip probability for each stage (like in the 3-hop code),
    # then compute coverage type as sc1, sc2, sc3.
    # sc1 = ((h1,h2) in S_f1_used)
    # sc2 = ((h3,h4) in S_f2_used)
    # sc3 = ((b1,b2) in S_f3_used)
    # 
    # We'll keep the same skip_p=0.4, same reservoir sampling logic.

    skip_p = 0.4
    coverage_reservoirs = defaultdict(list)
    coverage_seen_count = defaultdict(int)

    for (h1_idx, h2_idx), b1_idx in f1_dict.items():
        # skip stage-1
        if random.random() > (1 - skip_p):
            continue
        sc1 = ((h1_idx, h2_idx) in S_f1_used)

        for (h3_idx, h4_idx), b2_idx in f2_dict.items():
            # skip stage-2
            if random.random() > (1 - skip_p):
                continue
            sc2 = ((h3_idx, h4_idx) in S_f2_used)

            # final step: check if (b1_idx, b2_idx) in f3_dict
            if (b1_idx, b2_idx) in f3_dict:
                # skip stage-3
                if random.random() > (1 - skip_p):
                    continue
                t_idx = f3_dict[(b1_idx, b2_idx)]
                sc3 = ((b1_idx, b2_idx) in S_f3_used)

                ctype = coverage_type(sc1, sc2, sc3)
                coverage_seen_count[f"type_{ctype}"] += 1

                # input tokens: 4
                # output token: t_idx
                reservoir_update(
                    reservoir=coverage_reservoirs[f"type_{ctype}"],
                    tup=(
                        [vocab[h1_idx], vocab[h2_idx], vocab[h3_idx], vocab[h4_idx]],
                        vocab[t_idx]
                    ),
                    total_count=coverage_seen_count[f"type_{ctype}"],
                    capacity=args.test_size_for_type
                )

    for ctype, item_list in coverage_reservoirs.items():
        logging.info(f"{ctype}: {len(item_list)}\n    example: {item_list[:5]}")

    # 6) Final test data
    #    - pick some from train_inferred => "train_inferred" type
    #    - plus coverage_buckets from reservoir => "type_0..7"
    test_data = []

    # (a) from train_inferred
    train_sample_for_test = choose(train_inferred, args.test_size_for_type)
    for item in train_sample_for_test:
        item["type"] = "train_inferred"
        test_data.append(item)

    # (b) coverage=0..7
    for ctype, item_list in coverage_reservoirs.items():
        for item in item_list:
            item["type"] = ctype
            test_data.append(item)

    random.shuffle(train_inferred)
    random.shuffle(test_data)

    # 7) Atomic facts
    #    For parallel-2-hop, each function still has domain => pair, codomain => single token.
    atomic_facts_f1 = []
    for (h1_idx, h2_idx), b1_idx in f1_dict.items():
        atomic_facts_f1.append(form_item([vocab[h1_idx], vocab[h2_idx]], vocab[b1_idx]))

    atomic_facts_f2 = []
    for (h3_idx, h4_idx), b2_idx in f2_dict.items():
        atomic_facts_f2.append(form_item([vocab[h3_idx], vocab[h4_idx]], vocab[b2_idx]))

    atomic_facts_f3 = []
    for (b1_idx, b2_idx), t_idx in f3_dict.items():
        atomic_facts_f3.append(form_item([vocab[b1_idx], vocab[b2_idx]], vocab[t_idx]))

    # 8) Save results
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, "data",
                            f"parallel2hop.{args.num_tokens}.{args.max_train_data_num}.{'same-f123' if args.same_f123 else 'diff-f123'}.{'include-atomic' if args.include_atomic else 'inf'}")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    with open(os.path.join(save_dir, "train.json"), "w", encoding="utf-8") as f:
        if args.include_atomic:
            assert args.same_f123 # only support this for now
            json.dump(train_inferred+atomic_facts_f1, f, indent=2)
        else:
            json.dump(train_inferred, f, indent=2)
            
    with open(os.path.join(save_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)
    with open(os.path.join(save_dir, "atomic_facts_f1.json"), "w", encoding="utf-8") as f:
        json.dump(atomic_facts_f1, f, indent=2)
    with open(os.path.join(save_dir, "atomic_facts_f2.json"), "w", encoding="utf-8") as f:
        json.dump(atomic_facts_f2, f, indent=2)
    with open(os.path.join(save_dir, "atomic_facts_f3.json"), "w", encoding="utf-8") as f:
        json.dump(atomic_facts_f3, f, indent=2)

    print("[INFO] Done!")
    print(f"Train size: {len(train_inferred)}")
    print(f"Test size: {len(test_data)}")
    print(f"S_f1_used: {len(S_f1_used)}, S_f2_used: {len(S_f2_used)}, S_f3_used: {len(S_f3_used)}")
    print(f"OOD_f1: {len(OOD_f1)}, OOD_f2: {len(OOD_f2)}, OOD_f3: {len(OOD_f3)}")


if __name__ == "__main__":
    main()