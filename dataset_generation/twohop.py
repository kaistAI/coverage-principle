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
    Original single-token format. 
      "input_text": "<t_0><t_1><t_2>",
      "target_text": "<t_0><t_1><t_2><t_9></a>"
    Used for:
      - coverage reservoir (temporary storage)
      - atomic facts
    """
    inp = "".join(input_tokens)
    tgt = inp + output_token + "</a>"
    return {"input_text": inp, "target_text": tgt}


def form_item_2hop_target(h1_idx, h2_idx, h3_idx, t_idx,
                          vocab, f1_dict,
                          cot=False,
                          fake_bridge=False):
    """
    For the final 2-hop sample:
      - input => <t_h1><t_h2><t_h3>

    Then the final target has either:
      1) single token => <t_t>
      2) real bridging => <t_b1><t_t>  (if cot=True, fake_bridge=False)
      3) fake bridging => </b><t_t>   (if fake_bridge=True)

    Note: There's only one bridging entity b1 = f1(h1,h2).
    """
    input_tokens = [vocab[h1_idx], vocab[h2_idx], vocab[h3_idx]]
    inp_str = "".join(input_tokens)

    b1_idx = f1_dict.get((h1_idx, h2_idx), None)
    final_tok = vocab[t_idx]

    # Decide bridging behavior
    if fake_bridge:
        # always produce </b> before final
        bridging_str = "</b>"
    elif cot and (b1_idx is not None):
        bridging_str = vocab[b1_idx]
    else:
        bridging_str = ""

    target_str = inp_str + bridging_str + final_tok + "</a>"
    return {"input_text": inp_str, "target_text": target_str}


def process_item_S_f1(item, S_f2_index):
    """
    Used for parallel processing to find all 2-hop tuples 
    (h1, h2, h3, t) that can be formed using S_f1, S_f2.
    
    'item' is ( (h1,h2), b1 ).
    We look up 'b1' in S_f2_index => all (h3, t) pairs => (h1, h2, h3, t).
    """
    (h1, h2), b1 = item
    partial_set = set()
    # b1 must be in S_f2_index to form 2-hop
    if b1 not in S_f2_index:
        return partial_set
    for h3, t in S_f2_index[b1]:
        partial_set.add((h1, h2, h3, t))
    return partial_set


def coverage_type(sc1: bool, sc2: bool):
    """
    Determine coverage type based on whether (f1, f2) for the 2-hop 
    have been 'seen' (True) or not (False).
    bit = sc1<<1 + sc2 => 0..3
      bit=3 => coverage=0 (both seen)
      else => bit+1 => coverage=1..3
    """
    bit = (sc1 << 1) + sc2
    return 0 if bit == 3 else bit + 1


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
    - total_count: how many items have we seen so far (for this coverage)
    - capacity: reservoir size limit
    """
    if len(reservoir) < capacity:
        reservoir.append(form_item(tup[0], tup[1]))
    else:
        r = random.randint(0, total_count - 1)
        if r < capacity:
            reservoir[r] = form_item(tup[0], tup[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tokens", type=int, required=True,
                        help="Number of tokens (excluding the final '</a>' token).")
    parser.add_argument("--same_f12", action="store_true",
                        help="If true, f2 is a duplicate of f1 (same mapping).")
    parser.add_argument("--default_seen_ratio", type=float, default=0.7,
                        help="Ratio of the domain considered 'seen' for each function (f1, f2).")
    parser.add_argument("--max_train_data_num", type=int, default=382000,
                        help="Maximum number of 2-hop samples in the training set.")
    parser.add_argument("--test_size_for_type", type=int, default=3000,
                        help="Number of final test samples for each coverage type.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug-level logging.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--cot", action="store_true",
                        help="If true, produce bridging entity <b1> before final label.")
    parser.add_argument("--fake_bridge", action="store_true",
                        help="If true, produce </b> token before final label.")
    args = parser.parse_args()

    setup_logging(args.debug)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ------------------------------------------------------
    # 1) Vocab
    # ------------------------------------------------------
    vocab = [f"<t_{i}>" for i in range(args.num_tokens)]
    if args.fake_bridge:
        vocab.append("</b>")
    vocab.append("</a>")  
    # So if fake_bridge is true => vocab has an extra token: </b>

    # For a 2-hop chain: f1 domain is (e0,e1), f2 domain is (b1,e2).
    domain_f1 = list(itertools.product(range(args.num_tokens), range(args.num_tokens)))
    domain_f2 = list(itertools.product(range(args.num_tokens), range(args.num_tokens)))
    codomain = list(range(args.num_tokens))

    f1_dict, S_f1 = make_arbitrary_function(domain_f1, codomain, args.default_seen_ratio)
    if args.same_f12:
        f2_dict = dict(f1_dict)
        S_f2 = dict(S_f1)
    else:
        f2_dict, S_f2 = make_arbitrary_function(domain_f2, codomain, args.default_seen_ratio)

    # ------------------------------------------------------
    # 2) gather expansions => train_inferred
    # ------------------------------------------------------
    # (a) index f2 => b1 => [ (h3, t), ... ]
    S_f2_index = defaultdict(list)
    for (b1_idx, h3_idx), t_idx in S_f2.items():
        S_f2_index[b1_idx].append((h3_idx, t_idx))

    # (b) expansions
    with mp.Pool(processes=round(mp.cpu_count() * 0.9)) as pool:
        process_f1 = partial(process_item_S_f1, S_f2_index=S_f2_index)
        results2 = list(tqdm(pool.imap(process_f1, list(S_f1.items())),
                             total=len(S_f1),
                             desc="Processing inferred facts f1->f2"))
    seen_expected_inferred_idx = set().union(*results2)
    del results2

    logging.info(f"len(seen_expected_inferred_facts): {len(seen_expected_inferred_idx)}")

    # (c) sample up to max_train_data_num
    if len(seen_expected_inferred_idx) <= args.max_train_data_num:
        raise Exception(
            "All covered data is in train_inferred. Try smaller --max_train_data_num."
        )
    train_inferred_idx_set = set(random.sample(list(seen_expected_inferred_idx),
                                               args.max_train_data_num))
    del seen_expected_inferred_idx

    # (d) re-define S_f1, S_f2 for only the used pairs
    S_f1, S_f2 = {}, {}
    for (h1_idx, h2_idx, h3_idx, t_idx) in tqdm(train_inferred_idx_set):
        b1_idx = f1_dict[(h1_idx, h2_idx)]
        S_f1[(h1_idx, h2_idx)] = b1_idx
        S_f2[(b1_idx, h3_idx)] = f2_dict[(b1_idx, h3_idx)]

    OOD_f1 = {k for k in f1_dict if k not in S_f1}
    OOD_f2 = {k for k in f2_dict if k not in S_f2}

    logging.info(f"real_seen_f1_set len: {len(S_f1)}; OOD_f1: {len(OOD_f1)}")
    logging.info(f"real_seen_f2_set len: {len(S_f2)}; OOD_f2: {len(OOD_f2)}")

    # (e) build final train data
    train_inferred = []
    for (h1_idx, h2_idx, h3_idx, t_idx) in train_inferred_idx_set:
        sample = form_item_2hop_target(
            h1_idx=h1_idx, h2_idx=h2_idx, h3_idx=h3_idx,
            t_idx=t_idx,
            vocab=vocab,
            f1_dict=S_f1,
            cot=(not args.fake_bridge and args.cot),
            fake_bridge=args.fake_bridge
        )
        train_inferred.append(sample)

    logging.info(f"train_inferred: len={len(train_inferred)}  example={train_inferred[:5]}")

    # ------------------------------------------------------
    # 3) coverage type => reservoir
    # ------------------------------------------------------
    f2_index = defaultdict(list)
    for (b1_idx, h3_idx), t_idx in f2_dict.items():
        f2_index[b1_idx].append((h3_idx, t_idx))

    coverage_reservoirs = defaultdict(list)
    coverage_seen_count = defaultdict(int)

    skip_p = 0.4
    for (h1_idx, h2_idx), b1_idx in f1_dict.items():
        if random.random() > (1 - skip_p):
            continue
        sc1 = ((h1_idx, h2_idx) in S_f1)

        for (h3_idx, t_idx) in f2_index[b1_idx]:
            if random.random() > (1 - skip_p):
                continue
            sc2 = ((b1_idx, h3_idx) in S_f2)
            c_type = coverage_type(sc1, sc2)

            coverage_seen_count[f"type_{c_type}"] += 1
            reservoir_update(
                coverage_reservoirs[f"type_{c_type}"],
                tup=([vocab[h1_idx], vocab[h2_idx], vocab[h3_idx]], vocab[t_idx]),
                total_count=coverage_seen_count[f"type_{c_type}"],
                capacity=args.test_size_for_type
            )

    # ------------------------------------------------------
    # 4) final test data
    # ------------------------------------------------------
    test_data = []
    # (a) some from train_inferred => "train_inferred"
    train_sample_for_test = choose(train_inferred, args.test_size_for_type)
    for item in train_sample_for_test:
        item["type"] = "train_inferred"
        test_data.append(item)

    # (b) coverage=0..3 => re-convert if bridging
    final_coverage_items = []
    for ctype, item_list in coverage_reservoirs.items():
        for it in item_list:
            inp_str = it["input_text"]   # e.g. "<t_11><t_12><t_2>"
            out_str = it["target_text"]  # e.g. "<t_11><t_12><t_2><t_99></a>"
            tokens_in = inp_str.strip("<>").split("><")
            if len(tokens_in) != 3:
                continue
            h1_idx = int(tokens_in[0].split("_")[-1])
            h2_idx = int(tokens_in[1].split("_")[-1])
            h3_idx = int(tokens_in[2].split("_")[-1])

            tokens_out = out_str.replace("</a>", "").strip("<>").split("><")
            final_tok = tokens_out[-1]  # e.g. "t_99"
            t_idx = int(final_tok.split("_")[-1])

            # re-convert if needed
            new_item = form_item_2hop_target(
                h1_idx=h1_idx, h2_idx=h2_idx, h3_idx=h3_idx,
                t_idx=t_idx,
                vocab=vocab,
                f1_dict=f1_dict,  # if we want bridging for OOD coverage
                cot=(not args.fake_bridge and args.cot),
                fake_bridge=args.fake_bridge
            )
            new_item["type"] = ctype
            final_coverage_items.append(new_item)

    test_data.extend(final_coverage_items)

    random.shuffle(train_inferred)
    random.shuffle(test_data)

    # ------------------------------------------------------
    # 5) Atomic facts => single token
    # ------------------------------------------------------
    atomic_facts_f1 = []
    for (h1_idx, h2_idx), b1_idx in f1_dict.items():
        atomic_facts_f1.append(form_item([vocab[h1_idx], vocab[h2_idx]], vocab[b1_idx]))

    atomic_facts_f2 = []
    for (b1_idx, h3_idx), t_idx in f2_dict.items():
        atomic_facts_f2.append(form_item([vocab[b1_idx], vocab[h3_idx]], vocab[t_idx]))

    # ------------------------------------------------------
    # 6) Save results
    # ------------------------------------------------------
    mode_str = "inf"
    if args.fake_bridge:
        mode_str = "fakebridge"
    elif args.cot:
        mode_str = "cot"

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(
        base_dir, 
        "data",
        f"twohop.{args.num_tokens}.{args.max_train_data_num}.{'same-f12' if args.same_f12 else 'diff-f12'}.{mode_str}"
    )
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    with open(os.path.join(save_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_inferred, f, indent=2)
    with open(os.path.join(save_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)
    with open(os.path.join(save_dir, "atomic_facts_f1.json"), "w", encoding="utf-8") as f:
        json.dump(atomic_facts_f1, f, indent=2)
    with open(os.path.join(save_dir, "atomic_facts_f2.json"), "w", encoding="utf-8") as f:
        json.dump(atomic_facts_f2, f, indent=2)

    print("[INFO] Done!")
    print(f"Train size: {len(train_inferred)}")
    print(f"Test size: {len(test_data)}")
    print(f"S_f1: {len(S_f1)}, S_f2: {len(S_f2)}")
    print(f"OOD_f1: {len(OOD_f1)}, OOD_f2: {len(OOD_f2)}")


if __name__ == "__main__":
    main()
