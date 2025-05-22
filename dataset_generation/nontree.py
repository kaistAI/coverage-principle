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
    Create an arbitrary mapping from 'domain' to 'codomain', then split it into
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
    Convert a list of N input tokens plus 1 output token into a dictionary:
      {
        "input_text": "<t_0><t_1><t_2>",
        "target_text": "<t_0><t_1><t_2><t_9></a>"
      }
    Used for single-hop atomic facts.
    """
    inp = "".join(input_tokens)
    tgt = inp + output_token + "</a>"
    return {"input_text": inp, "target_text": tgt}


def form_item_2hop_cot(h1_idx, h2_idx, h3_idx, t_idx, vocab, f1_dict, cot=False):
    """
    For the final 2-hop samples:
      - We have input: e0=e_h1_idx, e1=e_h2_idx, e2=e_h3_idx
      - We have bridging entity b1 = f1(e0,e1) from S_f1
      - We have final label y = f2(b1,e1,e2)

    If 'cot' is True => the target has two tokens: <b1><y></a>
    Otherwise => the target has one token: <y></a>
    """
    b1_idx = f1_dict[(h1_idx, h2_idx)]
    bridging_token = vocab[b1_idx]
    out_token = vocab[t_idx]

    inp_tokens = [vocab[h1_idx], vocab[h2_idx], vocab[h3_idx]]
    inp = "".join(inp_tokens)

    if cot:
        # chain-of-thought style: <b1><final_label>
        tgt = inp + bridging_token + out_token + "</a>"
    else:
        # original style: <final_label>
        tgt = inp + out_token + "</a>"

    return {"input_text": inp, "target_text": tgt}


def process_item_S_f1(item, S_f2_index):
    """
    Used for parallel processing to find all 2-hop tuples of the form
    (h1, h2, h3, t) where:
       b1 = f1(h1, h2)
       t  = f2(b1, h2, h3).

    'item' is ( (h1,h2), b1 ) from S_f1.
    We look up (b1, h2) in S_f2_index => all (h3, t) pairs => produce (h1,h2,h3,t).
    """
    (h1, h2), b1 = item
    partial_set = set()
    # b1,e1 => e1 == h2
    if (b1, h2) not in S_f2_index:
        return partial_set
    for (h3, t) in S_f2_index[(b1, h2)]:
        partial_set.add((h1, h2, h3, t))
    return partial_set


def coverage_type(sc1: bool, sc2: bool):
    """
    Determine coverage type based on whether (f1, f2) for the 2-hop
    have been 'seen' (True) or not (False).
      bit = (sc1 << 1) + sc2 => 0..3 in decimal
      bit=3 => coverage=0  (both seen)
      bit<3 => coverage=bit+1
    """
    bit = (sc1 << 1) + sc2  # 0..3
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
    - tup: (input_tokens_list, output_token_string)  [the old style]
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
                        help="If true, f2 is a duplicate of f1 (same mapping) instead of being created independently.")
    parser.add_argument("--default_seen_ratio", type=float, default=0.7,
                        help="Ratio of the domain considered 'seen' for each function (f1, f2).")
    parser.add_argument("--max_train_data_num", type=int, default=382000,
                        help="Maximum number of 2-hop samples in the training set.")
    parser.add_argument("--test_size_for_type", type=int, default=2000,
                        help="Number of final test samples for each coverage type.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug-level logging (more verbose).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--cot", action="store_true",
                        help="If true, generate two tokens in the label: <bridge_token><final_token>.")
    args = parser.parse_args()

    setup_logging(args.debug)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ------------------------------------------------------
    # 1) Vocab, f1, f2 creation
    # ------------------------------------------------------
    vocab = [f"<t_{i}>" for i in range(args.num_tokens)] + ["</a>"]
    # f1 domain => (e0,e1).  f2 domain => (b1, e1, e2).
    domain_f1 = list(itertools.product(range(args.num_tokens), range(args.num_tokens)))
    domain_f2 = list(itertools.product(
        range(args.num_tokens),   # possible b1
        range(args.num_tokens),   # e1
        range(args.num_tokens)    # e2
    ))
    codomain = list(range(args.num_tokens))

    f1_dict, S_f1 = make_arbitrary_function(domain_f1, codomain, args.default_seen_ratio)

    if args.same_f12:
        # f2 is the same as f1, but that has fewer inputs than we need.
        # For minimal code difference, replicate results across the triple domain.
        f2_dict = {}
        for triple in domain_f2:
            key_2d = (triple[0], triple[1])  # (b1, e1)
            if key_2d in f1_dict:
                f2_dict[triple] = f1_dict[key_2d]
            else:
                f2_dict[triple] = random.choice(codomain)
        # define S_f2 from the partial set
        keys_f2 = list(f2_dict.keys())
        random.shuffle(keys_f2)
        cutoff_f2 = int(round(args.default_seen_ratio * len(keys_f2)))
        seen_f2_keys = keys_f2[:cutoff_f2]
        S_f2 = {k: f2_dict[k] for k in seen_f2_keys}
    else:
        f2_dict, S_f2 = make_arbitrary_function(domain_f2, codomain, args.default_seen_ratio)

    # ------------------------------------------------------
    # 2) Build 'train_inferred' from S_f1, S_f2
    # ------------------------------------------------------
    # (a) Index for S_f2: keyed by (b1_idx, e1_idx) => [(e2_idx, t_idx), ...]
    S_f2_index = {}
    for (b1_idx, e1_idx, e2_idx), t_idx in S_f2.items():
        S_f2_index.setdefault((b1_idx, e1_idx), []).append((e2_idx, t_idx))

    # (b) gather expansions => (h1,h2)->b1 => (b1,h2,h3)->t => (h1,h2,h3,t)
    with mp.Pool(processes=round(mp.cpu_count() * 0.9)) as pool:
        process_S_f1_partial = partial(process_item_S_f1, S_f2_index=S_f2_index)
        results2 = list(tqdm(pool.imap(process_S_f1_partial, list(S_f1.items())),
                             total=len(S_f1),
                             desc="Processing inferred facts based on S_f1, S_f2"))
    seen_expected_inferred_idx = set().union(*results2)
    del results2

    logging.info(f"len(seen_expected_inferred_facts): {len(seen_expected_inferred_idx)}")

    # (c) Randomly pick up to max_train_data_num
    if len(seen_expected_inferred_idx) <= args.max_train_data_num:
        raise Exception(
            "\n\n\n###############################################################\n"
            "All covered data is in train_inferred since --max_train_data_num is too large.\n"
            "Please make the value smaller.\n"
            "###############################################################\n\n\n"
        )
    train_inferred_idx_set: List = set(random.sample(list(seen_expected_inferred_idx),
                                                     args.max_train_data_num))
    del seen_expected_inferred_idx

    # (d) Re-define S_f1, S_f2 to reflect only used mappings
    S_f1, S_f2 = dict(), dict()
    for (h1_idx, h2_idx, h3_idx, t_idx) in tqdm(train_inferred_idx_set):
        b1_idx = f1_dict[(h1_idx, h2_idx)]
        S_f1[(h1_idx, h2_idx)] = b1_idx
        S_f2[(b1_idx, h2_idx, h3_idx)] = f2_dict[(b1_idx, h2_idx, h3_idx)]

    OOD_f1 = {key: value for key, value in f1_dict.items() if key not in S_f1}
    OOD_f2 = {key: value for key, value in f2_dict.items() if key not in S_f2}

    assert not (set(S_f1.items()) & set(OOD_f1.items()))
    assert not (set(S_f2.items()) & set(OOD_f2.items()))

    logging.info(f"real_seen_f1_set len: {len(S_f1)}")
    logging.info(f"real_seen_f2_set len: {len(S_f2)}")
    logging.info(f"real_not_seen_f1_set len: {len(OOD_f1)}")
    logging.info(f"real_not_seen_f2_set len: {len(OOD_f2)}")

    # (e) Final train_inferred data
    train_inferred = []
    for (h1_idx, h2_idx, h3_idx, t_idx) in train_inferred_idx_set:
        sample_dict = form_item_2hop_cot(
            h1_idx=h1_idx,
            h2_idx=h2_idx,
            h3_idx=h3_idx,
            t_idx=t_idx,
            vocab=vocab,
            f1_dict=f1_dict,
            cot=args.cot
        )
        train_inferred.append(sample_dict)

    logging.info(f"train_inferred:\n    example: {train_inferred[:10]}\n    len: {len(train_inferred)}")

    # ------------------------------------------------------
    # 3) Build coverage type samples (test) using skip+reservoir
    # ------------------------------------------------------
    # We build an index for the full f2_dict as well
    f2_index = defaultdict(list)
    for (b1_idx, e1_idx, e2_idx), t_idx in f2_dict.items():
        f2_index[(b1_idx, e1_idx)].append((e2_idx, t_idx))

    coverage_reservoirs = defaultdict(list)
    coverage_seen_count = defaultdict(int)

    skip_p = 0.0

    for (h1_idx, h2_idx), b1_idx in f1_dict.items():
        if random.random() > (1 - skip_p):
            continue
        sc1 = ((h1_idx, h2_idx) in S_f1)

        if (b1_idx, h2_idx) not in f2_index:
            continue
        for (h3_idx, t_idx) in f2_index[(b1_idx, h2_idx)]:
            if (h1_idx, h2_idx, h3_idx, t_idx) in train_inferred_idx_set:
                continue
            if random.random() > (1 - skip_p):
                continue
            sc2 = ((b1_idx, h2_idx, h3_idx) in S_f2)
            c_type = coverage_type(sc1, sc2)
            coverage_seen_count[f"type_{c_type}"] += 1

            # Here for coverage test samples, we do "classic" form_item for reservoir
            # but we need bridging if cot is set. We'll store input tokens + bridging
            # in the reservoir as well => let's do a local approach:
            # We'll store the final as a single token, then re-convert it below
            reservoir_update(
                reservoir=coverage_reservoirs[f"type_{c_type}"],
                tup=([vocab[h1_idx], vocab[h2_idx], vocab[h3_idx]], vocab[t_idx]),
                total_count=coverage_seen_count[f"type_{c_type}"],
                capacity=args.test_size_for_type
            )

    for ctype, item_list in coverage_reservoirs.items():
        logging.info(f"{ctype}: {len(item_list)}\n    example: {item_list[:5]}")

    # (a) coverage=0..3, plus we also want some "train_inferred" => test_data
    test_data = []

    # (1) from train_inferred => "train_inferred"
    train_sample_for_test = choose(train_inferred, args.test_size_for_type)
    for item in train_sample_for_test:
        item["type"] = "train_inferred"
        test_data.append(item)

    # (2) coverage=0..3 => re-convert them to have bridging if cot
    #    because reservoir_update used the old form_item style
    final_coverage_items = []
    for ctype, item_list in coverage_reservoirs.items():
        for it in item_list:
            # it["input_text"]: e.g. "<t_5><t_99><t_10>"
            # we parse them => h1,h2,h3 => find bridging => form final with or w/o CoT
            raw_inp_str = it["input_text"]  # e.g. "<t_5><t_99><t_10>"
            # parse it
            tokens_in = raw_inp_str.strip("<>").split("><")
            if len(tokens_in) != 3:
                continue
            # find bridging
            # h1_idx => int(tokens_in[0].split("_")[-1])
            # h2_idx => ...
            h1_idx = int(tokens_in[0].split("_")[-1])
            h2_idx = int(tokens_in[1].split("_")[-1])
            h3_idx = int(tokens_in[2].split("_")[-1])
            # final label
            raw_out_str = it["target_text"]  # e.g. "<t_5><t_99><t_10><t_34></a>"
            # parse final token from the output
            out_tok = raw_out_str.replace("</a>", "").strip("<>").split("><")[-1]
            t_idx = int(out_tok.split("_")[-1])

            # build the new item with bridging if needed
            new_item = form_item_2hop_cot(
                h1_idx=h1_idx,
                h2_idx=h2_idx,
                h3_idx=h3_idx,
                t_idx=t_idx,
                vocab=vocab,
                f1_dict=f1_dict,
                cot=args.cot
            )
            new_item["type"] = ctype
            final_coverage_items.append(new_item)

    # combine coverage items
    test_data.extend(final_coverage_items)

    random.shuffle(train_inferred)
    random.shuffle(test_data)

    # ------------------------------------------------------
    # 5) Atomic facts remain single-output
    # ------------------------------------------------------
    atomic_facts_f1 = []
    for (h1_idx, h2_idx), b1_idx in f1_dict.items():
        atomic_facts_f1.append(form_item([vocab[h1_idx], vocab[h2_idx]], vocab[b1_idx]))

    atomic_facts_f2 = []
    for (b1_idx, e1_idx, e2_idx), t_idx in f2_dict.items():
        atomic_facts_f2.append(form_item([vocab[b1_idx], vocab[e1_idx], vocab[e2_idx]], vocab[t_idx]))

    # ------------------------------------------------------
    # 6) Save results
    # ------------------------------------------------------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # label it with "cot" or "nocot" for clarity if you want
    mode_str = "cot" if args.cot else "inf"
    save_dir = os.path.join(
        base_dir,
        "data",
        f"nontree.{args.num_tokens}.{args.max_train_data_num}.{'same-f12' if args.same_f12 else 'diff-f12'}.{mode_str}"
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
