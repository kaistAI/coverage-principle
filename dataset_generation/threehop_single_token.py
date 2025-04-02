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
    Original single-token style item, used for:
      - coverage reservoir intermediate format
      - atomic facts f1, f2, f3
    """
    inp = "".join(input_tokens)
    tgt = inp + output_token + "</a>"
    return {"input_text": inp, "target_text": tgt}


def form_item_3hop_target(
    h1_idx, h2_idx, h3_idx, h4_idx, t_idx,
    vocab,
    f1_dict, f2_dict, f3_dict,
    cot=False,
    fake_bridge=False,
):
    """
    Creates the final input_text/target_text for a 3-hop sample:
      input: <t_h1><t_h2><t_h3><t_h4>
    
    Then we can produce the final target in one of three ways:
      1) Normal (if not cot and not fake_bridge): single token => <t_final>.
      2) CoT bridging (if cot=True and fake_bridge=False):
         => <t_b1><t_b2><t_final>.
      3) Fake bridging (if fake_bridge=True):
         => </b></b><t_final>. (Ignores the real bridging b1, b2.)
    """
    inp_tokens = [vocab[h1_idx], vocab[h2_idx], vocab[h3_idx], vocab[h4_idx]]
    inp_str = "".join(inp_tokens)
    
    # Actual bridging, if we wanted b1/b2
    b1_idx = f1_dict.get((h1_idx, h2_idx), None)
    b2_idx = None
    if b1_idx is not None:
        b2_idx = f2_dict.get((b1_idx, h3_idx), None)
    
    final_tok = vocab[t_idx]

    if fake_bridge:
        # always produce two </b> tokens before final
        # We appended </b> to the vocab if --fake_bridge is set => find it
        # or just literally use "</b>" if we don't want to parse indices
        # We'll assume the last index is the `</a>` token,
        # so the second-to-last index is `</b>` if you appended them in that order.
        # But simpler: just store it as a literal string:
        b_tok = "</b>"
        bridging_tokens = b_tok + b_tok
    elif cot:
        # produce the real bridging tokens
        if b1_idx is not None and b2_idx is not None:
            b1_tok = vocab[b1_idx]
            b2_tok = vocab[b2_idx]
            bridging_tokens = b1_tok + b2_tok
        else:
            bridging_tokens = ""  # fallback, if OOD
    else:
        # single token
        bridging_tokens = ""

    tgt_str = inp_str + bridging_tokens + final_tok + "</a>"
    return {"input_text": inp_str, "target_text": tgt_str}


def process_item_S_f1(item, S_f2_index, S_f3_index):
    """
    Used for parallel processing to find all 3-hop tuples 
    (h1, h2, h3, h4, t) that can be formed using S_f1, S_f2, S_f3.
    """
    (h1, h2), b1 = item
    partial_set = set()
    if b1 not in S_f2_index:
        return partial_set
    for (h3, b2) in S_f2_index[b1]:
        if b2 not in S_f3_index:
            continue
        for (h4, t) in S_f3_index[b2]:
            partial_set.add((h1, h2, h3, h4, t))
    return partial_set


def coverage_type(sc1: bool, sc2: bool, sc3: bool):
    """
    Determine coverage type based on whether (f1, f2, f3) for the 3-hop 
    have been 'seen' (True) or not (False).
    (1,1,1) => coverage=0, otherwise coverage=1..7
    """
    bits = (sc1 << 2) + (sc2 << 1) + sc3
    return 0 if bits == 7 else bits + 1


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


def parse_3hop_input(inp_text: str):
    """
    e.g. "<t_7><t_11><t_99><t_25>" => [7, 11, 99, 25]
    """
    tokens = inp_text.strip("<>").split("><")
    if len(tokens) != 4:
        return None
    return [int(tok.split("_")[-1]) for tok in tokens]


def reservoir_update(reservoir, tup, total_count, capacity):
    """
    Single-token style storage for coverage reservoir.
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
    parser.add_argument("--same_f123", action="store_true",
                        help="If true, f2 and f3 are duplicates of f1 (same mapping).")
    parser.add_argument("--default_seen_ratio", type=float, default=0.7,
                        help="Ratio of domain considered 'seen' for each function.")
    parser.add_argument("--max_train_data_num", type=int, default=382000,
                        help="Max # of 3-hop samples in the training set.")
    parser.add_argument("--test_size_for_type", type=int, default=3000,
                        help="Number of final test samples for each coverage type.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug-level logging.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--cot", action="store_true",
                        help="If true, produce <b1><b2><t> in final. Otherwise single token.")
    parser.add_argument("--fake_bridge", action="store_true",
                        help="If true, produce </b></b><t> in final. Overrides --cot if both are set.")
    args = parser.parse_args()

    setup_logging(args.debug)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # -------------------------------------------------------
    # 1) Vocab creation
    # -------------------------------------------------------
    # We'll only append </b> if fake_bridge is on
    vocab = [f"<t_{i}>" for i in range(args.num_tokens)]
    if args.fake_bridge:
        vocab.append("</b>")
    # We always have </a> as last
    vocab.append("</a>")

    # So if fake_bridge==True => len(vocab) = args.num_tokens + 2
    # else => len(vocab) = args.num_tokens + 1

    # We'll build functions next
    domain = list(itertools.product(range(args.num_tokens), range(args.num_tokens)))
    codomain = list(range(args.num_tokens))

    # f1
    f1_dict, S_f1 = make_arbitrary_function(domain, codomain, args.default_seen_ratio)
    if args.same_f123:
        f2_dict = dict(f1_dict)
        S_f2 = dict(S_f1)
        f3_dict = dict(f1_dict)
        S_f3 = dict(S_f1)
    else:
        f2_dict, S_f2 = make_arbitrary_function(domain, codomain, args.default_seen_ratio)
        f3_dict, S_f3 = make_arbitrary_function(domain, codomain, args.default_seen_ratio)

    # ------------------------------------------------------
    # 2) Build 'train_inferred'
    # ------------------------------------------------------
    S_f2_index = defaultdict(list)
    for (b1_idx, h3_idx), b2_idx in S_f2.items():
        S_f2_index[b1_idx].append((h3_idx, b2_idx))

    S_f3_index = defaultdict(list)
    for (b2_idx, h4_idx), t_idx in S_f3.items():
        S_f3_index[b2_idx].append((h4_idx, t_idx))

    process_f1 = partial(process_item_S_f1, S_f2_index=S_f2_index, S_f3_index=S_f3_index)
    with mp.Pool(processes=int(mp.cpu_count() * 0.9)) as pool:
        results2 = list(tqdm(pool.imap(process_f1, list(S_f1.items())),
                             total=len(S_f1),
                             desc="Processing 3-hop expansions"))
    seen_expected_inferred_idx = set().union(*results2)
    del results2

    logging.info(f"len(seen_expected_inferred_facts): {len(seen_expected_inferred_idx)}")

    if len(seen_expected_inferred_idx) <= args.max_train_data_num:
        raise Exception(
            "\n\n[Error] All data is in train_inferred. Try smaller --max_train_data_num.\n"
        )
    train_inferred_idx_set = set(random.sample(list(seen_expected_inferred_idx),
                                               args.max_train_data_num))
    del seen_expected_inferred_idx

    S_f1, S_f2, S_f3 = {}, {}, {}
    for (h1_idx, h2_idx, h3_idx, h4_idx, t_idx) in tqdm(train_inferred_idx_set):
        b1_idx = f1_dict[(h1_idx, h2_idx)]
        S_f1[(h1_idx, h2_idx)] = b1_idx

        b2_idx = f2_dict[(b1_idx, h3_idx)]
        S_f2[(b1_idx, h3_idx)] = b2_idx

        S_f3[(b2_idx, h4_idx)] = f3_dict[(b2_idx, h4_idx)]

    OOD_f1 = {k: v for k, v in f1_dict.items() if k not in S_f1}
    OOD_f2 = {k: v for k, v in f2_dict.items() if k not in S_f2}
    OOD_f3 = {k: v for k, v in f3_dict.items() if k not in S_f3}

    logging.info(f"real_seen_f1_set len: {len(S_f1)}")
    logging.info(f"real_seen_f2_set len: {len(S_f2)}")
    logging.info(f"real_seen_f3_set len: {len(S_f3)}")

    # Build final train data
    train_inferred = []
    for (h1_idx, h2_idx, h3_idx, h4_idx, t_idx) in train_inferred_idx_set:
        sample = form_item_3hop_target(
            h1_idx=h1_idx, h2_idx=h2_idx, h3_idx=h3_idx, h4_idx=h4_idx, t_idx=t_idx,
            vocab=vocab,
            f1_dict=S_f1,
            f2_dict=S_f2,
            f3_dict=S_f3,
            cot=(not args.fake_bridge and args.cot),
            fake_bridge=args.fake_bridge
        )
        train_inferred.append(sample)

    # ------------------------------------------------------
    # 3) coverage type samples
    # ------------------------------------------------------
    f2_index = defaultdict(list)
    for (b1_idx, h3_idx), b2_idx in f2_dict.items():
        f2_index[b1_idx].append((h3_idx, b2_idx))

    f3_index = defaultdict(list)
    for (b2_idx, h4_idx), t_idx in f3_dict.items():
        f3_index[b2_idx].append((h4_idx, t_idx))

    coverage_reservoirs = defaultdict(list)
    coverage_seen_count = defaultdict(int)

    skip_p = 0.4
    for (h1_idx, h2_idx), b1_idx in f1_dict.items():
        if random.random() > (1 - skip_p):
            continue
        sc1 = ((h1_idx, h2_idx) in S_f1)

        for (h3_idx, b2_idx) in f2_index.get(b1_idx, []):
            if random.random() > (1 - skip_p):
                continue
            sc2 = ((b1_idx, h3_idx) in S_f2)

            for (h4_idx, t_idx) in f3_index.get(b2_idx, []):
                if random.random() > (1 - skip_p):
                    continue
                sc3 = ((b2_idx, h4_idx) in S_f3)
                c_type = coverage_type(sc1, sc2, sc3)

                coverage_seen_count[f"type_{c_type}"] += 1
                reservoir_update(
                    reservoir=coverage_reservoirs[f"type_{c_type}"],
                    tup=([vocab[h1_idx], vocab[h2_idx], vocab[h3_idx], vocab[h4_idx]], vocab[t_idx]),
                    total_count=coverage_seen_count[f"type_{c_type}"],
                    capacity=args.test_size_for_type
                )

    # ------------------------------------------------------
    # 4) final test data
    # ------------------------------------------------------
    test_data = []
    # (a) some from train => "train_inferred"
    from_train_for_test = choose(train_inferred, args.test_size_for_type)
    for item in from_train_for_test:
        item["type"] = "train_inferred"
        test_data.append(item)

    # (b) coverage=0..7 => re-convert if needed
    coverage_reconverted = []
    for ctype, items in coverage_reservoirs.items():
        for it in items:
            # parse 4 input tokens => h1,h2,h3,h4
            inp_str = it["input_text"]
            out_str = it["target_text"]
            tokens_in = inp_str.strip("<>").split("><")
            if len(tokens_in) != 4:
                continue

            h1_idx = int(tokens_in[0].split("_")[-1])
            h2_idx = int(tokens_in[1].split("_")[-1])
            h3_idx = int(tokens_in[2].split("_")[-1])
            h4_idx = int(tokens_in[3].split("_")[-1])

            tokens_out = out_str.replace("</a>", "").strip("<>").split("><")
            final_tok = tokens_out[-1]
            t_idx = int(final_tok.split("_")[-1])

            # Re-convert
            new_item = form_item_3hop_target(
                h1_idx=h1_idx, h2_idx=h2_idx, h3_idx=h3_idx, h4_idx=h4_idx, t_idx=t_idx,
                vocab=vocab,
                f1_dict=f1_dict,  # use the full dict => can show bridging for OOD
                f2_dict=f2_dict,
                f3_dict=f3_dict,
                cot=(not args.fake_bridge and args.cot),
                fake_bridge=args.fake_bridge
            )
            new_item["type"] = ctype
            coverage_reconverted.append(new_item)
    test_data.extend(coverage_reconverted)

    random.shuffle(train_inferred)
    random.shuffle(test_data)

    # ------------------------------------------------------
    # 5) Atomic facts => single token only
    # ------------------------------------------------------
    atomic_facts_f1 = []
    for (h1_idx, h2_idx), b1_idx in f1_dict.items():
        atomic_facts_f1.append(form_item([vocab[h1_idx], vocab[h2_idx]], vocab[b1_idx]))

    atomic_facts_f2 = []
    for (b1_idx, h3_idx), b2_idx in f2_dict.items():
        atomic_facts_f2.append(form_item([vocab[b1_idx], vocab[h3_idx]], vocab[b2_idx]))

    atomic_facts_f3 = []
    for (b2_idx, h4_idx), t_idx in f3_dict.items():
        atomic_facts_f3.append(form_item([vocab[b2_idx], vocab[h4_idx]], vocab[t_idx]))

    # ------------------------------------------------------
    # 6) save
    # ------------------------------------------------------
    cot_or_fake = ""
    if args.fake_bridge:
        cot_or_fake = "fakebridge"
    elif args.cot:
        cot_or_fake = "cot"
    else:
        cot_or_fake = "inf"

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(
        base_dir, "data",
        f"threehop.{args.num_tokens}.{args.max_train_data_num}."
        f"{'same-f123' if args.same_f123 else 'diff-f123'}.{cot_or_fake}"
    )
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)
    with open(os.path.join(save_dir, "train.json"), "w") as f:
        json.dump(train_inferred, f, indent=2)
    with open(os.path.join(save_dir, "test.json"), "w") as f:
        json.dump(test_data, f, indent=2)
    with open(os.path.join(save_dir, "atomic_facts_f1.json"), "w") as f:
        json.dump(atomic_facts_f1, f, indent=2)
    with open(os.path.join(save_dir, "atomic_facts_f2.json"), "w") as f:
        json.dump(atomic_facts_f2, f, indent=2)
    with open(os.path.join(save_dir, "atomic_facts_f3.json"), "w") as f:
        json.dump(atomic_facts_f3, f, indent=2)

    print("[INFO] Done!")
    print(f"Train size: {len(train_inferred)}")
    print(f"Test size: {len(test_data)}")
    print(f"S_f1: {len(S_f1)}, S_f2: {len(S_f2)}, S_f3: {len(S_f3)}")
    print(f"OOD_f1: {len(OOD_f1)}, OOD_f2: {len(OOD_f2)}, OOD_f3: {len(OOD_f3)}")


if __name__ == "__main__":
    main()
