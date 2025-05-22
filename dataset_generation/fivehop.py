#!/usr/bin/env python
"""fivehop_uniform.py – efficient, fully-uniform sampling of in-domain (coverage-0)
COT five-hop chains.

Highlights
──────────
• **Uniform** over the *exact* ID space without materialising it.
• **Linear memory** – stores only S₁…S₅ and lightweight per-node cumulative
  arrays.
• Two test splits only: `train_inferred` and `type_0` (ID but disjoint from
  training).
• Default test size per split = **2 000**.
• Keeps the single global RNG seeding logic you requested.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
from bisect import bisect_left
from collections import defaultdict
from typing import Dict, List, Tuple, Sequence, Any, Set

import numpy as np
from tqdm import trange

# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def setup_logging(debug: bool):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def make_arbitrary_function(domain: Sequence[Tuple[int, int]], codomain: Sequence[int], seen_ratio: float):
    """Return a full mapping plus its *seen* subset (S_dict)."""
    mapping: Dict[Tuple[int, int], int] = {
        k: v for k, v in zip(domain, random.choices(codomain, k=len(domain)))
    }
    keys = list(mapping.keys())
    random.shuffle(keys)
    cutoff = int(round(len(keys) * seen_ratio))
    S = {k: mapping[k] for k in keys[:cutoff]}
    return mapping, S


def form_item(inp_tokens: List[str], out_token: str):
    inp = "".join(inp_tokens)
    return {"input_text": inp, "target_text": inp + out_token + "</a>"}

# -----------------------------------------------------------------------------
# Cumulative-sampling utilities
# -----------------------------------------------------------------------------

def cumulative_from_weights(ws: List[int]):
    cum, s = [], 0
    for w in ws:
        s += w
        cum.append(s)
    return cum


def choose_cumulative(items: Sequence[Any], cumulative: Sequence[int]):
    r = random.randint(1, cumulative[-1])
    idx = bisect_left(cumulative, r)
    return items[idx]

# -----------------------------------------------------------------------------
# Uniform five-hop sampler
# -----------------------------------------------------------------------------

class UniformFiveHopSampler:
    """Draws ID five-hop tuples uniformly without enumerating the set."""

    def __init__(
        self,
        S1: Dict[Tuple[int, int], int],
        S2: Dict[Tuple[int, int], int],
        S3: Dict[Tuple[int, int], int],
        S4: Dict[Tuple[int, int], int],
        S5: Dict[Tuple[int, int], int],
    ):
        # Build child indices for hops 2-5
        self.idx2 = defaultdict(list)  # b1 → [(h3, b2)]
        for (b1, h3), b2 in S2.items():
            self.idx2[b1].append((h3, b2))

        self.idx3 = defaultdict(list)  # b2 → [(h4, b3)]
        for (b2, h4), b3 in S3.items():
            self.idx3[b2].append((h4, b3))

        self.idx4 = defaultdict(list)  # b3 → [(h5, b4)]
        for (b3, h5), b4 in S4.items():
            self.idx4[b3].append((h5, b4))

        self.idx5 = defaultdict(list)  # b4 → [(h6, t)]
        for (b4, h6), t in S5.items():
            self.idx5[b4].append((h6, t))

        # Bottom-up completion counts ----------------------------------------
        c5 = {b4: len(v) for b4, v in self.idx5.items()}  # each tuple is terminal

        self.idx4_cum = {}
        c4_sum = {}
        for b3, lst in self.idx4.items():
            w = [c5[b4] for (_, b4) in lst]
            cum = cumulative_from_weights(w)
            self.idx4_cum[b3] = (lst, cum)
            c4_sum[b3] = cum[-1]

        self.idx3_cum = {}
        c3_sum = {}
        for b2, lst in self.idx3.items():
            w = [c4_sum[b3] for (_, b3) in lst]
            cum = cumulative_from_weights(w)
            self.idx3_cum[b2] = (lst, cum)
            c3_sum[b2] = cum[-1]

        self.idx2_cum = {}
        c2_sum = {}
        for b1, lst in self.idx2.items():
            w = [c3_sum[b2] for (_, b2) in lst]
            cum = cumulative_from_weights(w)
            self.idx2_cum[b1] = (lst, cum)
            c2_sum[b1] = cum[-1]

        # Hop-1 cumulative across all (h1, h2)
        self.h12_list = list(S1.keys())
        w1 = [c2_sum[S1[(h1, h2)]] for (h1, h2) in self.h12_list]
        self.h12_cum = cumulative_from_weights(w1)
        self.S1 = S1

    # ---------------------------------------------------------------------
    def sample(self) -> Tuple[int, int, int, int, int, int, int]:
        # Hop-1
        h1, h2 = choose_cumulative(self.h12_list, self.h12_cum)
        b1 = self.S1[(h1, h2)]

        # Hop-2
        lst2, cum2 = self.idx2_cum[b1]
        h3, b2 = choose_cumulative(lst2, cum2)

        # Hop-3
        lst3, cum3 = self.idx3_cum[b2]
        h4, b3 = choose_cumulative(lst3, cum3)

        # Hop-4
        lst4, cum4 = self.idx4_cum[b3]
        h5, b4 = choose_cumulative(lst4, cum4)

        # Hop-5 (uniform)
        h6, t = random.choice(self.idx5[b4])

        return h1, h2, h3, h4, h5, h6, t

# -----------------------------------------------------------------------------
# Target-string builder (COT always-on)
# -----------------------------------------------------------------------------

def form_item_5hop_target(
    h1: int,
    h2: int,
    h3: int,
    h4: int,
    h5: int,
    h6: int,
    t: int,
    vocab: List[str],
    f1: Dict[Tuple[int, int], int],
    f2: Dict[Tuple[int, int], int],
    f3: Dict[Tuple[int, int], int],
    f4: Dict[Tuple[int, int], int],
):
    inp = (
        vocab[h1]
        + vocab[h2]
        + vocab[h3]
        + vocab[h4]
        + vocab[h5]
        + vocab[h6]
    )
    b1 = vocab[f1[(h1, h2)]]
    b2 = vocab[f2[(f1[(h1, h2)], h3)]]
    b3 = vocab[f3[(f2[(f1[(h1, h2)], h3)], h4)]]
    b4 = vocab[f4[(f3[(f2[(f1[(h1, h2)], h3)], h4)], h5)]]
    tgt = inp + b1 + b2 + b3 + b4 + vocab[t] + "</a>"
    return {"input_text": inp, "target_text": tgt}

# -----------------------------------------------------------------------------
# Atomic facts helper
# -----------------------------------------------------------------------------

def emit_atomic(S: Dict[Tuple[int, int], int], vocab: List[str]):
    return [form_item([vocab[i], vocab[j]], vocab[o]) for (i, j), o in S.items()]

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_tokens", type=int, required=True)
    ap.add_argument("--max_train_data_num", type=int, default=600_000)
    ap.add_argument("--test_size_for_type", type=int, default=2000)
    ap.add_argument("--default_seen_ratio", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    setup_logging(args.debug)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ----------------------------- Vocabulary ------------------------------
    vocab = [f"<t_{i}>" for i in range(args.num_tokens)] + ["</a>"]

    # ------------------------ Sample functions f1-f5 -----------------------
    domain = list(itertools.product(range(args.num_tokens), range(args.num_tokens)))
    codomain = list(range(args.num_tokens))

    f1, S1 = make_arbitrary_function(domain, codomain, args.default_seen_ratio)
    f2, S2 = make_arbitrary_function(domain, codomain, args.default_seen_ratio)
    f3, S3 = make_arbitrary_function(domain, codomain, args.default_seen_ratio)
    f4, S4 = make_arbitrary_function(domain, codomain, args.default_seen_ratio)
    f5, S5 = make_arbitrary_function(domain, codomain, args.default_seen_ratio)

    # ------------------------- Uniform sampler -----------------------------
    sampler = UniformFiveHopSampler(S1, S2, S3, S4, S5)

    # ------------------------ Generate training set ------------------------
    train_tuples: Set[Tuple[int, ...]] = set()
    while len(train_tuples) < args.max_train_data_num:
        train_tuples.add(sampler.sample())

    train_inferred = [
        form_item_5hop_target(*tpl, vocab, f1, f2, f3, f4) for tpl in train_tuples
    ]

    # ------------------------ Generate type_0 test set ----------------------
    type0_tuples: Set[Tuple[int, ...]] = set()
    while len(type0_tuples) < args.test_size_for_type:
        tpl = sampler.sample()
        if tpl not in train_tuples:
            type0_tuples.add(tpl)

    type0_samples = [
        {**form_item_5hop_target(*tpl, vocab, f1, f2, f3, f4), "type": "type_0"}
        for tpl in type0_tuples
    ]

    # ------------------------ Select train slice for test -------------------
    train_slice = random.sample(train_inferred, args.test_size_for_type)
    for s in train_slice:
        s["type"] = "train_inferred"

    test_data = train_slice + type0_samples
    random.shuffle(test_data)

    # ---------------------------- Atomic facts ------------------------------
    atomic_facts_f1 = emit_atomic(S1, vocab)
    atomic_facts_f2 = emit_atomic(S2, vocab)
    atomic_facts_f3 = emit_atomic(S3, vocab)
    atomic_facts_f4 = emit_atomic(S4, vocab)
    atomic_facts_f5 = emit_atomic(S5, vocab)

    # ------------------------------- Save -----------------------------------
    save_dir = os.path.join(
        "data",
        f"fivehop.{args.num_tokens}.{args.max_train_data_num}.cot",
    )
    os.makedirs(save_dir, exist_ok=True)

    def _dump(name: str, obj: Any):
        with open(os.path.join(save_dir, name), "w") as f:
            json.dump(obj, f, indent=2)

    _dump("vocab.json", vocab)
    _dump("train.json", train_inferred)
    _dump("test.json", test_data)
    _dump("atomic_facts_f1.json", atomic_facts_f1)
    _dump("atomic_facts_f2.json", atomic_facts_f2)
    _dump("atomic_facts_f3.json", atomic_facts_f3)
    _dump("atomic_facts_f4.json", atomic_facts_f4)
    _dump("atomic_facts_f5.json", atomic_facts_f5)

    # --------------------------- Diagnostics --------------------------------
    print("[INFO] Done!")
    print(f"Train size: {len(train_inferred)} | Test size: {len(test_data)}")
    print(f"type_0 size: {len(type0_samples)} | train-slice: {len(train_slice)}")


if __name__ == "__main__":
    main()