import json
import numpy as np
import random
import os
from collections import defaultdict

def form_item(input_tokens, output_token):
    """
    Create a dict with:
      input_text = concatenation of input_tokens
      target_text = input_text + output_token + '</a>'
    """
    inp = "".join(input_tokens)
    tgt = inp + output_token + "</a>"
    return {"input_text": inp, "target_text": tgt}

def build_dataset(
    num_tokens=2000,
    b1_size=200,
    b2_size=200,
    num_pairs_f1_train=4000,
    num_pairs_f1_test=1000,
    num_pairs_f2_train=4000,
    num_pairs_f2_test=1000,
    held_out_fraction=0.05,      # fraction of (b1,b2) pairs to hold out => ensures type_3
    train_ratio_per_b1b2=0.7,    # fraction of 4-tuples for each (b1,b2) that go to train
    max_notcov_out=5000,         # limit how many not-covered examples we keep
    seed=0,
    outdir="data/many_to_one_types_with_type0",
    same_f12=False               # <<<<< NEW PARAM: if True => f1 == f2
):
    """
    Builds a multi-hop (parallel+hierarchical) dataset with coverage-based splits:
      - train.json
      - test.json (containing both covered and not-covered examples)
        * "type_0" for covered
        * "type_1".."type_5" for not-covered

    If 'same_f12=True', then f1 and f2 share the *identical* function mapping,
    i.e. for any pair (x,y), the same dictionary is used for both subcomp1 and subcomp2.

    Steps:
      1) Define partial f1, f2 for train vs. test domains (S_f1/T_f1, S_f2/T_f2),
         or unify them into one function if same_f12=True.
      2) From (S_f1 x S_f2), pick some (b1,b2) to hold out => subcomp3 not in training => ensures type_3.
      3) For the remaining (b1,b2), do random-split => train vs covered.
      4) Build "not-covered" from everything else. 
      5) Merge covered & not-covered into a single test set, with different "type" fields.
    """

    np.random.seed(seed)
    random.seed(seed)

    vocab = [f"<t_{i}>" for i in range(num_tokens)]

    # -------------
    # Step A: define partial f1, f2
    #        possibly unify them if same_f12=True
    # -------------
    def sample_pairs(num_pairs):
        s = set()
        out = []
        while len(out)<num_pairs:
            x = np.random.randint(num_tokens)
            y = np.random.randint(num_tokens)
            if (x,y) not in s:
                s.add((x,y))
                out.append((x,y))
        return out

    # S_f1, T_f1: disjoint sets
    S_f1 = sample_pairs(num_pairs_f1_train)
    T_f1 = sample_pairs(num_pairs_f1_test)
    # S_f2, T_f2: disjoint sets
    S_f2 = sample_pairs(num_pairs_f2_train)
    T_f2 = sample_pairs(num_pairs_f2_test)

    B1_train = list(range(b1_size))
    B2_train = list(range(b2_size))
    B1_test  = B1_train
    B2_test  = B2_train

    # We'll fill f1, f2 below. Possibly share them if same_f12=True
    f1 = {}
    f2 = {}

    if not same_f12:
        # --- Original approach: define f1, f2 separately
        for (h1,h2) in S_f1:
            f1[(h1,h2)] = random.choice(B1_train)
        for (h1,h2) in T_f1:
            f1[(h1,h2)] = random.choice(B1_test)

        for (h3,h4) in S_f2:
            f2[(h3,h4)] = random.choice(B2_train)
        for (h3,h4) in T_f2:
            f2[(h3,h4)] = random.choice(B2_test)

    else:
        # --- If same_f12=True: define a single function f12 for *all* pairs
        # So we gather the union of all subcomp1 + subcomp2 pairs
        # so that we can define a single function on them.
        union_f12_train = S_f1 + S_f2
        union_f12_test  = T_f1 + T_f2

        f12 = {}

        # For train domain pairs => map to B1_train (or B2_train, but it's the same set)
        for (x,y) in union_f12_train:
            if (x,y) not in f12:
                f12[(x,y)] = random.choice(B1_train)  # many->one style

        # For test domain pairs => map to B1_test
        for (x,y) in union_f12_test:
            if (x,y) not in f12:
                f12[(x,y)] = random.choice(B1_test)

        # Now set f1, f2 to the same dictionary f12
        for (h1,h2) in (S_f1 + T_f1):
            f1[(h1,h2)] = f12[(h1,h2)]
        for (h3,h4) in (S_f2 + T_f2):
            f2[(h3,h4)] = f12[(h3,h4)]

    # f3 for the final composition (b1,b2)-> t
    f3 = {}

    # ----------------------------------------------------
    # Step B: gather (b1,b2) in the train domain & hold out fraction => type_3
    # ----------------------------------------------------
    train_b1b2_dict = defaultdict(list)
    all_b1b2_train_domain = set()
    for (h1,h2) in S_f1:
        b1 = f1[(h1,h2)]
        for (h3,h4) in S_f2:
            b2 = f2[(h3,h4)]
            all_b1b2_train_domain.add((b1,b2))
            train_b1b2_dict[(b1,b2)].append((h1,h2,h3,h4))

    all_b1b2_train_domain = list(all_b1b2_train_domain)
    random.shuffle(all_b1b2_train_domain)
    held_out_count = int(round(held_out_fraction * len(all_b1b2_train_domain)))
    held_out_b1b2 = set(all_b1b2_train_domain[:held_out_count])
    allowed_b1b2  = set(all_b1b2_train_domain[held_out_count:])

    train_b1b2_set = set()  # which (b1,b2) actually appear in train

    # containers
    train_inferred = []
    covered_inferred_temp = []  # we'll label them "type_0"

    # Step B.1: For each (b1,b2) in allowed_b1b2 => random-split
    for (b1,b2) in allowed_b1b2:
        quadruples = train_b1b2_dict[(b1,b2)]
        t = np.random.randint(num_tokens)
        f3[(b1,b2)] = t

        random.shuffle(quadruples)
        cutoff = int(round(train_ratio_per_b1b2*len(quadruples)))
        train_quads = quadruples[:cutoff]
        covered_quads = quadruples[cutoff:]

        if len(train_quads)>0:
            train_b1b2_set.add((b1,b2))

        for (h1,h2,h3,h4) in train_quads:
            item = form_item([vocab[h1],vocab[h2],vocab[h3],vocab[h4]], vocab[t])
            train_inferred.append(item)

        for (h1,h2,h3,h4) in covered_quads:
            item = form_item([vocab[h1],vocab[h2],vocab[h3],vocab[h4]], vocab[t])
            # label as type_0
            item["type"] = "type_0"
            covered_inferred_temp.append(item)

    # Step B.2: For each (b1,b2) in held_out_b1b2 => define random t 
    # but do not place them in train or covered
    for (b1,b2) in held_out_b1b2:
        quadruples = train_b1b2_dict[(b1,b2)]
        t = np.random.randint(num_tokens)
        f3[(b1,b2)] = t
        # subcomp3 is never used => leads to type_3 in not_covered

    # -----------------------------
    # Step C: build the not-covered domain 
    # from (S_f1 ∪ T_f1) x (S_f2 ∪ T_f2), 
    # exclude train & covered quadruples
    # -----------------------------
    train_4set = set()
    covered_4set = set()

    # for (b1,b2) in allowed_b1b2, replicate the random-split
    for (b1,b2) in allowed_b1b2:
        quadruples = train_b1b2_dict[(b1,b2)]
        random.shuffle(quadruples)
        cutoff = int(round(train_ratio_per_b1b2*len(quadruples)))
        tr = quadruples[:cutoff]
        cv = quadruples[cutoff:]
        for q in tr:
            train_4set.add(q)
        for q in cv:
            covered_4set.add(q)

    # now gather all leftover combos
    S_f1_union = set(S_f1)|set(T_f1)
    S_f2_union = set(S_f2)|set(T_f2)
    all_notcov_candidates = []
    for (h1,h2) in S_f1_union:
        b1 = f1[(h1,h2)]
        for (h3,h4) in S_f2_union:
            b2 = f2[(h3,h4)]
            if (h1,h2,h3,h4) not in train_4set and (h1,h2,h3,h4) not in covered_4set:
                all_notcov_candidates.append((h1,h2,h3,h4,b1,b2))

    random.shuffle(all_notcov_candidates)
    all_notcov_candidates = all_notcov_candidates[:max_notcov_out]

    test_not_covered_temp = []

    def subcomp_covered(h1,h2,h3,h4,b1,b2):
        sc1 = ((h1,h2) in S_f1)
        sc2 = ((h3,h4) in S_f2)
        sc3 = ((b1,b2) in train_b1b2_set)
        return sc1, sc2, sc3


    # ----------------------------------------------------------------------
    # Explanation of Each "type"
    # ----------------------------------------------------------------------
    #
    # "type_0": (Covered)
    #   - All three subcomputations are covered by the training set:
    #       subcomp1 = (h1,h2)
    #       subcomp2 = (h3,h4)
    #       subcomp3 = (b1,b2)
    #     Hence, the final 4-token input is "in-coverage" (i.e., known subcomputations).
    #
    # "type_1": (Not-Covered)
    #   - Exactly one of subcomp1 or subcomp2 is covered
    #     (so either (h1,h2) or (h3,h4) appears in training, but not both),
    #     AND subcomp3 = (b1,b2) is also covered in training.
    #
    # "type_2": (Not-Covered)
    #   - Neither subcomp1 nor subcomp2 is covered,
    #     BUT subcomp3 = (b1,b2) is covered.
    #
    # "type_3": (Not-Covered)
    #   - Both subcomp1 and subcomp2 are covered individually,
    #     BUT subcomp3 = (b1,b2) never appears in training.
    #     (This usually arises when b1, b2 each appear separately in training
    #      but the pair (b1,b2) is "held out".)
    #
    # "type_4": (Not-Covered)
    #   - Exactly one of subcomp1 or subcomp2 is covered,
    #     AND subcomp3 is NOT covered.
    #
    # "type_5": (Not-Covered)
    #   - Neither subcomp1 nor subcomp2 is covered,
    #     AND subcomp3 is also not covered.
    #
    # By default, "type_0" means "covered," and "type_1" through "type_5"
    # indicate various ways that coverage can fail.

    def coverage_failure_type(sc1, sc2, sc3):
        """
        Returns 1..5 or 0 if fully covered.
        """
        c_atomic = sum([sc1, sc2])
        if sc3:
            # subcomp3 covered
            if c_atomic==1:
                return 1
            elif c_atomic==0:
                return 2
            # if c_atomic==2 => fully covered => skip
            return 0
        else:
            # subcomp3 not covered
            if c_atomic==2:
                return 3
            elif c_atomic==1:
                return 4
            elif c_atomic==0:
                return 5
            return 0

    for (h1,h2,h3,h4,b1,b2) in all_notcov_candidates:
        if (b1,b2) not in f3:
            f3[(b1,b2)] = np.random.randint(num_tokens)
        t = f3[(b1,b2)]
        sc1, sc2, sc3 = subcomp_covered(h1,h2,h3,h4,b1,b2)
        ctype = coverage_failure_type(sc1, sc2, sc3)
        if ctype==0:
            # means fully covered => skip
            continue
        item = form_item(
            [vocab[h1],vocab[h2],vocab[h3],vocab[h4]],
            vocab[t]
        )
        item["type"] = f"type_{ctype}"
        test_not_covered_temp.append(item)

    # -----------------------------
    # Step D: unify covered + not-covered => test
    # covered_inferred_temp => type_0
    # test_not_covered_temp => type_1..5
    # then store them together in test.json
    # -----------------------------
    test_data = covered_inferred_temp + test_not_covered_temp
    random.shuffle(test_data)

    # Prepare final "train_inferred" + "test_data"
    random.shuffle(train_inferred)

    # -----------------------------
    # Step E: atomic facts for analysis
    # -----------------------------
    atomic_facts_1 = []
    for (h1,h2), b1val in f1.items():
        atomic_facts_1.append(form_item([vocab[h1],vocab[h2]], vocab[b1val]))
    atomic_facts_2 = []
    for (h3,h4), b2val in f2.items():
        atomic_facts_2.append(form_item([vocab[h3],vocab[h4]], vocab[b2val]))
    atomic_facts_3 = []
    for (b1,b2), tval in f3.items():
        atomic_facts_3.append(form_item([vocab[b1],vocab[b2]], vocab[tval]))

    # -----------------------------
    # Step F: Save
    # -----------------------------
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    with open(os.path.join(outdir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_inferred, f, indent=2)

    with open(os.path.join(outdir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)

    with open(os.path.join(outdir, "atomic_facts_1.json"), "w", encoding="utf-8") as f:
        json.dump(atomic_facts_1, f, indent=2)
    with open(os.path.join(outdir, "atomic_facts_2.json"), "w", encoding="utf-8") as f:
        json.dump(atomic_facts_2, f, indent=2)
    with open(os.path.join(outdir, "atomic_facts_3.json"), "w", encoding="utf-8") as f:
        json.dump(atomic_facts_3, f, indent=2)

    # -----------------------------
    # Step G: Print Stats
    # -----------------------------
    # Among test_data, some have type_0 (covered), some have type_1..5
    type_counts = {}
    for item in test_data:
        tstr = item["type"]
        type_counts[tstr] = type_counts.get(tstr, 0) + 1

    total_test = len(test_data)
    print(f"\nDataset in '{outdir}':")
    print(f"  Train: {len(train_inferred)}")
    print(f"  Test:  {len(test_data)} (single file with type=0..5)\n")
    print("  Test breakdown by type:")
    for tkey in sorted(type_counts.keys()):
        cnt = type_counts[tkey]
        pct = 100.0*cnt/(total_test+1e-9)
        print(f"    {tkey}: {cnt} ({pct:.1f}%)")
    print()
    print(f"  atomic_facts_1: {len(atomic_facts_1)}")
    print(f"  atomic_facts_2: {len(atomic_facts_2)}")
    print(f"  atomic_facts_3: {len(atomic_facts_3)}")
    print(f"\n  (same_f12={same_f12})\n")


# ----------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------
if __name__=="__main__":
    """
    Example: set `same_f12=True` if you want f1 == f2.
    Otherwise, they remain separate functions.
    """
    # build_dataset(
    #     num_tokens=200, 
    #     b1_size=50,
    #     b2_size=50,
    #     num_pairs_f1_train=300,
    #     num_pairs_f1_test=100,
    #     num_pairs_f2_train=300,
    #     num_pairs_f2_test=100,
    #     held_out_fraction=0.10,
    #     train_ratio_per_b1b2=0.7,
    #     max_notcov_out=3000,
    #     seed=0,
    #     outdir="test_same_f12_false",
    #     same_f12=False   # f1 != f2
    # )

    build_dataset(
        num_tokens=200, 
        b1_size=50,
        b2_size=50,
        num_pairs_f1_train=300,
        num_pairs_f1_test=100,
        num_pairs_f2_train=300,
        num_pairs_f2_test=100,
        held_out_fraction=0.10,
        train_ratio_per_b1b2=0.7,
        max_notcov_out=3000,
        seed=42,
        outdir="same_f12",
        same_f12=True    # f1 == f2
    )
# ----------------------
# Example usage
# ----------------------
# if __name__=="__main__":
#     build_dataset(
#         num_tokens=200, 
#         b1_size=200,
#         b2_size=200,
#         num_pairs_f1_train=100,
#         num_pairs_f1_test=100,
#         num_pairs_f2_train=100,
#         num_pairs_f2_test=100,
#         held_out_fraction=0.15,  # ~8% of (b1,b2) in train domain => never used => type_3
#         train_ratio_per_b1b2=0.7,
#         max_notcov_out=30000,
#         seed=0,
#         outdir="test"
#     )