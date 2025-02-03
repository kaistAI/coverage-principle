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

def build_nontree_dag_dataset(
    num_tokens=2000,
    # how many (e1,e2) in train vs test for f1
    num_pairs_f1_train=4000,
    num_pairs_f1_test=1000,
    # how many (b1,e2,e3) in train vs test for f2
    num_triples_f2_train=4000,
    num_triples_f2_test=1000,

    # how many random (e1,e2,e3) combos we try for the train domain
    num_train_triplets=20000,
    # how many random (e1,e2,e3) combos for the not-covered domain
    num_notcov_triplets=30000,

    # fraction of train domain combos that go to "train" vs "covered test"
    train_ratio=0.7,

    seed=0,
    max_not_covered=5000,
    outdir="data/nontree_dag"
):
    """
    Non-tree DAG dataset with 2 subcomputations:
      1) b1 = f1(e1, e2)
      2) t  = f2(b1, e2, e3)

    We define:
      S_f1, T_f1 => partial coverage for subcomp1
      S_f2, T_f2 => partial coverage for subcomp2

    'Train domain' combos => 
      (e1,e2) in S_f1 => b1
      (b1,e2,e3) in S_f2 => t
    => Then random-split => train vs covered test.

    Everything else => not-covered test => type_1..3, depending on coverage bits sc1, sc2.
    """

    np.random.seed(seed)
    random.seed(seed)

    vocab = [f"<t_{i}>" for i in range(num_tokens)]

    # -------------------------------------------
    # Step A: define partial f1, f2
    # -------------------------------------------
    # subcomp1: (e1,e2)->b1
    # subcomp2: (b1,e2,e3)->t

    def sample_pairs(num_pairs):
        """Pick 'num_pairs' distinct (x,y) in [0..num_tokens)^2."""
        s = set()
        out = []
        while len(out) < num_pairs:
            x = np.random.randint(num_tokens)
            y = np.random.randint(num_tokens)
            if (x,y) not in s:
                s.add((x,y))
                out.append((x,y))
        return out

    def sample_triples(num_triples):
        """
        Pick 'num_triples' distinct (x,y,z) in [0..num_tokens)^3.
        """
        s = set()
        out = []
        while len(out) < num_triples:
            x = np.random.randint(num_tokens)
            y = np.random.randint(num_tokens)
            z = np.random.randint(num_tokens)
            if (x,y,z) not in s:
                s.add((x,y,z))
                out.append((x,y,z))
        return out

    # subcomp1 domain
    S_f1_pairs = sample_pairs(num_pairs_f1_train)
    T_f1_pairs = sample_pairs(num_pairs_f1_test)

    # subcomp2 domain
    S_f2_triples = sample_triples(num_triples_f2_train)
    T_f2_triples = sample_triples(num_triples_f2_test)

    # We'll store them in sets for coverage checks
    S_f1_set = set(S_f1_pairs)
    S_f2_set = set(S_f2_triples)

    # create f1 => random b1 for each pair
    f1 = {}
    for (e1,e2) in S_f1_pairs:
        b1 = np.random.randint(num_tokens)
        f1[(e1,e2)] = b1
    for (e1,e2) in T_f1_pairs:
        b1 = np.random.randint(num_tokens)
        f1[(e1,e2)] = b1

    # create f2 => random t for each triple
    f2 = {}
    for (b1,e2,e3) in S_f2_triples:
        t = np.random.randint(num_tokens)
        f2[(b1,e2,e3)] = t
    for (b1,e2,e3) in T_f2_triples:
        t = np.random.randint(num_tokens)
        f2[(b1,e2,e3)] = t

    # -------------------------------------------
    # Step B: build train domain combos => (e1,e2,e3)-> t
    # -------------------------------------------
    train_domain_dict = defaultdict(list)
    final_map = {}

    # We'll do random sampling of e1,e2,e3
    for _ in range(num_train_triplets):
        e1 = np.random.randint(num_tokens)
        e2 = np.random.randint(num_tokens)
        e3 = np.random.randint(num_tokens)
        # subcomp1 => (e1,e2) => b1 if in f1
        if (e1,e2) in f1:
            b1 = f1[(e1,e2)]
            # subcomp2 => (b1,e2,e3) => t if in f2
            if (b1,e2,e3) in f2:
                t = f2[(b1,e2,e3)]
                # We'll group them by (b1,e2,e3) or something
                train_domain_dict[(b1,e2,e3)].append((e1,e2,e3))
                final_map[(e1,e2,e3)] = t

    # random-split => train vs covered
    train_inferred = []
    covered_inferred = []
    train_b1e2e3_set = set()

    for (b1,e2,e3), triple_list in train_domain_dict.items():
        if len(triple_list)==0:
            continue
        random.shuffle(triple_list)
        cutoff = int(round(train_ratio * len(triple_list)))
        tr = triple_list[:cutoff]
        cv = triple_list[cutoff:]
        # if we have at least one => subcomp2 covered
        if len(tr)>0:
            train_b1e2e3_set.add((b1,e2,e3))

        for (e1,e2v,e3v) in tr:
            t = final_map[(e1,e2v,e3v)]
            item = form_item(
                [vocab[e1], vocab[e2v], vocab[e3v]],
                vocab[t]
            )
            train_inferred.append(item)

        for (e1,e2v,e3v) in cv:
            t = final_map[(e1,e2v,e3v)]
            item = form_item(
                [vocab[e1], vocab[e2v], vocab[e3v]],
                vocab[t]
            )
            # type_0 => fully covered
            item["type"] = "type_0"
            covered_inferred.append(item)

    # -------------------------------------------
    # Step C: build not-covered set
    # -------------------------------------------
    used_train_covered = set()
    for (b1,e2,e3), triple_list in train_domain_dict.items():
        random.shuffle(triple_list)
        cutoff = int(round(train_ratio * len(triple_list)))
        tr = triple_list[:cutoff]
        cv = triple_list[cutoff:]
        for q in (tr+cv):
            used_train_covered.add(q)

    # now sample (e1,e2,e3) again for not-covered domain
    notcov_candidates = []
    for _ in range(num_notcov_triplets):
        e1 = np.random.randint(num_tokens)
        e2 = np.random.randint(num_tokens)
        e3 = np.random.randint(num_tokens)
        if (e1,e2,e3) not in used_train_covered:
            notcov_candidates.append((e1,e2,e3))

    random.shuffle(notcov_candidates)
    notcov_candidates = notcov_candidates[:max_not_covered]

    def covered_subcomp1(e1,e2):
        return ((e1,e2) in S_f1_set)
    def covered_subcomp2(b1,e2,e3):
        return ((b1,e2,e3) in S_f2_set)

    def coverage_failure_type(sc1, sc2):
        """
        bits (sc1, sc2) => { (0,0),(0,1),(1,0),(1,1) } => type_3..type_0
        We'll define:
          (1,1) => type_0 (fully covered)
          (0,1) => type_1
          (1,0) => type_2
          (0,0) => type_3
        """
        if sc1 and sc2:
            # fully covered => skip if we're building not-covered
            return 0
        elif (not sc1) and sc2:
            return 1
        elif sc1 and (not sc2):
            return 2
        else:
            return 3

    test_not_covered = []

    for (e1,e2,e3) in notcov_candidates:
        # define or reuse subcomp1 => if (e1,e2) in f1 => b1
        if (e1,e2) in f1:
            b1 = f1[(e1,e2)]
        else:
            # define random b1? We'll do it for the sake of a label
            b1 = np.random.randint(num_tokens)
        # define or reuse subcomp2 => if (b1,e2,e3) in f2 => t
        if (b1,e2,e3) in f2:
            t = f2[(b1,e2,e3)]
        else:
            t = np.random.randint(num_tokens)

        sc1 = covered_subcomp1(e1,e2)
        sc2 = covered_subcomp2(b1,e2,e3)
        ctype = coverage_failure_type(sc1, sc2)
        if ctype==0:
            # means fully covered => skip
            continue
        item = form_item(
            [vocab[e1], vocab[e2], vocab[e3]],
            vocab[t]
        )
        item["type"] = f"type_{ctype}"
        test_not_covered.append(item)

    # unify test
    test_data = covered_inferred + test_not_covered
    random.shuffle(test_data)

    # shuffle train
    random.shuffle(train_inferred)

    # atomic facts for analysis
    atomic_facts_1 = []
    for (e1,e2), b1 in f1.items():
        atomic_facts_1.append(form_item([vocab[e1], vocab[e2]], vocab[b1]))

    atomic_facts_2 = []
    for (b1,e2,e3), tval in f2.items():
        atomic_facts_2.append(form_item([vocab[b1], vocab[e2], vocab[e3]], vocab[tval]))

    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)

    with open(os.path.join(outdir, "train.json"), "w") as f:
        json.dump(train_inferred, f, indent=2)

    with open(os.path.join(outdir, "test.json"), "w") as f:
        json.dump(test_data, f, indent=2)

    with open(os.path.join(outdir, "atomic_facts_1.json"), "w") as f:
        json.dump(atomic_facts_1, f, indent=2)
    with open(os.path.join(outdir, "atomic_facts_2.json"), "w") as f:
        json.dump(atomic_facts_2, f, indent=2)

    # stats
    type_counts = {}
    for item in test_data:
        tstr = item["type"]
        type_counts[tstr] = type_counts.get(tstr, 0)+1

    total_test = len(test_data)
    print(f"\nNon-tree DAG dataset in '{outdir}':")
    print(f"  Train: {len(train_inferred)} (3->1 examples)")
    print(f"  Test:  {len(test_data)}  (type_0..3)\n")
    print("  Test breakdown by type:")
    for k in sorted(type_counts.keys()):
        c = type_counts[k]
        pct = 100.*c/(total_test+1e-9)
        print(f"    {k}: {c} ({pct:.1f}%)")
    print()
    print(f"  atomic_facts_1: {len(atomic_facts_1)}")
    print(f"  atomic_facts_2: {len(atomic_facts_2)}")


# ------------------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------------------
if __name__=="__main__":
    build_nontree_dag_dataset(
        num_tokens=50,               # we keep 50 so collisions are not too sparse
        num_pairs_f1_train=2000,     # cover many (e1,e2) pairs for subcomp1
        num_pairs_f1_test=200,       
        num_triples_f2_train=20000,  # huge coverage for (b1,e2,e3) in subcomp2
        num_triples_f2_test=1000,    
        num_train_triplets=500000,   # massive random sampling => many collisions
        num_notcov_triplets=500000,  # similarly large for not-covered
        train_ratio=0.8,            # push 80% collisions into train
        seed=42,
        max_not_covered=50000,       # keep up to 50k not-covered
        outdir="data/nontree_dag_demo"
    )