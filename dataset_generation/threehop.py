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

def build_3hop_dataset(
    num_tokens=2000,
    # We pick how many (h1,h2) we define as train vs test for subcomp1
    num_pairs_f1_train=4000,
    num_pairs_f1_test=1000,
    # how many (b1,h3) we define as train vs test for subcomp2
    num_pairs_f2_train=4000,
    num_pairs_f2_test=1000,
    # how many (b2,h4) for subcomp3
    num_pairs_f3_train=4000,
    num_pairs_f3_test=1000,

    # fraction of each subcomp's 4-token combos that go to actual train vs covered
    train_ratio=0.7,

    # how many extra not-covered combos we keep
    max_not_covered=5000,

    seed=0,
    outdir="data/3hop_hier"
):
    """
    Build a 3-hop hierarchical dataset:

      1) b1 = f1(h1,h2)
      2) b2 = f2(b1,h3)
      3) t  = f3(b2,h4)

    We'll store:
      - train.json        -> 4->1 examples
      - test.json         -> 4->1 examples
         * "type_0"  => fully covered (subcomp1, subcomp2, subcomp3 all in train domain)
         * "type_1..type_7" => partial coverage failure

    We also store atomic_facts_1,2,3 for analysis of each sub-func.

    Coverage logic:
      subcomp1 covered if (h1,h2) in S_f1
      subcomp2 covered if (b1,h3) in S_f2
      subcomp3 covered if (b2,h4) in S_f3
    """

    np.random.seed(seed)
    random.seed(seed)

    vocab = [f"<t_{i}>" for i in range(num_tokens)]

    # -----------------------------
    # Step A: build partial f1, f2, f3
    # -----------------------------

    def sample_pairs(num_pairs, domain_size):
        """
        Return 'num_pairs' distinct (x,y) from [0..domain_size)^2
        or from [0..num_tokens)^2 if domain_size==num_tokens
        """
        s = set()
        out = []
        while len(out) < num_pairs:
            x = np.random.randint(domain_size)
            y = np.random.randint(domain_size)
            if (x,y) not in s:
                s.add((x,y))
                out.append((x,y))
        return out

    # For subcomp1: (h1,h2)-> b1
    #   - We'll just pick them from [0..num_tokens) for demonstration
    S_f1 = sample_pairs(num_pairs_f1_train, domain_size=num_tokens)
    T_f1 = sample_pairs(num_pairs_f1_test,  domain_size=num_tokens)

    # For subcomp2: (b1,h3)-> b2
    #   - The domain "b1" might be up to num_tokens as well. 
    #     For demonstration, we won't constrain b1. We'll treat them as indices in [0..num_tokens).
    S_f2 = sample_pairs(num_pairs_f2_train, domain_size=num_tokens)
    T_f2 = sample_pairs(num_pairs_f2_test,  domain_size=num_tokens)

    # For subcomp3: (b2,h4)-> t
    S_f3 = sample_pairs(num_pairs_f3_train, domain_size=num_tokens)
    T_f3 = sample_pairs(num_pairs_f3_test,  domain_size=num_tokens)

    # Many->one style or direct random assignment:
    f1 = {}
    for (h1,h2) in S_f1:
        b1 = np.random.randint(num_tokens)
        f1[(h1,h2)] = b1
    for (h1,h2) in T_f1:
        b1 = np.random.randint(num_tokens)
        f1[(h1,h2)] = b1

    f2 = {}
    for (b1,h3) in S_f2:
        b2 = np.random.randint(num_tokens)
        f2[(b1,h3)] = b2
    for (b1,h3) in T_f2:
        b2 = np.random.randint(num_tokens)
        f2[(b1,h3)] = b2

    f3 = {}
    for (b2,h4) in S_f3:
        t = np.random.randint(num_tokens)
        f3[(b2,h4)] = t
    for (b2,h4) in T_f3:
        t = np.random.randint(num_tokens)
        f3[(b2,h4)] = t

    # We'll store which pairs are "train domain" for subcomp1,2,3
    S_f1_set = set(S_f1)
    S_f2_set = set(S_f2)
    S_f3_set = set(S_f3)

    # -----------------------------
    # Step B: build possible 4-token combos => final t
    #   We define "train domain" combos if
    #     (h1,h2) in S_f1
    #     => b1 = f1[(h1,h2)]
    #     => subcomp2 => (b1,h3) in S_f2
    #     => b2 = ...
    #     => subcomp3 => (b2,h4) in S_f3
    # Then we define random-split => train vs covered. 
    # Everything else => not_covered
    # (Then we unify covered & not_covered => test)
    # -----------------------------
    train_4tups_dict = defaultdict(list)  # store combos that are "fully in train domain"

    # We'll store f3out for each such combo
    final_out_map = {}

    # 1) gather all "fully train domain" combos
    #   meaning (h1,h2) in S_f1, then b1= f1[(h1,h2)]
    #           (b1,h3) in S_f2, then b2= ...
    #           (b2,h4) in S_f3, then t= ...
    # We'll systematically pick (h1,h2,h3,h4) from S_f1, [0..], S_f2,... etc
    # But that might be huge. Let's do random sampling:

    num_train_quadruples = 1000000
    for _ in range(num_train_quadruples):
        h1 = np.random.randint(num_tokens)
        h2 = np.random.randint(num_tokens)
        h3 = np.random.randint(num_tokens)
        h4 = np.random.randint(num_tokens)
        if (h1,h2) in f1:   # subcomp1 domain?
            b1 = f1[(h1,h2)]
            if (b1,h3) in f2:  # subcomp2 domain?
                b2 = f2[(b1,h3)]
                if (b2,h4) in f3:  # subcomp3 domain?
                    t = f3[(b2,h4)]
                    train_4tups_dict[(b1,b2)].append((h1,h2,h3,h4))
                    final_out_map[(h1,h2,h3,h4)] = t

    # We'll random-split each (b1,b2) subset into train vs covered
    train_inferred = []
    covered_inferred = []
    train_b1b2_set = set()

    for (b1,b2), quadruples in train_4tups_dict.items():
        if len(quadruples)==0:
            continue
        random.shuffle(quadruples)
        cutoff = int(round(train_ratio * len(quadruples)))
        tr = quadruples[:cutoff]
        cv = quadruples[cutoff:]
        # if we have at least 1 item in tr => subcomp2 => subcomp3 => effectively "trained"
        # for final coverage
        if len(tr)>0:
            train_b1b2_set.add((b1,b2))

        # produce actual items
        for (h1,h2,h3,h4) in tr:
            t = final_out_map[(h1,h2,h3,h4)]
            inp_tokens = [vocab[h1], vocab[h2], vocab[h3], vocab[h4]]
            out_token  = vocab[t]
            train_inferred.append(form_item(inp_tokens, out_token))

        for (h1,h2,h3,h4) in cv:
            t = final_out_map[(h1,h2,h3,h4)]
            inp_tokens = [vocab[h1], vocab[h2], vocab[h3], vocab[h4]]
            out_token  = vocab[t]
            item = form_item(inp_tokens, out_token)
            item["type"] = "type_0"  # fully covered
            covered_inferred.append(item)

    # now gather everything else => not_covered
    # "everything else" = all combos from [0..num_tokens)^4 minus used combos
    # can be large. We'll do random sampling again
    used_train_covered = set()
    for item in train_inferred:
        # we won't parse back from text, simpler to store them earlier
        pass
    # we do store them as sets now
    used_train_quads = set()
    for (b1,b2), quadruples in train_4tups_dict.items():
        random.shuffle(quadruples)
        cutoff = int(round(train_ratio * len(quadruples)))
        tr = quadruples[:cutoff]
        cv = quadruples[cutoff:]
        for q in tr:
            used_train_quads.add(q)
        for q in cv:
            used_train_quads.add(q)

    # create not_covered via random
    not_covered_candidates = []
    num_global_samples = 30000
    for _ in range(num_global_samples):
        h1 = np.random.randint(num_tokens)
        h2 = np.random.randint(num_tokens)
        h3 = np.random.randint(num_tokens)
        h4 = np.random.randint(num_tokens)
        if (h1,h2,h3,h4) not in used_train_quads:
            not_covered_candidates.append((h1,h2,h3,h4))

    random.shuffle(not_covered_candidates)
    not_covered_candidates = not_covered_candidates[:max_not_covered]

    test_not_covered = []

    # coverage check
    def covered_subcomp1(h1,h2):
        return (h1,h2) in S_f1_set
    def covered_subcomp2(b1,h3):
        return (b1,h3) in S_f2_set
    def covered_subcomp3(b2,h4):
        return (b2,h4) in S_f3_set

    def coverage_failure_type_3(sc1, sc2, sc3):
        """
        Example:
          if all are True => type_0 (fully covered). 
          else we produce type_1..type_7 or skip if you prefer.
        We'll do a simple code that yields 1..7 for any partial coverage fail.
        We'll encode the pattern as bits sc1,sc2,sc3 => up to 8 patterns.
        sc1 in {0,1}, sc2 in {0,1}, sc3 in {0,1}.
        (1,1,1) => 0 => fully covered => skip from not-covered set
        else => 1..7
        """
        bits = (sc1<<2) + (sc2<<1) + sc3
        # bits in [0..7], 7 means (1,1,1).
        if bits==7:
            return 0
        else:
            return bits+1  # map [0..6] => [1..7]

    for (h1,h2,h3,h4) in not_covered_candidates:
        # define or reuse f1, f2, f3 if possible
        if (h1,h2) in f1:
            b1 = f1[(h1,h2)]
        else:
            b1 = np.random.randint(num_tokens)

        if (b1,h3) in f2:
            b2 = f2[(b1,h3)]
        else:
            b2 = np.random.randint(num_tokens)

        if (b2,h4) in f3:
            t = f3[(b2,h4)]
        else:
            t = np.random.randint(num_tokens)

        sc1 = covered_subcomp1(h1,h2)
        sc2 = covered_subcomp2(b1,h3)
        sc3 = covered_subcomp3(b2,h4)
        ctype = coverage_failure_type_3(sc1, sc2, sc3)
        if ctype==0:
            # means fully covered => skip
            continue
        item = form_item([vocab[h1], vocab[h2], vocab[h3], vocab[h4]], vocab[t])
        item["type"] = f"type_{ctype}"
        test_not_covered.append(item)

    # unify covered + not_covered => final test set
    test_data = covered_inferred + test_not_covered
    random.shuffle(test_data)

    # build atomic facts for analysis
    atomic_facts_1 = []
    for (h1,h2), b1 in f1.items():
        item = form_item([vocab[h1], vocab[h2]], vocab[b1])
        atomic_facts_1.append(item)

    atomic_facts_2 = []
    for (b1,h3), b2 in f2.items():
        item = form_item([vocab[b1], vocab[h3]], vocab[b2])
        atomic_facts_2.append(item)

    atomic_facts_3 = []
    for (b2,h4), t in f3.items():
        item = form_item([vocab[b2], vocab[h4]], vocab[t])
        atomic_facts_3.append(item)

    # shuffle final sets
    random.shuffle(train_inferred)
    random.shuffle(test_data)

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

    # Print stats
    type_counts = {}
    for item in test_data:
        tstr = item["type"]
        type_counts[tstr] = type_counts.get(tstr, 0) + 1
    total_test = len(test_data)
    print(f"\n3-hop dataset in '{outdir}':")
    print(f"  Train: {len(train_inferred)}  (4->1 examples)")
    print(f"  Test:  {len(test_data)}   (4->1 examples, type=0..7)\n")
    print("  Test breakdown by type:")
    for tkey in sorted(type_counts.keys()):
        cnt = type_counts[tkey]
        frac = 100.0*cnt/(1e-9+total_test)
        print(f"    {tkey}: {cnt} ({frac:.1f}%)")
    print()
    print(f"  atomic_facts_1: {len(atomic_facts_1)}")
    print(f"  atomic_facts_2: {len(atomic_facts_2)}")
    print(f"  atomic_facts_3: {len(atomic_facts_3)}\n")


if __name__=="__main__":
    build_3hop_dataset(
        num_tokens=50,  # smaller
        num_pairs_f1_train=600,
        num_pairs_f1_test=200,
        num_pairs_f2_train=600,
        num_pairs_f2_test=200,
        num_pairs_f3_train=600,
        num_pairs_f3_test=200,
        train_ratio=0.7,
        max_not_covered=100000,
        seed=42,
        outdir="test"
    )