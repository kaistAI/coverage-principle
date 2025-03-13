import argparse
import numpy as np
import random
from collections import defaultdict
import logging
from tqdm import tqdm
import os
import json


# --------------------- Helper Functions ---------------------
def setup_logging(debug_mode: bool):
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s - %(message)s")


def form_items(tokens: list, target: str) -> dict:
    """
    Combine list of tokens into input text and append the target with a closing tag.
    """
    input_text = "".join(tokens)
    target_text = input_text + target + "</a>"
    return {"input_text": input_text, "target_text": target_text}


def choose(arr, ratio_or_count):
    """
    Randomly select a subset of items from a list.
    If ratio_or_count is a float, select that fraction of items.
    If it is an int, select that many items.
    """
    if isinstance(ratio_or_count, float):
        num = round(ratio_or_count * len(arr))
    elif isinstance(ratio_or_count, int):
        num = ratio_or_count
    else:
        assert False, "Invalid ratio_or_count type"
    if num >= len(arr):
        return arr
    rand_inds = np.random.choice(len(arr), num, replace=False).tolist()
    return [arr[i] for i in rand_inds]


def coverage_type(sc1: bool, sc2: bool, sc3: bool) -> int:
    """
    Determine coverage type based on three boolean flags.
    Returns 0 if all three are seen, otherwise returns a value between 1 and 7.
    """
    bits = (sc1 << 2) + (sc2 << 1) + sc3
    return 0 if bits == 7 else bits + 1


def reservoir_update(reservoir, tup, total_count, capacity):
    """
    Perform a single reservoir sampling update for a given coverage bucket.
    """
    if len(reservoir) < capacity:
        reservoir.append(form_items(tup[0], tup[1]))
    else:
        r = random.randint(0, total_count - 1)
        if r < capacity:
            reservoir[r] = form_items(tup[0], tup[1])



def main():
    # ----- Argument Parsing and Setup -----
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_entities", type=int, required=True,
                        help="Number of entity tokens")
    parser.add_argument("--num_relations", type=int, required=True,
                        help="Number of relation tokens")
    parser.add_argument("--num_out_degree", type=int, default=20,
                        help="Number of relations per entity token")
    parser.add_argument("--default_seen_ratio", type=float, default=0.7,
                        help="Ratio of the domain considered 'seen'")
    parser.add_argument("--max_train_data_num", type=int, default=382000,
                        help="Max number of 3-hop samples for training")
    parser.add_argument("--test_size_for_type", type=int, default=3000,
                        help="Number of final test samples per coverage type")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug-level logging")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    setup_logging(args.debug)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ----- Vocabulary Generation -----
    # Create entities and relations tokens
    entities = [f"<e_{i}>" for i in range(args.num_entities)]
    relations = [f"<r_{i}>" for i in range(args.num_relations)]
    # Combine tokens to form vocabulary and append closing tag
    vocab = entities + relations + ["</a>"]
    assert len(vocab) == len(set(vocab)), "Vocabulary contains duplicate tokens"
    print("vocab size:", len(vocab))
    
    # Initialize structures for atomic facts
    atomic_dict = {}              # subject -> list of (relation, object)
    atomic_facts = []             # atomic facts in form_items format
    atomic_facts_hr_to_t_dict = {}  # mapping (subject, relation) -> object
    atomics = []                  # list of tuples (subject, relation, object)
    
    # Generate atomic facts for each entity
    for i in tqdm(range(args.num_entities), desc="Generating atomic facts"):
        h = entities[i]
        selected_relations = np.random.choice(args.num_relations, size=args.num_out_degree, replace=False).tolist()
        for r_idx in selected_relations:
            t_entity = entities[np.random.randint(args.num_entities)]
            r = relations[r_idx]
            t = t_entity
            atomic_facts.append(form_items([h, r], t))
            atomic_facts_hr_to_t_dict[(h, r)] = t
            atomics.append((h, r, t))
            atomic_dict.setdefault(h, []).append((r, t))
    print(f"Total atomic facts: {len(atomics)}")
    
    # ----- Create Seen Expected Facts for f1, f2, f3 -----
    seen_expected_f1 = random.sample(atomics, round(args.default_seen_ratio * len(atomics)))
    seen_expected_f1_dict = {}
    for h, r, t in seen_expected_f1:
        seen_expected_f1_dict.setdefault(h, set()).add((r, t))
    
    seen_expected_f2 = random.sample(atomics, round(args.default_seen_ratio * len(atomics)))
    seen_expected_f2_dict = {}
    for h, r, t in seen_expected_f2:
        seen_expected_f2_dict.setdefault(h, set()).add((r, t))
    
    seen_expected_f3 = random.sample(atomics, round(args.default_seen_ratio * len(atomics)))
    seen_expected_f3_dict = {}
    for h, r, t in seen_expected_f3:
        seen_expected_f3_dict.setdefault(h, set()).add((r, t))
    
    print(f"Overlap between f1 and f2: {len(set(seen_expected_f1) & set(seen_expected_f2))}")
    print(f"Overlap between f1 and f3: {len(set(seen_expected_f1) & set(seen_expected_f3))}")
    print(f"Overlap between f2 and f3: {len(set(seen_expected_f2) & set(seen_expected_f3))}")
    
    # ----- Generate Possible and Train Inferred Facts -----
    possible_inferred_facts = set()
    for h in tqdm(entities, desc="Generating possible inferred facts"):
        for r1, b1 in atomic_dict[h]:
            for r2, b2 in atomic_dict[b1]:
                for r3, t in atomic_dict[b2]:
                    possible_inferred_facts.add((h, r1, r2, r3, t))
    print(f"Total possible inferred facts: {len(possible_inferred_facts)}")
    
    seen_expected_train_inferred_facts = set()
    for h in tqdm(entities, desc="Generating train inferred facts"):
        for r1, b1 in seen_expected_f1_dict.get(h, []):
            for r2, b2 in seen_expected_f2_dict.get(b1, []):
                for r3, t in seen_expected_f3_dict.get(b2, []):
                    seen_expected_train_inferred_facts.add((h, r1, r2, r3, t))
    
    train_inferred = choose(list(seen_expected_train_inferred_facts), args.max_train_data_num)
    print(f"Example train inferred fact: {train_inferred[0] if train_inferred else 'None'}")
    
    # ----- Create Seen Atomic Facts for f1, f2, f3 -----
    seen_f1_atomic_facts, seen_f2_atomic_facts, seen_f3_atomic_facts = {}, {}, {}
    seen_f1_atomics, seen_f2_atomics, seen_f3_atomics = set(), set(), set()
    
    for fact in train_inferred:
        # fact is a tuple: (h, r1, r2, r3, t)
        h, r1, r2, r3, t = fact
        b1 = atomic_facts_hr_to_t_dict[(h, r1)]
        seen_f1_atomic_facts.setdefault(h, set()).add((r1, b1))
        seen_f1_atomics.add((h, r1, b1))
        
        b2 = atomic_facts_hr_to_t_dict[(b1, r2)]
        seen_f2_atomic_facts.setdefault(b1, set()).add((r2, b2))
        seen_f2_atomics.add((b1, r2, b2))
        
        assert t == atomic_facts_hr_to_t_dict[(b2, r3)]
        seen_f3_atomic_facts.setdefault(b2, set()).add((r3, t))
        seen_f3_atomics.add((b2, r3, t))
    
    # ----- Generate Covered and OOD Data -----
    train_inferred = set(train_inferred)
    possible_covered_data = set()
    for h, f1_set in tqdm(seen_f1_atomic_facts.items(), desc="Generating covered data"):
        for r1, b1 in f1_set:
            for r2, b2 in seen_f2_atomic_facts.get(b1, []):
                for r3, t in seen_f3_atomic_facts.get(b2, []):
                    if (h, r1, r2, r3, t) in train_inferred:
                        continue
                    possible_covered_data.add((h, r1, r2, r3, t))
    print(f"Total possible covered data: {len(possible_covered_data)}")
    
    covered_data = choose(list(possible_covered_data), args.test_size_for_type)
    possible_ood_data = possible_inferred_facts - (possible_covered_data | train_inferred)
    print(f"Total possible OOD data: {len(possible_ood_data)}")
    
    train_inferred = list(train_inferred)
    
    # ----- Reservoir Sampling for OOD Data -----
    coverage_reservoirs = defaultdict(list)  # coverage type -> list of items
    coverage_seen_count = defaultdict(int)   # coverage type -> count of seen items
    skip_p = 0.4  # skip probability for OOD data
    
    for ood_data in possible_ood_data:
        if random.random() > (1 - skip_p):
            continue
        h, r1, r2, r3, t = ood_data
        b1 = atomic_facts_hr_to_t_dict[(h, r1)]
        b2 = atomic_facts_hr_to_t_dict[(b1, r2)]
        assert t == atomic_facts_hr_to_t_dict[(b2, r3)]
        
        sc1 = ((h, r1, b1) in seen_f1_atomics)
        sc2 = ((b1, r2, b2) in seen_f2_atomics)
        sc3 = ((b2, r3, t) in seen_f3_atomics)
        c_type = coverage_type(sc1, sc2, sc3)
        
        key = f"type_{c_type}"
        coverage_seen_count[key] += 1
        reservoir_update(
            reservoir=coverage_reservoirs[key],
            tup=([h, r1, r2, r3], t),
            total_count=coverage_seen_count[key],
            capacity=args.test_size_for_type
        )
    
    for ctype, items in coverage_reservoirs.items():
        logging.info(f"{ctype}: {len(items)} items, example: {items[:5]}")
    
    # ----- Prepare Final Training and Test Data -----
    train_inferred_final = [form_items(list(item[:4]), item[-1]) for item in train_inferred]
    
    test_data = []
    # Add a subset of training inferred data to the test set
    train_sample_for_test = choose(train_inferred_final, args.test_size_for_type)
    for item in train_sample_for_test:
        item["type"] = "train_inferred"
        test_data.append(item)
    
    # Add covered data with type "type_0"
    covered_data_final = [form_items(list(item[:4]), item[-1]) for item in covered_data]
    for item in covered_data_final:
        item["type"] = "type_0"
        test_data.append(item)
    
    # Add OOD data from reservoir sampling to the test set
    for ctype, items in coverage_reservoirs.items():
        for item in items:
            item["type"] = ctype
            test_data.append(item)
    
    random.shuffle(train_inferred_final)
    random.shuffle(test_data)
    
    # ----- Save Data to Files -----
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, "data",
                            f"threehop-ent&rel.{args.num_entities}.{args.num_relations}.{args.max_train_data_num}.same-f123.inf")
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    with open(os.path.join(save_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_inferred_final, f, indent=2)
    with open(os.path.join(save_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)
    with open(os.path.join(save_dir, "atomic_facts.json"), "w", encoding="utf-8") as f:
        json.dump(atomic_facts, f, indent=2)
    
    print("[INFO] Done!")

if __name__ == "__main__":
    main()
