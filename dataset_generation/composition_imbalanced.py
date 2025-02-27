import json
import numpy as np
import random
from tqdm.auto import tqdm
import os
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import defaultdict

def build_dicts(entities):
    entity2ind = {}
    ind2entity = []
    for i in range(len(entities)):
        entity = entities[i]
        if entity not in ind2entity:
            ind2entity.append(entity)
            entity2ind[entity] = len(ind2entity) - 1
    return ind2entity, entity2ind

def choose(arr, ratio_or_count):
    """
    Randomly choose a subset of `arr`.
    If `ratio_or_count` is float, interpret as fraction of arr length.
    If int, interpret as exact number to choose.
    """
    if isinstance(ratio_or_count, float):
        num = round(ratio_or_count * len(arr))
    elif isinstance(ratio_or_count, int):
        num = ratio_or_count
    else:
        raise ValueError("ratio_or_count must be float or int")
    if num >= len(arr):
        return arr
    rand_inds = np.random.choice(len(arr), num, replace=False)
    return [arr[i] for i in rand_inds]

def split(arr, ratio_or_count):
    """
    Splits arr into [train, test] with ratio_or_count controlling the train size.
    """
    if isinstance(ratio_or_count, float):
        num = round(ratio_or_count * len(arr))
    elif isinstance(ratio_or_count, int):
        num = ratio_or_count
    else:
        raise ValueError("ratio_or_count must be float or int")
    train, test = [], []
    rand_inds = set(np.random.choice(len(arr), num, replace=False))
    for i in tqdm(range(len(arr))):
        if i in rand_inds:
            train.append(arr[i])
        else:
            test.append(arr[i])
    return [train, test]

def form_items(c, t):
    """
    Builds dict {input_text: "...", target_text: "..."}
    """
    input_text = "".join(c)
    target_text = input_text + "".join([t, "</a>"])
    return {
        "input_text": input_text,
        "target_text": target_text
    }

def build_dataset(num_entities, num_relations, out_degree=20, split_train_inferred=False):
    """
    Creates a dataset with 1-hop facts (atomic_facts) and 2-hop facts.
    - If split_train_inferred=False: returns all atomic_facts + all 2-hop (inferred_facts).
    - If split_train_inferred=True: 
        * Splits atomic facts into ID vs OOD.
        * 2-hop test data has only two types: "test_inferred_ID" (if both 1-hops are ID) 
          or "test_inferred_OOD" (if both 1-hops are OOD).
        * The rest (partial OOD or partial ID) is used for training only.
        * We also introduce an imbalance for training facts based on the bridge entity’s group 
          (majority vs. minority).
        * Each fact’s "type" is assigned to one of:
            - "train_maj" / "train_min"
            - "ID_maj" / "ID_min"
            - "OOD_maj" / "OOD_min"
    """

    # 1. Build entity strings
    entities = [f"<e_{i}>" for i in range(num_entities)]
    ind2entity, entity2ind = build_dicts(entities)

    # 2. Define majority/minority sets by splitting the list of entities in half
    half_size = num_entities // 2
    majority_entities = set(entities[:half_size])
    minority_entities = set(entities[half_size:])

    # 3. Build relation strings
    relations = [f"<r_{i}>" for i in range(num_relations)]
    ind2relation, relation2ind = build_dicts(relations)

    # 4. Create the 1-hop (atomic) dictionary
    atomic_dict = dict()  # maps h -> list[(r, t)]
    atomic_facts = []
    atomics = []

    for i in tqdm(range(num_entities), desc="Generating 1-hop facts"):
        h = ind2entity[i]
        # pick out_degree random relations
        rel_indices = np.random.choice(num_relations, size=out_degree, replace=False)
        for r_idx in rel_indices:
            r = ind2relation[r_idx]
            # choose a random tail
            col_idx = np.random.randint(num_entities)
            t = ind2entity[col_idx]
            atomic_facts.append(form_items([h, r], t))
            atomics.append((h, r, t))
            if h not in atomic_dict:
                atomic_dict[h] = []
            atomic_dict[h].append((r, t))

    if not split_train_inferred:
        # Simple case: just produce 2-hop from all
        inferred_facts = []
        for ent in tqdm(entities, desc="Generating 2-hop facts (no split)"):
            if ent not in atomic_dict:
                continue
            for (r1, b) in atomic_dict[ent]:
                if b not in atomic_dict:
                    continue
                for (r2, t) in atomic_dict[b]:
                    inferred_facts.append(form_items([ent, r1, r2], t))
        return entities, relations, atomic_facts, inferred_facts

    # Otherwise, do the ID/OOD splitting (5% OOD) for 1-hop facts
    OOD_ratio = 0.05
    train_list, test_list = split(atomics, round(len(atomics) * OOD_ratio))
    OOD_facts = set(train_list)
    ID_facts = set(test_list)
    # Actually, the naming above is a bit reversed because of the usage of `split`,
    # but let's rename for clarity:
    # We want OOD_facts to be the smaller set => 5% => train_list
    # We want ID_facts to be the bigger set => 95% => test_list
    # So it's correct:  OOD_facts = 5%,  ID_facts = 95%.

    # We'll keep the rest of the ID vs OOD subsets you had, 
    # but we do not actually need o1_id_atomic_facts, etc. 
    # (We’ll keep them if you still want them for other usage.)
    only_used_ID_facts = random.sample(list(ID_facts), 2000)
    o2_ID_facts = only_used_ID_facts[:1000]
    o1_ID_facts = only_used_ID_facts[1000:]

    id_atomic_facts = [form_items([h, r], t) for (h, r, t) in ID_facts]
    ood_atomic_facts = [form_items([h, r], t) for (h, r, t) in OOD_facts]
    o1_id_atomic_facts = [form_items([h, r], t) for (h, r, t) in o1_ID_facts]
    o2_id_atomic_facts = [form_items([h, r], t) for (h, r, t) in o2_ID_facts]

    # 5. Build final sets of 2-hop: 
    #    - train_inferred (with majority/minority imbalance)
    #    - test_inferred_ID (fully ID)
    #    - test_inferred_OOD (fully OOD)
    train_inferred = []
    test_inferred_id = []
    test_inferred_ood = []

    # For logging
    stats = defaultdict(int)

    for ent in tqdm(entities, desc="Generating 2-hop facts (with split)"):
        if ent not in atomic_dict:
            continue
        for (r1, b) in atomic_dict[ent]:
            if b not in atomic_dict:
                continue
            for (r2, t) in atomic_dict[b]:
                # Are both 1-hop facts OOD => test_inferred_ood
                if (ent, r1, b) in OOD_facts and (b, r2, t) in OOD_facts:
                    item = form_items([ent, r1, r2], t)
                    if b in majority_entities:
                        item["type"] = "OOD_maj"
                    else:
                        item["type"] = "OOD_min"
                    test_inferred_ood.append(item)
                    stats[item["type"]] += 1

                # Are both 1-hop facts ID => possibly train or test_inferred_id
                elif (ent, r1, b) in ID_facts and (b, r2, t) in ID_facts:
                    # Check majority/minority => 4:1 imbalance
                    if b in majority_entities:
                        # ~99.5% to train
                        if np.random.uniform() > 0.005:
                            item = form_items([ent, r1, r2], t)
                            item["type"] = "train_maj"
                            train_inferred.append(item)
                            stats["train_maj"] += 1
                        else:
                            item = form_items([ent, r1, r2], t)
                            item["type"] = "ID_maj"
                            test_inferred_id.append(item)
                            stats["ID_maj"] += 1
                    else:
                        # ~24.875% to train
                        if np.random.uniform() < 0.24875:
                            item = form_items([ent, r1, r2], t)
                            item["type"] = "train_min"
                            train_inferred.append(item)
                            stats["train_min"] += 1
                        else:
                            item = form_items([ent, r1, r2], t)
                            item["type"] = "ID_min"
                            test_inferred_id.append(item)
                            stats["ID_min"] += 1

                # # Otherwise => partial ID/OOD => 
                # # => we do NOT want them in the final test, so just put them in train with imbalance
                # else:
                #     # Bridge entity => majority => 99.5% train
                #     if b in majority_entities:
                #         if np.random.uniform() > 0.005:
                #             item = form_items([ent, r1, r2], t)
                #             item["type"] = "train_maj"
                #             train_inferred.append(item)
                #             stats["train_maj"] += 1
                #         else:
                #             # In principle, you could put partial in test or skip entirely. 
                #             # We'll just put them in train to keep it simple.
                #             item = form_items([ent, r1, r2], t)
                #             item["type"] = "train_maj"
                #             train_inferred.append(item)
                #             stats["train_maj"] += 1
                #     else:
                #         # minority => ~24.875% train
                #         if np.random.uniform() < 0.24875:
                #             item = form_items([ent, r1, r2], t)
                #             item["type"] = "train_min"
                #             train_inferred.append(item)
                #             stats["train_min"] += 1
                #         else:
                #             item = form_items([ent, r1, r2], t)
                #             item["type"] = "train_min"
                #             train_inferred.append(item)
                #             stats["train_min"] += 1

    # 6. Create nonsense facts
    nonsenses = []
    for i in tqdm(range(num_entities), desc="Generating nonsense"):
        num_rows = out_degree
        selected_rows = np.random.choice(num_entities, size=num_rows, replace=False).tolist()
        for row_idx in selected_rows:
            col_idx_1 = np.random.randint(num_entities)
            col_idx_2 = np.random.randint(num_entities)
            e1 = ind2entity[i]
            e2 = ind2entity[row_idx]
            e3 = ind2entity[col_idx_1]
            e4 = ind2entity[col_idx_2]
            nonsenses.append((e1, e2, e3, e4))

    nonsenses = set(nonsenses)
    nonsenses_facts = [form_items([t1, t2, t3], t4) for (t1, t2, t3, t4) in nonsenses]

    # Print the stats as a quick sanity check
    print("==== Data Statistics ====")
    print(f"train_maj: {stats['train_maj']}")
    print(f"train_min: {stats['train_min']}")
    print(f"ID_maj: {stats['ID_maj']}")
    print(f"ID_min: {stats['ID_min']}")
    print(f"OOD_maj: {stats['OOD_maj']}")
    print(f"OOD_min: {stats['OOD_min']}")
    print("=========================")

    # Return the relevant pieces
    return (entities, relations,
            id_atomic_facts, ood_atomic_facts,
            o1_id_atomic_facts, o2_id_atomic_facts,
            train_inferred, test_inferred_id,
            test_inferred_ood, nonsenses_facts
    )

# =====================================================================
# Example usage
# =====================================================================
NUM_ENTITY_IN = 2000
NUM_RELATION = 200

( train_entities, train_relations,
  id_atomic_facts, ood_atomic_facts,
  o1_id_atomic_facts, o2_id_atomic_facts,
  train_inferred, test_inferred_id,
  test_inferred_ood, nonsenses ) = build_dataset(NUM_ENTITY_IN, NUM_RELATION, 
                                                 split_train_inferred=True)

# Build a vocabulary
vocab = []
vocab += train_entities
vocab += train_relations
vocab += ["<mask>", "<sep>", "<a>", "</a>", "<q>", "</q>"]
assert len(vocab) == len(set(vocab))

print("Vocabulary size:", len(vocab))

# Pick some subset sizes for test, etc.
test_size = 900

# Partition the big lists by type
test_inferred_id_maj = [x for x in test_inferred_id if x["type"] == "ID_maj"]
test_inferred_id_min = [x for x in test_inferred_id if x["type"] == "ID_min"]
test_inferred_ood_maj = [x for x in test_inferred_ood if x["type"] == "OOD_maj"]
test_inferred_ood_min = [x for x in test_inferred_ood if x["type"] == "OOD_min"]

# Now choose separately:
test_inferred_id_maj_ds = choose(test_inferred_id_maj, test_size)
test_inferred_id_min_ds = choose(test_inferred_id_min, test_size)
test_inferred_ood_maj_ds = choose(test_inferred_ood_maj, test_size)
test_inferred_ood_min_ds = choose(test_inferred_ood_min, test_size)

# Then combine them back for final ID test set and OOD test set
final_test_inferred_id = test_inferred_id_maj_ds + test_inferred_id_min_ds
final_test_inferred_ood = test_inferred_ood_maj_ds + test_inferred_ood_min_ds

nonsense_ds = choose(nonsenses, test_size)

# Keep all atomic facts for training if desired,
# or you could combine them however you wish.
all_atomics = id_atomic_facts + ood_atomic_facts

# Possibly downsample train_inferred if you like.
phi = 9.0
train_inferred_ds = choose(train_inferred, round(phi * len(id_atomic_facts)))

# Build final "test" set of probes
probes = []

# If you want ID_maj + ID_min + OOD_maj + OOD_min all in one test file:
probes.extend(final_test_inferred_id)
probes.extend(final_test_inferred_ood)
probes.extend(nonsense_ds)

dataset_name = f"composition.{NUM_ENTITY_IN}.{NUM_RELATION}.inf-imbalanced"
os.makedirs(f"data/{dataset_name}", exist_ok=True)

with open(f"data/{dataset_name}/train.json", "w", encoding='utf-8') as f:
    # Combine atomic + 2-hop train
    json.dump(train_inferred_ds, f, ensure_ascii=False)

with open(f"data/{dataset_name}/valid.json", "w", encoding='utf-8') as f:
    # For example, we can use OOD test set as "validation"
    json.dump(final_test_inferred_ood, f, ensure_ascii=False)

with open(f"data/{dataset_name}/test.json", "w", encoding='utf-8') as f:
    json.dump(probes, f, ensure_ascii=False)

# Save vocab
with open(f"data/{dataset_name}/vocab.json", "w", encoding='utf-8') as f:
    json.dump(vocab, f, ensure_ascii=False)