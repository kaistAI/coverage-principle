from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from itertools import combinations, chain, product
from typing import Dict, List, Tuple, Set, FrozenSet
from tqdm import tqdm

import multiprocessing as mp
from functools import partial

import networkx as nx

###############################################################################
# helpers
###############################################################################

def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


def parse_input_tokens(s: str) -> Tuple[int, int, int]:
    return tuple(int(tok.split("_")[-1]) for tok in s.strip("<>").split("><"))


def powerset(iterable):
    """Return all possible subsets of the iterable except empty set and full set."""
    s = list(iterable)
    return [frozenset(subset) for r in range(1, len(s)) for subset in combinations(s, r)]


###############################################################################
# Union–Find for equivalence classes
###############################################################################

class UnionFind:
    def __init__(self):
        self.parent = {}
        self.size = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


###############################################################################
# Build equivalence classes for all possible subsequences
###############################################################################

def extract_subsequence(full_sequence, indices):
    """Extract a subsequence from the full sequence using the provided indices."""
    return tuple(full_sequence[i] for i in indices)


def build_equiv_classes_for_subset(train_map: Dict[Tuple[int, int, int], int], 
                                 subset_indices: FrozenSet[int], 
                                 min_evidence: int = 1) -> UnionFind:
    """
    Build equivalence classes for a specific subset of indices.
    Two subsequences are equivalent if:
      ⋆ there exist at least 'min_evidence' distinct complements such that both 
        combinations are in the training data with matching targets, and
      ⋆ on every complement that appears for both subsequences, the targets match.
    """
    # Convert indices from 0-based to 1-based for logging
    subset_indices_1based = {i+1 for i in subset_indices}
    logging.debug(f"Building equivalence classes for subset {subset_indices_1based}")
    
    # 1) gather behavior for each subsequence
    # Subsequence -> {Complement -> Target}
    behavior = defaultdict(dict)
    complement_indices = frozenset(range(3)) - subset_indices
    
    for full_seq, target in train_map.items():
        subseq = extract_subsequence(full_seq, subset_indices)
        complement = extract_subsequence(full_seq, complement_indices)
        behavior[subseq][complement] = target
    
    # 2) Count shared evidence and check for contradictions
    pair_evidence = defaultdict(int)
    contradictions = set()
    
    for (subseq1, compl_map1), (subseq2, compl_map2) in combinations(behavior.items(), 2):
        # if subseq1 >= subseq2:
        #     continue
            
        # Find shared complements
        shared_complements = set(compl_map1) & set(compl_map2)
        
        # Check for contradictions
        for comp in shared_complements:
            if compl_map1[comp] != compl_map2[comp]:
                contradictions.add(tuple(sorted([subseq1, subseq2])))
                break
        
        # If no contradictions, count matching evidence
        if tuple(sorted([subseq1, subseq2])) not in contradictions and shared_complements:
            matching_evidence = sum(1 for comp in shared_complements 
                                  if compl_map1[comp] == compl_map2[comp])
            pair_evidence[(subseq1, subseq2)] = matching_evidence
    
    # 3) Union subsequences with sufficient evidence
    uf = UnionFind()
    
    for (subseq1, subseq2), evidence_count in pair_evidence.items():
        if evidence_count >= min_evidence and tuple(sorted([subseq1, subseq2])) not in contradictions:
            uf.union(subseq1, subseq2)
    
    # Initialize all subsequences in the Union-Find structure
    for subseq in behavior:
        _ = uf.find(subseq)
    
    num_classes = len({uf.find(subseq) for subseq in behavior})
    logging.info(f"Subset {subset_indices_1based}: {num_classes} equivalence classes from {len(behavior)} subsequences")
    return uf


def build_all_equiv_classes(train_map: Dict[Tuple[int, int, int], int], 
                           min_evidence: int = 1,
                           ground_truth: bool = False) -> Dict[FrozenSet[int], UnionFind]:
    """Build equivalence classes for all possible subsequences."""
    all_subsets = powerset(range(3)) if not ground_truth else [frozenset((0,1))]  # All non-trivial subsets of {0,1,2}
    return {subset: build_equiv_classes_for_subset(train_map, subset, min_evidence) 
            for subset in all_subsets}


###############################################################################
# Substitution graph & coverage
###############################################################################

def build_full_subst_graph(all_triples, triple2t, equiv_classes):
    """Build substitution graph using the bucketing approach for better alignment with the coverage principle."""
    G = nx.Graph()
    for tr in all_triples:
        G.add_node(tr)
    
    # Create buckets based on equivalence classes
    buckets = defaultdict(list)
    
    for tr in all_triples:
        h1, h2, h3 = tr
        t = triple2t[tr]
        
        # For each subset of indices we have equivalence classes for
        for subset_indices in equiv_classes:
            # Extract the subsequence for this subset
            subseq = extract_subsequence(tr, subset_indices)
            
            # Extract the complement indices and values
            complement_indices = frozenset(range(3)) - subset_indices
            complement = extract_subsequence(tr, complement_indices)
            
            # Get the equivalence class for this subsequence
            equiv_class = equiv_classes[subset_indices].find(subseq)
            
            # Add to bucket: (equiv_class, complement, target)
            buckets[(equiv_class, complement, t)].append(tr)
    
    # Connect all pairs in each bucket
    edge_count = 0
    for triples in buckets.values():
        for a, b in combinations(triples, 2):
            G.add_edge(a, b)
            edge_count += 1
    
    logging.info(f"Substitution graph: |V|={G.number_of_nodes()}, |E|={edge_count}")
    return G

def find_edges_in_batch(batch, triple2t, equiv_classes):
    """Find all valid edges in a batch of triple pairs."""
    edges = []
    
    for a, b in batch:
        # Only connect if they have the same target
        if triple2t[a] != triple2t[b]:
            continue
        
        # For each subset that we have equivalence classes for
        for subset_indices in equiv_classes:
            # Check if they differ ONLY on the current subset
            if all(a[i] == b[i] for i in range(3) if i not in subset_indices) and \
               any(a[i] != b[i] for i in subset_indices):
                
                # Extract the subsequences for this subset
                subseq_a = extract_subsequence(a, subset_indices)
                subseq_b = extract_subsequence(b, subset_indices)
                
                # Check if they're in the same equivalence class
                uf = equiv_classes[subset_indices]
                if uf.find(subseq_a) == uf.find(subseq_b):
                    edges.append((a, b))
                    break  # Found a connecting subset, no need to check others
    
    return edges


def compute_coverage(G: nx.Graph, train_triples: List[Tuple[int, int, int]]) -> Set[Tuple[int, int, int]]:
    covered = set()
    for tr in train_triples:
        if tr not in covered:
            covered.update(nx.node_connected_component(G, tr))
    logging.info(f"Covered nodes: {len(covered)}")
    return covered


###############################################################################
# main
###############################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--visualise", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--min_evidence", type=int, default=1,
                   help="minimum evidence for equivalence classes")
    ap.add_argument("--k_sweep", action="store_true",
                   help="Run multiple min_evidence values (1-8) and report coverage")
    ap.add_argument("--ground_truth", action="store_true",
                   help="Run multiple min_evidence values (1-8) and report coverage")
    args = ap.parse_args()

    setup_logging(args.debug)

    def jload(name):
        with open(os.path.join(args.data_dir, name), "r", encoding="utf-8") as f:
            return json.load(f)

    train = jload("train.json")
    test = jload("test.json")

    parse = parse_input_tokens

    train_triples = [parse(it["input_text"]) for it in train]
    test_triples = [parse(it["input_text"]) for it in test if it["type"] == "type_0"]
    # test_triples = [d for d in test_triples if d not in train_triples]  # remove duplicates
    logging.info(f"train triples: {len(train_triples)}  test triples: {len(test_triples)}")
    assert len(train_triples) == len(set(train_triples)), "duplicate triples in train.json"
    assert len(test_triples) == len(set(test_triples)), "duplicate triples in test.json"

    def label(it):
        t_tok = it["target_text"].replace("</a>", "").strip("<>").split("><")[-1]
        return int(t_tok.split("_")[-1])

    triple2t = {parse(it["input_text"]): label(it) for it in train + test}
    train_map = {tr: triple2t[tr] for tr in train_triples}


    if args.k_sweep:
        k_sweep_results = {}
        
        # For tracking coverage of each example at each k
        # Map from test example index to highest k where it's covered
        type0_indices = [i for i, item in enumerate(test) if item.get("type") == "type_0"]
        coverage_by_example = {i: 0 for i in type0_indices}  # Default to uncovered
        
        # Cache the behavior maps for efficiency
        logging.info("Caching behavior maps for all subsets (will speed up k-sweep)...")
        behavior_maps = {}
        
        # This function extracts and caches the behavior map for a subset
        def get_behavior_map(subset_indices):
            if subset_indices not in behavior_maps:
                # Initialize the behavior map
                behavior = defaultdict(dict)
                complement_indices = frozenset(range(3)) - subset_indices
                
                for full_seq, target in train_map.items():
                    subseq = extract_subsequence(full_seq, subset_indices)
                    complement = extract_subsequence(full_seq, complement_indices)
                    behavior[subseq][complement] = target
                
                behavior_maps[subset_indices] = behavior
            
            return behavior_maps[subset_indices]
        
        # Pre-compute all behavior maps
        all_subsets = powerset(range(3)) if not args.ground_truth else [frozenset((0,1))]
        for subset in all_subsets:
            _ = get_behavior_map(subset)
        
        # Run the k-sweep
        from tqdm import tqdm
        k_values = range(1, 21)  # k from 1 to 8
        
        for k in tqdm(k_values, desc="K-sweep progress"):
            logging.info(f"Running coverage analysis with min_evidence = {k}")
            
            # Build equivalence classes with behavior maps and current k
            equiv_classes = {}
            for subset in all_subsets:
                behavior = behavior_maps[subset]
                
                # Initialize UnionFind
                uf = UnionFind()
                
                # Count shared evidence and check for contradictions
                pair_evidence = defaultdict(int)
                contradictions = set()
                
                for (subseq1, compl_map1), (subseq2, compl_map2) in combinations(behavior.items(), 2):
                    # Find shared complements
                    shared_complements = set(compl_map1) & set(compl_map2)
                    
                    # Check for contradictions
                    for comp in shared_complements:
                        if compl_map1[comp] != compl_map2[comp]:
                            contradictions.add(tuple(sorted([subseq1, subseq2])))
                            break
                    
                    # If no contradictions, count matching evidence
                    if tuple(sorted([subseq1, subseq2])) not in contradictions and shared_complements:
                        matching_evidence = sum(1 for comp in shared_complements 
                                            if compl_map1[comp] == compl_map2[comp])
                        pair_evidence[(subseq1, subseq2)] = matching_evidence
                
                # Union subsequences with sufficient evidence
                for (subseq1, subseq2), evidence_count in pair_evidence.items():
                    if evidence_count >= k and tuple(sorted([subseq1, subseq2])) not in contradictions:
                        uf.union(subseq1, subseq2)
                
                # Initialize all subsequences in the UnionFind structure
                for subseq in behavior:
                    _ = uf.find(subseq)
                
                equiv_classes[subset] = uf
            
            # Build graph and compute coverage
            G = build_full_subst_graph(train_triples + test_triples, triple2t, equiv_classes)
            covered = compute_coverage(G, train_triples)
            
            # Count coverage for type_0 and track individual test examples
            type0_total = 0
            type0_covered = 0
            
            for i in type0_indices:
                type0_total += 1
                if parse(test[i]["input_text"]) in covered:
                    type0_covered += 1
                    # Update this example's coverage threshold to current k
                    coverage_by_example[i] = k
            
            # Calculate percentage
            coverage_pct = (type0_covered / type0_total * 100) if type0_total > 0 else 0
            k_sweep_results[k] = (type0_covered, coverage_pct)
            
            logging.info(f"k={k}: {type0_covered}/{type0_total} type_0 covered ({coverage_pct:.2f}%)")
        
        # Write results to JSON file
        os.makedirs("k_sweep_results", exist_ok=True)
        k_sweep_file = os.path.join("k_sweep_results", f"{args.data_dir.split('/')[-2]}{'_ground-truth' if args.ground_truth else ''}.json")
        with open(k_sweep_file, "w", encoding="utf-8") as f:
            json.dump(k_sweep_results, f, indent=2)
        
        logging.info(f"K-sweep results written to {k_sweep_file}")
        
        # Create modified test data with coverage thresholds
        new_test_data = test.copy()
        coverage_threshold_counts = {i: 0 for i in range(9)}  # Count examples for each threshold
        
        # Create entries with coverage threshold types
        for i in type0_indices:
            threshold = coverage_by_example[i]
            coverage_threshold_counts[threshold] += 1
            
            # Create a duplicate entry with the coverage threshold type
            duplicate_entry = test[i].copy()
            duplicate_entry["type"] = f"covered_{threshold}"
            new_test_data.append(duplicate_entry)
        
        # Write the modified test file
        threshold_test_file = os.path.join(args.data_dir, f"test_annotated{'_ground-truth' if args.ground_truth else ''}.json")
        with open(threshold_test_file, "w", encoding="utf-8") as f:
            json.dump(new_test_data, f, indent=2)
        
        # Log threshold distribution
        logging.info("Coverage threshold distribution:")
        for threshold, count in coverage_threshold_counts.items():
            logging.info(f"  covered_{threshold}: {count} examples")
        
        logging.info(f"Enhanced test data with coverage thresholds written to {threshold_test_file}")
    





    else:
        # Build equivalence classes for all possible subsequences
        equiv_classes = build_all_equiv_classes({tr: triple2t[tr] for tr in train_triples}, 
                                            min_evidence=args.min_evidence,
                                            ground_truth=args.ground_truth)
        
        # Build full substitution graph
        G = build_full_subst_graph(train_triples + test_triples, triple2t, equiv_classes)
        
        # Compute coverage
        covered = compute_coverage(G, train_triples)

        # Annotate test.json
        for it in test:
            it["coverage"] = bool(parse(it["input_text"]) in covered)
        
        # Coverage report by type
        totals = defaultdict(int)
        hits = defaultdict(int)

        for it in test:
            typ = it.get("type", "UNK")
            totals[typ] += 1
            if it["coverage"]:
                hits[typ] += 1

        for typ in sorted(totals):
            pct = 100.0 * hits[typ] / totals[typ]
            logging.info(f"Coverage [{typ}] : {hits[typ]} / {totals[typ]}  ({pct:.2f}%)")

        # Write annotated test file with appropriate suffix
        annotated_file = os.path.join(args.data_dir, f"test_annotated{'_ground-truth' if args.ground_truth else '_full'}.json")
        with open(annotated_file, "w", encoding="utf-8") as f:
            json.dump(test, f, indent=2)
        logging.info(f"{annotated_file} written")

        # Visualization logic
        if args.visualise:
            # ------------------------------------------------------------------
            # 1.  keep only the triples we want to see
            # ------------------------------------------------------------------
            type0_triples = []
            logging.info("Selecting type_0 triples for visualization...")
            for it in test:
                if it.get("type") == "type_0":
                    tr = parse(it["input_text"])
                    assert tr not in train_triples, f"duplicate found: {tr!r}"
                    type0_triples.append(tr)

            train_set = set(train_triples)
            type0_set = set(type0_triples)
            
            # ------------------------------------------------------------------
            # 2.  build a *visualisation* graph on this subset (with caching)
            # ------------------------------------------------------------------
            viz_triples = train_triples + type0_triples
            
            # Create cache directory if it doesn't exist
            cache_dir = os.path.join(args.data_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Cache file path for the graph
            cache_file = os.path.join(cache_dir, f"viz_graph_min{args.min_evidence}.pkl")
            
            # Try to load from cache first
            if os.path.exists(cache_file):
                logging.info(f"Loading visualization graph from cache: {cache_file}")
                import pickle
                with open(cache_file, 'rb') as f:
                    G_viz = pickle.load(f)
                logging.info(f"Loaded graph with |V|={G_viz.number_of_nodes()}, |E|={G_viz.number_of_edges()}")
            else:
                logging.info("Building visualization graph (this may take a while)...")
                G_viz = build_full_subst_graph(viz_triples, triple2t, equiv_classes)
                
                # Save to cache for future use
                logging.info(f"Saving visualization graph to cache: {cache_file}")
                import pickle
                with open(cache_file, 'wb') as f:
                    pickle.dump(G_viz, f)

            # ------------------------------------------------------------------
            # 3. compute layout with optimized algorithm
            # ------------------------------------------------------------------
            # Try to use ForceAtlas2 for faster layout if available
            try:
                from fa2_modified import ForceAtlas2
                logging.info("Using ForceAtlas2 for graph layout...")
                
                # Cache the layout calculation
                layout_cache = os.path.join(cache_dir, f"layout_min{args.min_evidence}.pkl")
                
                if os.path.exists(layout_cache):
                    logging.info(f"Loading layout from cache: {layout_cache}")
                    with open(layout_cache, 'rb') as f:
                        pos = pickle.load(f)
                else:
                    logging.info(f"Computing optimized layout for {G_viz.number_of_nodes()} nodes...")
                    
                    # Use ForceAtlas2 for larger graphs
                    if G_viz.number_of_nodes() > 50:
                        forceatlas2 = ForceAtlas2(
                            outboundAttractionDistribution=True,
                            linLogMode=False,
                            adjustSizes=False,
                            edgeWeightInfluence=1.0,
                            jitterTolerance=1.0,
                            barnesHutOptimize=True,
                            barnesHutTheta=1.2,
                            multiThreaded=False,  # Set to True if your installation supports it
                            scalingRatio=2.0,
                            strongGravityMode=False,
                            gravity=1.0,
                            verbose=False
                        )
                        
                        # Compute layout with progress indication
                        from tqdm import tqdm
                        logging.info("Computing ForceAtlas2 layout...")
                        pos = {}
                        iterations = 500
                        
                        # Compute initial positions using spring_layout with few iterations
                        initial_pos = nx.spring_layout(G_viz, seed=0, iterations=5)
                        
                        # Run ForceAtlas2 with progress bar
                        for i in tqdm(range(iterations), desc="Layout iterations"):
                            pos = forceatlas2.forceatlas2_networkx_layout(
                                G_viz, pos=initial_pos if i == 0 else pos, iterations=1
                            )
                    else:
                        # For smaller graphs, spring_layout works well
                        logging.info("Computing spring layout...")
                        pos = nx.spring_layout(G_viz, seed=0, iterations=100)
                        
                    # Save layout to cache
                    logging.info(f"Saving layout to cache: {layout_cache}")
                    with open(layout_cache, 'wb') as f:
                        pickle.dump(pos, f)
            except ImportError:
                logging.info("ForceAtlas2 not available, using spring_layout...")
                pos = nx.spring_layout(G_viz, seed=0)

            # ------------------------------------------------------------------
            # 4.  prepare visualization with batched processing
            # ------------------------------------------------------------------
            import plotly.graph_objects as go
            
            if not os.path.exists('coverage_visualization'):
                os.makedirs('coverage_visualization', exist_ok=True)
            plot_save_dir = os.path.join('coverage_visualization', f"{args.data_dir.split('/')[-2]}_full_min{args.min_evidence}.html")
                
            logging.info(f"Preparing visualization data...")

            buckets = {                 # label → 3 empty lists (x,y,text)
                "train (covered)"       : ([], [], []),
                "type_0 ✓ covered"      : ([], [], []),
                "type_0 ✗ uncovered"    : ([], [], []),
            }

            # Process nodes in batches to avoid memory issues with large graphs
            BATCH_SIZE = 5000
            node_batches = [list(G_viz.nodes())[i:i+BATCH_SIZE] 
                        for i in range(0, len(G_viz.nodes()), BATCH_SIZE)]
            
            for batch in node_batches:
                for n in batch:
                    if n not in pos:
                        logging.warning(f"Node {n} missing from layout positions, skipping")
                        continue
                        
                    x, y = pos[n]
                    txt = str((*n, triple2t[n]))

                    if n in train_set:
                        label = "train (covered)"
                    elif n in type0_set:
                        if n in covered:
                            label = "type_0 ✓ covered"
                        else:
                            label = "type_0 ✗ uncovered"
                    else:
                        continue  # Skip other node types

                    buckets[label][0].append(x)
                    buckets[label][1].append(y)
                    buckets[label][2].append(txt)

            STYLE = {
                "train (covered)"    : dict(symbol="diamond", size=8, color="#1f77b4"),
                "type_0 ✓ covered"   : dict(symbol="circle",  size=7, color="#2ca02c"),
                "type_0 ✗ uncovered" : dict(symbol="x",       size=8, color="#d62728"),
            }
            
            # Create traces only for non-empty buckets
            node_traces = []
            for label, (xs, ys, texts) in buckets.items():
                if not xs:  # Skip empty buckets
                    continue
                    
                node_traces.append(
                    go.Scatter(
                        x=xs, y=ys,
                        mode="markers",
                        name=label,
                        marker=STYLE[label],
                        text=texts,
                        hovertemplate="%{text}",
                    )
                )
            
            # Process edges in batches to avoid memory issues
            logging.info("Processing edges for visualization...")
            edge_x, edge_y = [], []
            
            EDGE_BATCH_SIZE = 10000
            edge_list = list(G_viz.edges())
            edge_batches = [edge_list[i:i+EDGE_BATCH_SIZE] 
                        for i in range(0, len(edge_list), EDGE_BATCH_SIZE)]
            
            from tqdm import tqdm
            for batch in tqdm(edge_batches, desc="Processing edge batches"):
                batch_x, batch_y = [], []
                for u, v in batch:
                    if u not in pos or v not in pos:
                        continue
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    batch_x.extend([x0, x1, None])
                    batch_y.extend([y0, y1, None])
                edge_x.extend(batch_x)
                edge_y.extend(batch_y)
            
            # Create edge trace
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                mode="lines",
                line=dict(width=0.4, color="#bbbbbb"),
                hoverinfo="skip",
                showlegend=False,
            )

            # ------------------------------------------------------------------
            # 5.  create and save the figure
            # ------------------------------------------------------------------
            logging.info("Creating Plotly figure...")
            fig = go.Figure(
                data=[edge_trace] + node_traces,
                layout=go.Layout(
                    title=f"Full Substitution Graph • min_evidence={args.min_evidence} • {len(G_viz.nodes())} nodes, {len(G_viz.edges())} edges",
                    hovermode="closest",
                    margin=dict(l=20, r=20, t=60, b=20),
                    xaxis=dict(visible=False), 
                    yaxis=dict(visible=False),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(255, 255, 255, 0.8)"
                    ),
                ),
            )
            
            logging.info(f"Writing HTML graph to {plot_save_dir}...")
            fig.write_html(plot_save_dir)
            logging.info(f"HTML visualization saved → {plot_save_dir}")


if __name__ == "__main__":
    main()
