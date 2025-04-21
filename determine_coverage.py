#!/usr/bin/env python
"""coverage_dag.py – strict equivalence + Plotly hover

Fixes after debugging feedback
──────────────────────────────
1. **Equivalence classes** now require *identical observed behaviour*:
   two pairs (h1,h2) and (h1',h2') are equivalent **iff** their whole
   observed mapping  {h3 → t}  is identical (not just consistent on the
   intersection).  That guarantees they share the same (latent) f1 output
   in a two‑hop task.

2. **Substitution graph** edges are added only between triples that
   (a) share the same equivalence class for (h1,h2),
   (b) keep h3 fixed, **and**
   (c) have the same final target *t*.
   Hence vertices with different targets or different f1 outcomes never
   connect, eliminating the spurious fully‑connected components you saw.

No other behaviour was altered; Plotly visualisation still lets you hover
for *(h1,h2,h3,t)*.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple, Set

import networkx as nx

###############################################################################
# helpers
###############################################################################

def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


def parse_input_tokens(s: str) -> Tuple[int, int, int]:
    return tuple(int(tok.split("_")[-1]) for tok in s.strip("<>").split("><"))

###############################################################################
# Union–Find for equivalence classes
###############################################################################

class UnionFind:
    def __init__(self):
        self.parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.size: Dict[Tuple[int, int], int] = {}

    def find(self, x: Tuple[int, int]) -> Tuple[int, int]:
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: Tuple[int, int], b: Tuple[int, int]):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

###############################################################################
# Build equivalence classes (strict)
###############################################################################

def build_equiv_classes(train_map: Dict[Tuple[int,int,int], int], min_evidence: int = 1) -> UnionFind:
    """
    Two (h1,h2) slices are equivalent iff
      ⋆ there exist at least 'min_evidence' distinct h3 values such that both 
        (h1,h2,h3) and (h1',h2',h3) are in the training data and their targets match,  **and**
      ⋆ on every h3 that appears for *both* slices the targets match.
    """
    # 1) gather partial maps  (h1,h2) → {h3 : t}
    behaviour = defaultdict(dict)
    for (h1,h2,h3), t in train_map.items():
        behaviour[(h1,h2)][h3] = t

    # 2) Count shared evidence between each pair of slices
    pair_evidence = defaultdict(int)  # Key: ((h1,h2), (h1',h2')), Value: count of shared h3 values
    contradictions = set()  # Store pairs that have any contradictions
    
    # First pass: count shared evidence and check for contradictions
    for (pair1, h3map1), (pair2, h3map2) in combinations(behaviour.items(), 2):
        if pair1 >= pair2:  # Skip duplicate combinations
            continue
            
        # Find shared h3 values
        shared_h3 = set(h3map1).intersection(h3map2)
        
        # Check for contradictions
        for h3 in shared_h3:
            if h3map1[h3] != h3map2[h3]:
                contradictions.add((pair1, pair2))
                break
        
        # If no contradictions and they share any h3, count the evidence
        if (pair1, pair2) not in contradictions and shared_h3:
            # Count how many h3 values they share with matching output
            matching_evidence = sum(1 for h3 in shared_h3 if h3map1[h3] == h3map2[h3])
            pair_evidence[(pair1, pair2)] = matching_evidence

    # 3) union any two slices that have at least min_evidence shared h3 values
    # and no contradictions
    uf = UnionFind()
    
    for (pair1, pair2), evidence_count in pair_evidence.items():
        if evidence_count >= min_evidence and (pair1, pair2) not in contradictions:
            uf.union(pair1, pair2)

    logging.info("Equivalence classes (min evidence = %d): %d",
                 min_evidence, len({uf.find(p) for p in behaviour}))
    return uf

###############################################################################
# Substitution graph & coverage
###############################################################################

def build_subst_graph(all_triples: List[Tuple[int, int, int]], triple2t: Dict[Tuple[int, int, int], int], uf: UnionFind) -> nx.Graph:
    G = nx.Graph()
    for tr in all_triples:
        G.add_node(tr)

    buckets: Dict[Tuple[Tuple[int, int], int, int], List[Tuple[int, int, int]]] = defaultdict(list)
    for tr in all_triples:
        h1, h2, h3 = tr
        buckets[(uf.find((h1, h2)), h3, triple2t[tr])].append(tr)

    for triples in buckets.values():
        for a, b in combinations(triples, 2):
            G.add_edge(a, b)
    logging.info("Substitution graph |V|=%d  |E|=%d", G.number_of_nodes(), G.number_of_edges())
    return G


def compute_coverage(G: nx.Graph, train_triples: List[Tuple[int, int, int]]) -> Set[Tuple[int, int, int]]:
    covered: Set[Tuple[int, int, int]] = set()
    for tr in train_triples:
        if tr not in covered:
            covered.update(nx.node_connected_component(G, tr))
    logging.info("Covered nodes: %d", len(covered))
    return covered

###############################################################################
# main
###############################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--visualise", help=".html for interactive Plotly graph")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--min_evidence", type=int, default=1,
                       help="minimum evidence for equivalence classes")
    args = ap.parse_args()

    setup_logging(args.debug)

    def jload(name):
        with open(os.path.join(args.data_dir, name), "r", encoding="utf-8") as f:
            return json.load(f)

    train = jload("train.json")
    test = jload("test.json")

    parse = parse_input_tokens

    train_triples = [parse(it["input_text"]) for it in train]
    test_triples = [parse(it["input_text"]) for it in test]
    test_triples = [d for d in test_triples if d not in train_triples]  # remove duplicates
    logging.info("train triples: %d  test triples: %d", len(train_triples), len(test_triples))
    assert len(train_triples) == len(set(train_triples)), "duplicate triples in train.json"
    assert len(test_triples) == len(set(test_triples)), "duplicate triples in test.json"

    def label(it):
        t_tok = it["target_text"].replace("</a>", "").strip("<>").split("><")[-1]
        return int(t_tok.split("_")[-1])

    triple2t: Dict[Tuple[int, int, int], int] = {parse(it["input_text"]): label(it) for it in train + test}

    train_map = {tr: triple2t[tr] for tr in train_triples}

    uf = build_equiv_classes({(*tr,): triple2t[tr] for tr in train_triples}, min_evidence=args.min_evidence)  # cast keys
    G = build_subst_graph(train_triples + test_triples, triple2t, uf)
    covered = compute_coverage(G, train_triples)

    # annotate test.json
    for it in test:
        it["coverage"] = bool(parse(it["input_text"]) in covered)
        
    # ------------------------------------------------------------------
    # ❶  Percentage‑covered-by‑type report
    # ------------------------------------------------------------------
    from collections import defaultdict

    totals = defaultdict(int)
    hits   = defaultdict(int)

    for it in test:                                  # test already has "coverage" flag
        typ = it.get("type", "UNK")                  # fall back if field is missing
        totals[typ] += 1
        if it["coverage"]:
            hits[typ] += 1
        # if typ=="type_3" and it["coverage"]:
        #     logging.info("type_3 coverage: %s", it["input_text"])

    for typ in sorted(totals):
        pct = 100.0 * hits[typ] / totals[typ]
        logging.info("Coverage [%s] : %d / %d  (%.2f%%)", typ, hits[typ], totals[typ], pct)

        
    with open(os.path.join(args.data_dir, "test_annotated.json"), "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2)
    logging.info("test_annotated.json written")

    # ------------------------------------------------------------------
    # 1.  keep only the triples we want to see
    # ------------------------------------------------------------------
    type0_triples = []
    for it in test:
        if it.get("type") == "type_0":
            tr = parse(it["input_text"])
            assert tr not in train_triples, f"duplicate found: {tr!r}"
            type0_triples.append(tr)

    train_set  = set(train_triples)
    type0_set  = set(type0_triples)

    # ------------------------------------------------------------------
    # 2.  build a *visualisation* graph on this subset ------------------
    #     (equivalence classes still come only from training pairs!)
    # ------------------------------------------------------------------
    viz_triples = train_triples + type0_triples + type3_triples  # Add type3_triples
    G_viz       = build_subst_graph(viz_triples, triple2t, uf)

    # ------------------------------------------------------------------
    # 3.  colour & shape buckets  ---------------------------------------
    # ------------------------------------------------------------------
    import plotly.graph_objects as go
    logging.info("Writing Plotly graph → %s", args.visualise)

    pos = nx.spring_layout(G_viz, seed=0)

    buckets = {                 # label → 3 empty lists (x,y,text)
        "train (covered)"       : ([], [], []),
        "type_0 ✓ covered"      : ([], [], []),
        "type_0 ✗ uncovered"    : ([], [], []),
    }

    for n in G_viz.nodes():
        x, y = pos[n]
        txt  = str((*n, triple2t[n]))

        if n in train_set:
            label = "train (covered)"
        elif n in type0_set:  # Change this to be more specific about type_0
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
        "type_3 ✓ covered"   : dict(symbol="star",    size=8, color="#9467bd"),  # Add these two lines
        "type_3 ✗ uncovered" : dict(symbol="cross",   size=8, color="#ff7f0e"),  # with distinct markers
    }
    node_traces = [
        go.Scatter(
            x=xs, y=ys,
            mode="markers",
            name=label,
            marker=STYLE[label],
            text=texts,
            hovertemplate="%{text}",
        )
        for label, (xs, ys, texts) in buckets.items()
    ]

    # light grey background edges
    edge_x, edge_y = [], []
    for u, v in G_viz.edges():
        x0, y0 = pos[u];  x1, y1 = pos[v]
        edge_x += [x0, x1, None];  edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.4, color="#bbbbbb"),
        hoverinfo="skip",
        showlegend=False,
    )

    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            title="Substitution Graph • hover shows (h1,h2,h3,t)",
            hovermode="closest",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        ),
    )
    fig.write_html(args.visualise)
    logging.info("HTML saved → %s", args.visualise)



if __name__ == "__main__":
    main()
