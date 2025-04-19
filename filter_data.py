#!/usr/bin/env python3
"""
check_pure_type0.py  –  scan (and optionally filter) every dataset directory
                        under a common root.

For each sub‑directory that contains *train.json* and *test.json*:

1. Count rows in test.json whose `"type"` is "type_0".
2. Among those, count the rows whose triple is **not** found in train.json
   (“pure type_0”).
3. Report statistics.
4. If --filter is given *and* pure_count ≥ --min_pure
      • keep a reproducible random sample of exactly --min_pure pure rows
      • keep every other test row (non‑type_0 or non‑pure duplicates)
      • overwrite test.json with this filtered list
   else
      • leave the dataset untouched.

Exit code is non‑zero iff
   – `--strict` is passed **and** at least one dataset has pure_count < min_pure
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

# ── helpers ────────────────────────────────────────────────────────────────
def load_json(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, data: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def parse_triple(s: str) -> Tuple[int, int, int]:
    return tuple(int(tok.split("_")[-1]) for tok in s.strip("<>").split("><"))

# ── pure‑count + optional filter ------------------------------------------
def analyse_and_maybe_filter(ds_dir: Path,
                             min_pure: int,
                             do_filter: bool,
                             rng: random.Random) -> Tuple[int, int, bool]:
    """
    Return (pure, total_type0, filtered_flag).
    If `do_filter` and pure≥min_pure, overwrite test.json with filtered version
    and set filtered_flag=True.
    """
    train = load_json(ds_dir / "train.json")
    test  = load_json(ds_dir / "test.json")

    train_triples = {parse_triple(r["input_text"]) for r in train}

    pure_rows, nonpure_rows = [], []
    for row in test:
        if row.get("type") == "type_0":
            if parse_triple(row["input_text"]) not in train_triples:
                pure_rows.append(row)
            else:
                # nonpure_rows.append(row)        # duplicate of train
                continue
        else:
            nonpure_rows.append(row)            # any other type

    pure_cnt   = len(pure_rows)
    total_type = pure_cnt + sum(1 for r in nonpure_rows if r.get("type")=="type_0")

    filtered = False
    if do_filter and pure_cnt >= min_pure:
        # ── (a)  exact 2 000 *pure* type_0 rows ────────────────────────────
        sampled_pure = rng.sample(pure_rows, k=min_pure)

        # ── (b)  collect *all* remaining rows, bucketed by their "type" ────
        buckets = {}                      # label -> list[Dict]
        for row in nonpure_rows:
            buckets.setdefault(row.get("type", "UNK"), []).append(row)

        # ── (c)  down‑sample every bucket to ≤ 2 000 rows (reproducible) ───
        trimmed_rest = []
        for label, rows in buckets.items():
            if len(rows) > min_pure:
                rows = rng.sample(rows, k=min_pure)
            trimmed_rest.extend(rows)

        # ── (d)  write back ------------------------------------------------
        new_test = sampled_pure + trimmed_rest
        save_json(ds_dir / "test.json", new_test)
        filtered = True


    return pure_cnt, total_type, filtered

# ── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="root directory containing dataset subdirs")
    ap.add_argument("--min_pure", type=int, default=2000,
                    help="minimum (and, with --filter, *target*) number of pure type_0 rows")
    ap.add_argument("--strict", action="store_true",
                    help="exit non‑zero if any dataset falls below threshold")
    ap.add_argument("--filter", action="store_true",
                    help="if threshold satisfied, down‑sample pure rows to exactly min_pure")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for reproducible sampling (default 42)")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    root = Path(args.root).expanduser().resolve()
    assert root.is_dir(), f"{root} is not a directory"

    failed, filtered_ds = [], []

    print(f"Scanning datasets in {root}  (threshold {args.min_pure})\n")
    hdr = f"{'dataset':<30} {'pure':>5}/{ 'total':<5}   status"
    if args.filter:
        hdr += "   action"
    print(hdr)
    print("-"*len(hdr))

    for ds in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (ds/"train.json").exists() or not (ds/"test.json").exists():
            continue

        pure, total, filtered = analyse_and_maybe_filter(
            ds, args.min_pure, args.filter, rng
        )
        status = "OK" if pure >= args.min_pure else "FAIL"
        action = "filtered" if filtered else "-"
        if status == "FAIL":
            failed.append(ds.name)
        if filtered:
            filtered_ds.append(ds.name)

        line = f"{ds.name:<30} {pure:>5}/{total:<5}   {status}"
        if args.filter:
            line += f"   {action}"
        print(line)

    # summary
    if failed:
        msg = "Datasets below threshold: " + ", ".join(failed)
        if args.strict:
            raise SystemExit(msg)
        else:
            print("\n" + msg)
    if filtered_ds:
        print("\nDatasets filtered to exactly "
              f"{args.min_pure} pure rows: " + ", ".join(filtered_ds))

if __name__ == "__main__":
    main()
