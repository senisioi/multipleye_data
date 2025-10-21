#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage as olinkage
from scipy.cluster.hierarchy import dendrogram, cophenet
from sklearn.preprocessing import StandardScaler
from elinkage import elinkage
from gold_tree import DEFAULT_TREE

METHOD = 'ward'  # linkage method


def linkage(X, method=METHOD):
    if method == 'elinkage':
        return elinkage(X.tolist())
    return olinkage(X, method=method)



def build_label_maps(tree_dict):
    macro, family, sub = {}, {}, {}
    for mac, lst in tree_dict.get("macro", {}).items():
        for name in lst:
            macro[name] = mac
    for fam, subs in tree_dict.get("families", {}).items():
        for subname, names in subs.items():
            for name in names:
                family[name] = fam
                sub[name] = subname
                if name not in macro:
                    macro[name] = fam
    return macro, family, sub

def make_gold_distance(langs, macro, family, sub):
    n = len(langs)
    gold = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            li, lj = langs[i], langs[j]
            if li == lj:
                d = 0.0
            elif macro.get(li) == macro.get(lj):
                if family.get(li) == family.get(lj):
                    d = 1.0 if sub.get(li) == sub.get(lj) else 2.0
                else:
                    d = 3.0
            else:
                d = 4.0
            gold[i, j] = gold[j, i] = d
    return gold

def get_cophenetic_dists(Z):
    """Return condensed cophenetic distances robustly across SciPy versions."""
    ret = cophenet(Z)
    # Some SciPy versions return only the condensed distances; others (coef, dists)
    if isinstance(ret, tuple) or isinstance(ret, list):
        return ret[1]
    return ret

def mantel_like(dist_a, dist_b):
    a = np.asarray(dist_a).ravel()
    b = np.asarray(dist_b).ravel()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    return float(np.dot(a, b) / (len(a) - 1))

def to_newick(Z, labels):
    """Export a Newick string from a SciPy linkage matrix (with branch lengths)."""
    n = len(labels)
    heights = {i:0.0 for i in range(n)}
    children = {}
    for idx, (c1, c2, dist, _) in enumerate(Z, start=n):
        c1, c2 = int(c1), int(c2)
        children[idx] = (c1, c2)
        heights[idx] = float(dist)

    def rec(i):
        if i < n:
            return f"{labels[i]}:0.0"
        c1, c2 = children[i]
        h = heights[i]
        left = rec(c1)
        right = rec(c2)
        bl1 = max(h - heights[c1], 0.0)
        bl2 = max(h - heights[c2], 0.0)
        left = left.rsplit(":", 1)[0] + f":{bl1:.6f}"
        right = right.rsplit(":", 1)[0] + f":{bl2:.6f}"
        return f"({left},{right})"
    root = len(Z) + n - 1
    return rec(root) + ";"

def parse_args():
    p = argparse.ArgumentParser(description="Build Ward dendrogram using chosen feature set; compare to GOLD tree; draw tree.")
    p.add_argument("--input", required=True, help="CSV with data (must contain language names + numeric features).")
    p.add_argument("--lang-col", default="stimulus_name", help="Column name for languages.")
    p.add_argument("--mode", choices=["all","list","best"], required=True,
                   help="Feature selection mode: 'all' numeric features, 'list' of specific features, or 'best' auto-search.")
    p.add_argument("--features", default=None,
                   help="Comma-separated list of features (only used if --mode list).")
    p.add_argument("--max-k", type=int, default=3, choices=[1,2,3],
                   help="Max subset size to search in 'best' mode (1..3).")
    p.add_argument("--tree-json", default=None, help="Optional JSON file to override the default GOLD tree.")
    p.add_argument("--outdir", default=".", help="Output directory.")
    p.add_argument("--export", action="store_true", help="Also export the dendrogram in Newick format.")
    return p.parse_args()



def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.lang_col not in df.columns:
        print(f"ERROR: '{args.lang_col}' not found in input.", file=sys.stderr)
        sys.exit(1)

    langs = df[args.lang_col].astype(str).tolist()
    num_df = df.select_dtypes(include=[np.number]).copy()
    all_features = num_df.columns.tolist()
    if not all_features:
        print("ERROR: No numeric features found.", file=sys.stderr)
        sys.exit(1)

    # GOLD tree distances
    tree = DEFAULT_TREE
    if args.tree_json:
        with open(args.tree_json, "r", encoding="utf-8") as f:
            tree = json.load(f)
    macro, family, sub = build_label_maps(tree)
    gold = make_gold_distance(langs, macro, family, sub)
    gold_condensed = squareform(gold, checks=False)
    gold_csv = os.path.join(args.outdir, "gold_tree_distances.csv")
    pd.DataFrame(gold, index=langs, columns=langs).to_csv(gold_csv)

    # --- Feature selection per mode ---
    if args.mode == "all":
        chosen = all_features
    elif args.mode == "list":
        if not args.features:
            print("ERROR: --features must be provided in 'list' mode.", file=sys.stderr)
            sys.exit(1)
        chosen = [f.strip() for f in args.features.split(",")]
        missing = [f for f in chosen if f not in all_features]
        if missing:
            print("ERROR: Some requested features are not numeric or missing:", missing, file=sys.stderr)
            sys.exit(1)
    else:  # best
        best = {"subset": None, "r": -np.inf}
        # 1-feature
        for f in all_features:
            X = num_df[[f]].values.astype(float)
            s = X.std()
            if s == 0: 
                continue
            X = (X - X.mean()) / s
            # Compare *cophenetic distances* of the Ward tree to GOLD
            Z = linkage(X, method=METHOD)
            coph = get_cophenetic_dists(Z)
            r = mantel_like(coph, gold_condensed)
            if r > best["r"]:
                best = {"subset": (f,), "r": r}
        # 2-features
        if args.max_k >= 2:
            for f1, f2 in itertools.combinations(all_features, 2):
                X = num_df[[f1, f2]].values.astype(float)
                std = X.std(axis=0, ddof=0); std[std==0] = 1.0
                X = (X - X.mean(axis=0)) / std
                Z = linkage(X, method=METHOD)
                coph = get_cophenetic_dists(Z)
                r = mantel_like(coph, gold_condensed)
                if r > best["r"]:
                    best = {"subset": (f1, f2), "r": r}
        # 3-features
        if args.max_k >= 3:
            for f1, f2, f3 in itertools.combinations(all_features, 3):
                X = num_df[[f1, f2, f3]].values.astype(float)
                std = X.std(axis=0, ddof=0); std[std==0] = 1.0
                X = (X - X.mean(axis=0)) / std
                Z = linkage(X, method=METHOD)
                coph = get_cophenetic_dists(Z)
                r = mantel_like(coph, gold_condensed)
                if r > best["r"]:
                    best = {"subset": (f1, f2, f3), "r": r}
        chosen = list(best["subset"])
        best_r = best["r"]
        # Save the auto-search meta
        meta_path = os.path.join(args.outdir, "best_subset.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"best_subset": chosen, "mantel_r_vs_gold": float(best_r)}, f, indent=2)

    # --- Build final Ward linkage with chosen features ---
    X = df[chosen].values.astype(float)
    std = X.std(axis=0, ddof=0); std[std == 0] = 1.0
    X = (X - X.mean(axis=0)) / std
    Z = linkage(X, method=METHOD)

    # --- Compute similarity to GOLD using cophenetic distances ---
    coph = get_cophenetic_dists(Z)
    r_final = mantel_like(coph, gold_condensed)

    # --- Draw & save dendrogram ---
    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=langs, leaf_rotation=90)
    title_subset = ", ".join(chosen) if len(chosen) <= 5 else f"{len(chosen)} features"
    plt.title(f"Ward dendrogram ({title_subset}) | Mantel r vs GOLD = {r_final:.3f}")
    plt.xlabel("Language")
    plt.ylabel("Distance")
    plt.tight_layout()
    dendro_png = os.path.join(args.outdir, "dendrogram.png")
    plt.savefig(dendro_png, dpi=200)
    plt.close()

    # Optional Newick
    if args.export:
        newick = to_newick(Z, langs)
        with open(os.path.join(args.outdir, "dendrogram.nwk"), "w", encoding="utf-8") as f:
            f.write(newick)

    print("Chosen features:", chosen)
    print("Mantel r vs GOLD:", r_final)
    print("Saved:")
    print(" -", gold_csv)
    print(" -", dendro_png)
    if args.mode == "best":
        print(" -", os.path.join(args.outdir, "best_subset.json"))
    if args.export:
        print(" -", os.path.join(args.outdir, "dendrogram.nwk"))

if __name__ == "__main__":
    main()
