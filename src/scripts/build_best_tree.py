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
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

from gold_tree import DEFAULT_TREE

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

def mantel_like(dist_a, dist_b):
    a = np.asarray(dist_a).ravel()
    b = np.asarray(dist_b).ravel()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    return float(np.dot(a, b) / (len(a) - 1))

def linkage_to_newick(Z, labels):
    """
    Convert scipy linkage matrix Z to Newick string with given labels.
    Produces an unrooted-like Newick (no branch lengths). For branch lengths,
    one can subtract child heights from parent height; here we include lengths.
    """
    from collections import defaultdict
    n = len(labels)
    # height of each node from linkage
    heights = {i:0.0 for i in range(n)}
    children = {}
    for idx, (c1, c2, dist, _) in enumerate(Z, start=n):
        c1, c2 = int(c1), int(c2)
        children[idx] = (c1, c2)
        heights[idx] = dist

    def node_newick(i):
        if i < n:
            return f"{labels[i]}:0.0"
        c1, c2 = children[i]
        h = heights[i]
        # branch lengths = parent height - child height
        bl1 = max(h - heights[c1], 0.0)
        bl2 = max(h - heights[c2], 0.0)
        left = node_newick(c1)
        right = node_newick(c2)
        # replace trailing :0.0 on children with computed lengths
        left = left.rsplit(":", 1)[0] + f":{bl1:.6f}"
        right = right.rsplit(":", 1)[0] + f":{bl2:.6f}"
        return f"({left},{right})"

    root = len(Z) + n - 1
    return node_newick(root) + ";"


def plot_dendrogram_plotly_colored(Z, langs, lang_to_color, title="", width=1200, height=700):
    """
    Build a Plotly dendrogram where each branch segment is colored
    by the color of its rightmost leaf (language color).
    """
    # Use SciPy dendrogram (no plotting) to get coordinates
    ddata = dendrogram(Z, labels=langs, no_plot=True)

    # ddata["icoord"] and ddata["dcoord"] list the x/y points for each U-link
    icoord = ddata["icoord"]
    dcoord = ddata["dcoord"]
    leaf_order = ddata["ivl"]   # leaf labels in display order

    # Map leaf label -> color (fallback black)
    leaf_color = {lab: lang_to_color.get(lab, "#000000") for lab in leaf_order}

    # Build Plotly traces for each segment
    # Strategy: color each U-link by the color of the segment that touches the rightmost x of that U.
    # That makes branch colors flow consistently toward a leaf color.
    traces = []
    for xs, ys in zip(icoord, dcoord):
        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)

        # We choose the rightmost x position; find the nearest leaf tick to that x
        x_right = xs.max()
        # The scipy dendrogram places leaves at x=5,15,25,... so map to nearest leaf index:
        # ddata['leaves'] has integer ordering, but for a robust approach,
        # find the closest tick label by minimal absolute x-distance to leaf positions:
        # The leaf positions are in ddata['leaves_color_list'] order, but we can infer them from x of tick labels
        # Instead, use ddata['ivl'] order and the tick positions observed: [5, 15, 25, ...]
        leaf_positions = [5 + 10*i for i in range(len(leaf_order))]
        nearest_idx = int(np.argmin([abs(x_right - xp) for xp in leaf_positions]))
        col = leaf_color[leaf_order[nearest_idx]]

        # Add the polyline as a separate trace
        traces.append(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color=col, width=2.5),
            hoverinfo="skip",
            showlegend=False
        ))

    # Build tick labels at leaf positions with rotation
    x_ticks = [5 + 10*i for i in range(len(leaf_order))]
    x_text = leaf_order

    fig = go.Figure(data=traces)
    fig.update_layout(
        width=width, height=height,
        title=title,
        xaxis=dict(
            tickmode="array",
            tickvals=x_ticks,
            ticktext=x_text,
            tickfont=dict(size=15),
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title="Distance",
            showgrid=True,
            zeroline=False
        ),
        plot_bgcolor="white",
        margin=dict(l=40, r=20, t=60, b=120)
    )
    # rotate labels (Plotly way: use tickangle)
    fig.update_xaxes(tickangle=60)

    # Subtle axis styling
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.08)")
    return fig

def main():
    ap = argparse.ArgumentParser(description="Find best feature subset against a gold phylogeny, then build/plot dendrogram (and optionally export Newick).")
    ap.add_argument("--input", required=True, help="CSV with data.")
    ap.add_argument("--lang-col", default="stimulus_name", help="Language name column.")
    ap.add_argument("--outdir", default=".", help="Where to write outputs.")
    ap.add_argument("--tree-json", default=None, help="Optional override JSON for the gold tree.")
    ap.add_argument("--max-k", type=int, default=3, help="Max feature subset size to search (1, 2, or 3).")
    ap.add_argument("--export", action="store_true", help="Also export dendrogram as Newick (.nwk).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.lang_col not in df.columns:
        print(f"ERROR: column '{args.lang_col}' not in CSV.", file=sys.stderr)
        sys.exit(1)
    languages_not_used = ["Swiss German", "Romansh", "Cantonese", "Kalaallisut", "Hausa"]
    df = df[~df[args.lang_col].isin(languages_not_used)].copy()
    langs = df[args.lang_col].astype(str).tolist()
    num_df = df.select_dtypes(include=[np.number]).copy()
    features = num_df.columns.tolist()
    if not features:
        print("ERROR: no numeric features detected.", file=sys.stderr)
        sys.exit(1)

    # Load/prepare tree
    tree = DEFAULT_TREE
    if args.tree_json:
        with open(args.tree_json, "r", encoding="utf-8") as f:
            tree = json.load(f)

    macro, family, sub = build_label_maps(tree)
    gold = make_gold_distance(langs, macro, family, sub)
    gold_path = os.path.join(args.outdir, "gold_tree_distances.csv")
    pd.DataFrame(gold, index=langs, columns=langs).to_csv(gold_path)
    gold_condensed = squareform(gold, checks=False)

    # Search best subset
    best = {"subset": None, "r": -np.inf}
    
    # singletons
    for f in features:
        X = num_df[[f]].values.astype(float)
        s = X.std()
        if s == 0: 
            continue
        X = (X - X.mean()) / s
        r = mantel_like(pdist(X, "euclidean"), gold_condensed)
        if r > best["r"]:
            best = {"subset": (f,), "r": r}
    '''
    # pairs
    if args.max_k >= 2:
        for f1, f2 in itertools.combinations(features, 2):
            X = num_df[[f1, f2]].values.astype(float)
            std = X.std(axis=0, ddof=0); std[std==0] = 1.0
            X = (X - X.mean(axis=0)) / std
            r = mantel_like(pdist(X, "euclidean"), gold_condensed)
            if r > best["r"]:
                best = {"subset": (f1, f2), "r": r}
    # triplets
    if args.max_k >= 3:
    '''
    features = ["num. types",
                "function words ratio",
                "word length"]
    for ftrs in itertools.combinations(features, args.max_k):
        ftrs = list(ftrs)
        X = num_df[ftrs].values.astype(float)
        std = X.std(axis=0, ddof=0); std[std==0] = 1.0
        X = (X - X.mean(axis=0)) / std
        r = mantel_like(pdist(X, "euclidean"), gold_condensed)
        if r > best["r"]:
            best = {"subset": ftrs, "r": r}

    subset = best["subset"]
    rbest = best["r"]
    if subset is None:
        print("ERROR: could not find a valid subset.", file=sys.stderr)
        sys.exit(1)

    # Build dendrogram for the best subset
    X = df[list(subset)].values.astype(float)
    std = X.std(axis=0, ddof=0); std[std==0] = 1.0
    X = (X - X.mean(axis=0)) / std
    Z = linkage(X, method="ward")

    from src.language_constants import LANG_COLORS, CODE2LANG
    lang_to_color = {}
    for lang in LANG_COLORS:
        lang_label = CODE2LANG.get(lang, lang)
        lang_to_color[lang_label] = LANG_COLORS[lang]

    fig, ax = plt.subplots(figsize=(14, 7))

    # Draw dendrogram (leaf_rotation controls text rotation)
    ddata = dendrogram(
        Z,
        labels=langs,
        leaf_rotation=60,      # rotate labels (change to 90 if you prefer)
        leaf_font_size=15,
        color_threshold=1.5,     # ensure matplotlib doesn't recolor links automatically
        above_threshold_color="#000000"
    )

    # Make it cleaner
    #ax.set_title(f"Dendrogram using best features {subset} (Mantel r vs GOLD = {rbest:.3f})", pad=14)
    #ax.set_xlabel("Language", labelpad=8)
    #ax.set_ylabel("Distance")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(False)

    # Color each x-axis tick label by its language color
    for tick in ax.get_xticklabels():
        lang_label = tick.get_text()
        tick.set_color(lang_to_color.get(lang_label, "#000000"))

    # Optional: add a colored dot underneath each leaf for extra clarity
    # 'ivl' is the list of leaf labels in the order they appear; 'leaves' holds leaf positions
    #leaf_labels = ddata["ivl"]
    #leaf_positions = ddata["leaves"]
    #ymin, ymax = ax.get_ylim()
    #y_dot = ymin - 0.02 * (ymax - ymin)  # slightly below the axis range

    plt.tight_layout()

    # Save high-quality outputs
    dendro_png = os.path.join(args.outdir, "dendrogram_best_features.png")
    dendro_svg = os.path.join(args.outdir, "dendrogram_best_features.svg")
    plt.savefig(dendro_png, dpi=300, bbox_inches="tight")
    plt.savefig(dendro_svg, bbox_inches="tight")
    plt.close(fig)

    # Save subset and score
    meta_path = os.path.join(args.outdir, "best_subset.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"best_subset": list(subset), "mantel_r_vs_gold": float(rbest)}, f, indent=2)

    # Optional Newick (flag name fixed to --export-newick)
    if getattr(args, "export_newick", False):
        newick = linkage_to_newick(Z, langs)
        newick_path = os.path.join(args.outdir, "dendrogram_best_features.nwk")
        with open(newick_path, "w", encoding="utf-8") as f:
            f.write(newick)

    print("Best subset:", subset)
    print("Mantel r vs GOLD:", rbest)
    print("Saved:")
    print(" -", gold_path)
    print(" -", dendro_png)
    print(" -", dendro_svg)
    print(" -", meta_path)
    if getattr(args, "export_newick", False):
        print(" -", newick_path)



if __name__ == "__main__":
    main()
