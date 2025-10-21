#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a LaTeX table from a CSV with:
- Language column (left-aligned)
- A configurable set of feature columns (right-aligned)
- Soft row colors based on a 'color' hex column (if present)
- Robust LaTeX escaping and booktabs formatting

Usage:
  python make_table.py input.csv [output.tex]

Configure the desired columns by setting the global FEATURES list below.
If FEATURES = [], the script will use all numeric columns (excluding the language column).
"""

import os
import re
import sys
import math
import pandas as pd

# =======================
# CONFIGURE HERE
# =======================
# Put the *exact* feature column names you want (in order), excluding the language column.
# If left empty, the script will include all numeric columns (excluding the language column).
FEATURES: list[str] = ["num. words", "num. types", "type token ratio", "punct. ratio", "function words ratio", "num. sentences", "word length"]

# Language column name (falls back to first column if not found)
LANG_COL_DEFAULT = "stimulus_name"

# Row color intensity (0..100) applied as "!{SOFT_PCT}"
SOFT_PCT = 20

# Caption/label
CAPTION = f"Linguistic features by language. Row colors softened to {SOFT_PCT}\\% intensity."
LABEL = "tab:lang_features_param"
# =======================


def latex_escape(text: str) -> str:
    """Minimal but safe LaTeX escaping for table cells."""
    return (str(text)
            .replace("\\", r"\textbackslash{}")
            .replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
            .replace("#", r"\#").replace("_", r"\_")
            .replace("{", r"\{").replace("}", r"\}")
            .replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}"))


def is_number(x) -> bool:
    """Return True if x behaves like a number (excluding bools)."""
    return (isinstance(x, (int, float)) or pd.api.types.is_number(x)) and not isinstance(x, bool)


def fmt_cell(val: object) -> str:
    """Format numeric cells: floats -> 2 decimals, ints -> no decimals, else as string."""
    if pd.isna(val):
        return ""
    if is_number(val):
        fval = float(val)
        if math.isfinite(fval) and not float(fval).is_integer():
            return f"{fval:.2f}"
        try:
            return f"{int(round(fval))}"
        except Exception:
            return f"{fval:.2f}"
    return latex_escape(val)


def collect_color_names(df: pd.DataFrame) -> dict[str, str]:
    """Collect unique valid HEX colors and map to LaTeX color names."""
    if "color" not in df.columns:
        return {}
    unique_hex = []
    for raw in df["color"].astype(str).str.strip().str.upper():
        hex6 = raw.lstrip("#")
        if re.fullmatch(r"[0-9A-F]{6}", hex6) and hex6 not in unique_hex:
            unique_hex.append(hex6)
    return {h: f"rowclr{idx+1}" for idx, h in enumerate(unique_hex)}


def main():
    if len(sys.argv) < 2:
        print("Usage: python make_table.py input.csv [output.tex]", file=sys.stderr)
        sys.exit(1)

    infis = sys.argv[1]
    outf = None if len(sys.argv) < 3 else sys.argv[2]
    if not os.path.isfile(infis):
        print(f"ERROR: File not found: {infis}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(infis)

    # Determine language column
    lang_col = LANG_COL_DEFAULT if LANG_COL_DEFAULT in df.columns else df.columns[0]

    # Determine features
    if FEATURES:
        features = FEATURES[:]
        missing = [c for c in features if c not in df.columns]
        if missing:
            print(f"ERROR: Requested features not found in CSV: {missing}", file=sys.stderr)
            sys.exit(3)
    else:
        # Use all numeric columns except the language column and 'color'
        numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in {lang_col, "color"}]
        if not numeric_cols:
            print("ERROR: No numeric columns found for FEATURES and FEATURES was empty.", file=sys.stderr)
            sys.exit(4)
        features = numeric_cols

    # Subset in order
    cols = [lang_col] + features
    sub = df[cols].copy()

    # Format feature columns
    for c in features:
        sub[c] = sub[c].map(fmt_cell)

    # Prepare color definitions
    color_names = collect_color_names(df)
    use_colors = bool(color_names)

    preamble = (
        "% LaTeX requirements:\n"
        "% \\usepackage[table,xcdraw]{xcolor} % load BEFORE hyperref\n"
        "% \\usepackage{booktabs}\n"
    )

    color_defs = ""
    if use_colors:
        color_defs = "\n".join([fr"\definecolor{{{name}}}{{HTML}}{{{h}}}" for h, name in color_names.items()])

    # Build header
    align = "l" + "r" * len(features)  # Language left, features right
    header_cells = [r"\textbf{Language}"] + [fr"\textbf{{{latex_escape(c)}}}" for c in features]
    header_row = " & ".join(header_cells) + r" \\"

    header = (
        "\\begin{table*}[ht]\n"
        "\\centering\n"
        "\\scriptsize\n"
        "\\setlength{\\tabcolsep}{6pt}\n"
        "\\renewcommand{\\arraystretch}{1.15}\n"
        f"\\caption{{{CAPTION}}}\n"
        f"\\label{{{LABEL}}}\n"
        f"\\begin{{tabular}}{{@{{}}{align}@{{}}}}\n"
        "\\toprule\n"
        f"{header_row}\n"
        "\\midrule\n"
    )

    # Build rows
    rows = []
    for i, row in sub.iterrows():
        row_prefix = ""
        if use_colors:
            hex6 = str(df.loc[i, "color"]).strip().lstrip("#").upper() if "color" in df.columns else ""
            if re.fullmatch(r"[0-9A-F]{6}", hex6) and hex6 in color_names:
                row_prefix = fr"\rowcolor{{{color_names[hex6]}!{SOFT_PCT}}} "

        lang_val = latex_escape(row[lang_col])
        vals = [lang_val] + [row[c] for c in features]
        valss = []
        for v in vals:
            try:
                vv = float(v)
            except:
                valss.append(v)
                continue
            if vv > 0:
                valss.append(v)
            else:
                valss.append('-')
        rows.append(row_prefix + " & ".join(vals) + r" \\")

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )

    latex_table = preamble + ("\n" + color_defs if color_defs else "") + "\n\n" + header + "\n".join(rows) + "\n" + footer

    # Output path
    out_path = outf if outf else os.path.splitext(os.path.basename(infis))[0] + ".tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

    print(latex_table[:1000])  # preview
    print("\nSaved LaTeX to:", out_path)


if __name__ == "__main__":
    main()
