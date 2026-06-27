"""
Paired statistical comparison of the Original (A) vs Enhanced (B)
recommender for the COMP4135 group project.

Usage
-----
    python evaluation/analyze_results.py evaluation/results.csv \
        --out evaluation/results_summary.csv \
        --figures-dir report/figures

For every Likert question the script prints
    - mean and std of each condition
    - paired-sample t-test (t, p)
    - Wilcoxon signed-rank test (W, p) — robust alternative
    - Cohen's d_z effect size
and optionally writes two PNGs into ``figures-dir``:
    - fig_mean_bars.png      grouped bar chart with error bars
    - fig_paired_box.png     per-question boxplot for each condition
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

QUESTIONS = [f"Q{i}" for i in range(1, 9)]
QUESTION_LABELS = {
    "Q1": "Recommendation match",
    "Q2": "Diversity",
    "Q3": "Novelty",
    "Q4": "Transparency",
    "Q5": "Control",
    "Q6": "Ease-of-use",
    "Q7": "Satisfaction",
    "Q8": "Trust",
}


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"participant_id", "condition", *QUESTIONS}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Missing columns in {csv_path}: {sorted(missing)}")
    df[list(QUESTIONS)] = df[list(QUESTIONS)].apply(pd.to_numeric, errors="coerce")
    return df


def pair_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per participant with A_* and B_* columns."""
    a = df[df["condition"].str.upper() == "A"].set_index("participant_id")
    b = df[df["condition"].str.upper() == "B"].set_index("participant_id")
    common = a.index.intersection(b.index)
    if len(common) == 0:
        sys.exit("No participant has both A and B rows in the CSV.")
    a = a.loc[common][QUESTIONS].add_prefix("A_")
    b = b.loc[common][QUESTIONS].add_prefix("B_")
    return pd.concat([a, b], axis=1)


def cohens_dz(diff: pd.Series) -> float:
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def analyse(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(paired)
    for q in QUESTIONS:
        a = paired[f"A_{q}"].dropna()
        b = paired[f"B_{q}"].dropna()
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        diff = b - a
        if len(diff) < 2 or diff.std(ddof=1) == 0:
            t_stat = p_t = w_stat = p_w = float("nan")
        else:
            t_stat, p_t = stats.ttest_rel(b, a)
            try:
                w_stat, p_w = stats.wilcoxon(b, a, zero_method="wilcox")
            except ValueError:
                w_stat = p_w = float("nan")
        rows.append({
            "question": q,
            "label": QUESTION_LABELS[q],
            "n": len(diff),
            "mean_A": round(a.mean(), 2),
            "sd_A": round(a.std(ddof=1), 2),
            "mean_B": round(b.mean(), 2),
            "sd_B": round(b.std(ddof=1), 2),
            "mean_diff(B-A)": round(diff.mean(), 2),
            "t": round(t_stat, 3) if pd.notna(t_stat) else None,
            "p_t": round(p_t, 4) if pd.notna(p_t) else None,
            "W": round(w_stat, 3) if pd.notna(w_stat) else None,
            "p_wilcoxon": round(p_w, 4) if pd.notna(p_w) else None,
            "cohen_dz": round(cohens_dz(diff), 3),
        })
    return pd.DataFrame(rows)


def make_plots(paired: pd.DataFrame, out_dir: Path) -> None:
    """Generate paired mean-bar and box plots for the report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [QUESTION_LABELS[q] for q in QUESTIONS]
    mean_a = [paired[f"A_{q}"].mean() for q in QUESTIONS]
    mean_b = [paired[f"B_{q}"].mean() for q in QUESTIONS]
    sem_a = [paired[f"A_{q}"].std(ddof=1) / np.sqrt(len(paired)) for q in QUESTIONS]
    sem_b = [paired[f"B_{q}"].std(ddof=1) / np.sqrt(len(paired)) for q in QUESTIONS]

    x = np.arange(len(QUESTIONS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w / 2, mean_a, w, yerr=sem_a, label="A — Original",
           color="#9aa4ad", capsize=3)
    ax.bar(x + w / 2, mean_b, w, yerr=sem_b, label="B — Enhanced",
           color="#3b82f6", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean Likert score (1–5)")
    ax.set_ylim(0, 5.2)
    ax.set_title(f"Paired means per question (n={len(paired)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_mean_bars.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True)
    for ax, q, label in zip(axes.flat, QUESTIONS, labels):
        bp_kwargs = dict(widths=0.6, patch_artist=True,
                         boxprops=dict(facecolor="#e5e7eb"))
        try:
            ax.boxplot([paired[f"A_{q}"], paired[f"B_{q}"]],
                       tick_labels=["A", "B"], **bp_kwargs)
        except TypeError:
            ax.boxplot([paired[f"A_{q}"], paired[f"B_{q}"]],
                       labels=["A", "B"], **bp_kwargs)
        ax.set_title(label, fontsize=9)
        ax.set_ylim(0, 5.5)
    fig.suptitle(f"Distribution per condition (n={len(paired)})")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_paired_box.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path, help="Path to results.csv")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional path to save the per-question table.")
    ap.add_argument("--figures-dir", type=Path, default=None,
                    help="If set, write PNG plots into this directory.")
    args = ap.parse_args()

    df = load(args.csv)
    paired = pair_conditions(df)
    print(f"Loaded {len(paired)} paired participants from {args.csv}")
    print()

    result = analyse(paired)
    print(result.to_string(index=False))
    print()
    print("Positive `mean_diff(B-A)` means Enhanced beats Original.")
    print("Reject H0 (no difference) when p < 0.05.")
    print("Cohen's d_z: 0.2 small, 0.5 medium, 0.8 large.")

    if args.out:
        result.to_csv(args.out, index=False)
        print(f"\nSaved table to {args.out}")
    if args.figures_dir:
        make_plots(paired, args.figures_dir)
        print(f"Saved figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
