"""
Plot k-sweep T3 rolled-up summaries for the fraud dataset.

Reads T3 summary CSVs from results/t3_rolled/ and plots mean ± 1 SE error
bars for each metric vs k, one line per representation.

Files loaded: fraud_rskf_*_summary.csv that contain a 'k' column (k-sweep
summaries; col-sweep summaries have 'col_config' instead and are skipped).

CLI:
  python plot_k_sweep.py [--mode {all,tfidf}] [--results-dir PATH]
                         [--output PATH] [--metric auc ap logloss brier bss]

Modes:
  all    : tfidf (svd=384) + all embedding reps  [default]
  tfidf  : all four TF-IDF svd-dim variants
"""

import argparse
import pathlib

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent / "results" / "t3_rolled"

K_VALUES = [5, 10, 20, 50, 100]
ALL_METRICS = ["auc", "ap", "logloss", "brier", "bss"]
METRIC_LABELS = {
    "auc":     "AUC",
    "ap":      "PR-AUC",
    "logloss": "Log-loss",
    "brier":   "Brier score",
    "bss":     "Brier Skill Score",
}

# Per-mode rep allowlists (order determines plot/legend order)
REPS_ALL   = ["tfidf", "minilm", "bge-large", "openai-small", "openai-large"]
REPS_TFIDF = ["tfidf", "tfidf_svd1024", "tfidf_svd1536", "tfidf_svd3072"]

# Full ordered list used for color assignment consistency across modes
REPS_ORDERED = [
    "tfidf", "tfidf_svd1024", "tfidf_svd1536", "tfidf_svd3072",
    "minilm", "bge-large", "openai-small", "openai-large",
]

MODE_REPS = {"all": REPS_ALL, "tfidf": REPS_TFIDF}
REP_LABELS = {
    "tfidf":         "TF-IDF (svd=384)",
    "tfidf_svd1024": "TF-IDF (svd=1024)",
    "tfidf_svd1536": "TF-IDF (svd=1536)",
    "tfidf_svd3072": "TF-IDF (svd=3072)",
    "minilm":        "MiniLM",
    "bge-large":     "BGE-large",
    "openai-small":  "OpenAI-small",
    "openai-large":  "OpenAI-large",
}


def load_t3_files(results_dir: pathlib.Path) -> dict[str, pd.DataFrame]:
    """Return {rep: DataFrame} for all k-sweep summary files found."""
    data: dict[str, pd.DataFrame] = {}
    for path in sorted(results_dir.glob("fraud_rskf_*_summary.csv")):
        df = pd.read_csv(path)
        if "k" not in df.columns:
            continue
        if df.empty:
            continue
        rep = df["rep"].iloc[0]
        data[rep] = df
        print(f"  loaded {path.name}  ({len(df)} rows, rep={rep})")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot k-sweep T3 summaries for the fraud dataset."
    )
    parser.add_argument("--mode", choices=["all", "tfidf"], default="all",
                        help="all: tfidf(384) + embeddings; tfidf: all svd-dim variants. [all]")
    parser.add_argument("--results-dir", type=pathlib.Path, default=DEFAULT_RESULTS_DIR,
                        help="Directory with T3 summary CSVs. [../results/t3_rolled]")
    parser.add_argument("--output", type=pathlib.Path, default=None,
                        help="Output PNG path. [fraud_k_sweep_{mode}.png]")
    parser.add_argument("--metric", nargs="+", choices=ALL_METRICS, default=["ap", "logloss", "bss"],
                        help="Metrics to plot (one panel each). [ap logloss bss]")
    args = parser.parse_args()

    out = args.output or (SCRIPT_DIR / f"fraud_k_sweep_{args.mode}.png")

    print(f"Mode: {args.mode}")
    print(f"Loading T3 k-sweep summaries from {args.results_dir}...")
    rep_data = load_t3_files(args.results_dir)
    if not rep_data:
        raise RuntimeError(f"No k-sweep summary files found in {args.results_dir}.")
    print()

    # Filter to the mode's allowlist, preserving order
    allowlist = MODE_REPS[args.mode]
    available_reps = [r for r in allowlist if r in rep_data]

    # Assign colors in the order reps will be drawn
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    rep_colors = {rep: colors[i % len(colors)] for i, rep in enumerate(available_reps)}

    metrics = args.metric
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        mean_col = f"{metric}_mean"
        se_col = f"{metric}_se"

        for rep in available_reps:
            df = rep_data[rep].sort_values("k")
            # Use only the k values present in the data that match K_VALUES
            df_k = df[df["k"].isin(K_VALUES)]
            if df_k.empty:
                continue
            xs = df_k["k"].values
            means = df_k[mean_col].values
            ses = df_k[se_col].values
            label = REP_LABELS.get(rep, rep)
            ax.errorbar(xs, means, yerr=ses, fmt="-o",
                        color=rep_colors[rep], linewidth=1.5,
                        markersize=4, capsize=3, label=label)

        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("k (GMM components)")
        ax.set_xticks(K_VALUES)

    handles = [
        mlines.Line2D([], [], color=rep_colors[rep], marker="o",
                      label=REP_LABELS.get(rep, rep))
        for rep in available_reps
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.08), frameon=False)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    # ------------------------------------------------------------------
    # Numeric tables (one per plotted metric)
    # ------------------------------------------------------------------
    print()
    for metric in metrics:
        mean_col = f"{metric}_mean"
        se_col   = f"{metric}_se"
        cols = []
        for k in K_VALUES:
            cols += [f"k={k} mean", f"k={k} se"]
        rows = []
        for rep in available_reps:
            df_k = rep_data[rep][rep_data[rep]["k"].isin(K_VALUES)].sort_values("k")
            row = []
            for k in K_VALUES:
                match = df_k[df_k["k"] == k]
                if not match.empty:
                    row += [round(float(match[mean_col].iloc[0]), 4),
                            round(float(match[se_col].iloc[0]),   4)]
                else:
                    row += [float("nan"), float("nan")]
            rows.append(row)
        labels = [REP_LABELS.get(r, r) for r in available_reps]
        tbl = pd.DataFrame(rows, index=labels, columns=cols)
        print(f"=== {METRIC_LABELS[metric]} ===")
        print(tbl.to_string())
        print()


if __name__ == "__main__":
    main()
