"""
Plot column-sweep T3 rolled-up summaries for the fraud dataset.

Reads T3 col-sweep summary CSVs (fraud_rskf_*_colsweep_summary.csv) from
results/t3_rolled/ and produces one panel per metric.

Default mode (--classic, on by default):
  Two sweep lines only; x-axis shows numerics 1-7; backward sweep runs
  left→right (n columns from the benefits end).

Non-classic mode (--no-classic):
  Adds individual-column scatter points; x-axis shows column names;
  backward sweep runs right→left.

Panel layout:  1-3 metrics → 1×N;  4 → 2×2;  5 → 2×3.

CLI args and defaults:
  --results-dir PATH    T3 summary CSV directory  [../results/t3_rolled]
  --output PATH         Output PNG                [fraud_col_sweep.png]
  --rep NAME            Rep to plot               [first available]
  --metric M [M ...]    Metrics (one panel each)  [ap logloss bss]
                          choices: auc ap logloss brier bss
  --no-error-bars       Disable ±SE error bars     [error bars on by default]
  --se-multiplier F     Multiplier applied to SE   [1.0]
  --no-classic          Switch to non-classic mode [classic on by default]
"""

import argparse
import pathlib

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent / "results" / "t3_rolled"
DEFAULT_OUTPUT = SCRIPT_DIR / "fraud_col_sweep.png"

CANONICAL_COLS = [
    "title", "location", "department", "company_profile",
    "description", "requirements", "benefits",
]
N = len(CANONICAL_COLS)

# Config label at each x position (1-indexed) for each sweep
FWD_AT_X = ["title", "fwd2", "fwd3", "fwd4", "fwd5", "fwd6", "full"]
BACK_AT_X = ["back7", "back6", "back5", "back4", "back3", "back2", "benefits"]
BACK_AT_X_CLASSIC = ["benefits", "back2", "back3", "back4", "back5", "back6", "back7"]

ALL_METRICS = ["auc", "ap", "logloss", "brier", "bss"]
DEFAULT_METRICS = ["ap", "logloss", "bss"]
METRIC_LABELS = {
    "auc":     "AUC",
    "ap":      "PR-AUC",
    "logloss": "Log-loss",
    "brier":   "Brier score",
    "bss":     "Brier Skill Score",
}


def get_layout(n: int) -> tuple[int, int]:
    if n <= 3:
        return 1, n
    if n == 4:
        return 2, 2
    return 2, 3


def load_t3_files(results_dir: pathlib.Path) -> dict[str, pd.DataFrame]:
    """Return {rep: DataFrame} for col-sweep summary files."""
    data: dict[str, pd.DataFrame] = {}
    for path in sorted(results_dir.glob("fraud_rskf_*_colsweep_summary.csv")):
        df = pd.read_csv(path)
        if "col_config" not in df.columns:
            continue
        if df.empty:
            continue
        rep = df["rep"].iloc[0]
        data[rep] = df
        print(f"  loaded {path.name}  ({len(df)} rows, rep={rep})")
    return data


def draw_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    se_multiplier: float = 1.0,
    classic: bool = False,
) -> None:
    """Draw one metric panel from T3 col-sweep DataFrame."""
    mean_col = f"{metric}_mean"
    se_col = f"{metric}_se"

    idx = df.set_index("col_config")
    xs = np.arange(1, N + 1)

    def _mean(label: str) -> float:
        return float(idx.loc[label, mean_col]) if label in idx.index else float("nan")

    def _se(label: str) -> float:
        return float(idx.loc[label, se_col]) * se_multiplier if label in idx.index else 0.0

    def _err(labels):
        vals = [_se(lbl) for lbl in labels]
        return vals if se_multiplier > 0 else None

    if not classic:
        ind_means = [_mean(col) for col in CANONICAL_COLS]
        ind_err = _err(CANONICAL_COLS)
        ax.errorbar(xs, ind_means, yerr=ind_err, fmt="o", color="steelblue",
                    alpha=0.5, markersize=4, capsize=3 if ind_err else 0,
                    zorder=3, label="Single column")

    fwd_means = [_mean(lbl) for lbl in FWD_AT_X]
    fwd_err = _err(FWD_AT_X)
    ax.errorbar(xs, fwd_means, yerr=fwd_err, fmt="-", color="steelblue",
                linewidth=1.8, capsize=3 if fwd_err else 0,
                zorder=2, label="Forward sweep")

    back_labels = BACK_AT_X_CLASSIC if classic else BACK_AT_X
    back_means = [_mean(lbl) for lbl in back_labels]
    back_err = _err(back_labels)
    ax.errorbar(xs, back_means, yerr=back_err, fmt="-", color="darkorange",
                linewidth=1.8, capsize=3 if back_err else 0,
                zorder=2, label="Backward sweep")

    if metric in ("logloss", "brier"):
        ax.invert_yaxis()

    ax.set_xticks(xs)
    if classic:
        ax.set_xticklabels(xs)
    else:
        ax.set_xticklabels(CANONICAL_COLS, rotation=40, ha="right", fontsize=8)
    ax.set_title(METRIC_LABELS[metric])
    ax.set_xlabel("# of text columns" if classic else "text column")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot col-sweep T3 summaries for the fraud dataset."
    )
    parser.add_argument("--results-dir", type=pathlib.Path, default=DEFAULT_RESULTS_DIR,
                        help="Directory with T3 summary CSVs. [../results/t3_rolled]")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT,
                        help="Output PNG path. [fraud_col_sweep.png]")
    parser.add_argument("--rep", type=str, default=None,
                        help="Plot only this rep (default: first available).")
    parser.add_argument("--metric", nargs="+", choices=ALL_METRICS, default=DEFAULT_METRICS,
                        help=f"Metrics to plot (one panel each). [{', '.join(DEFAULT_METRICS)}]")
    parser.add_argument("--no-error-bars", action="store_false", dest="error_bars",
                        help="Disable ±SE error bars (on by default).")
    parser.add_argument("--se-multiplier", type=float, default=1.0,
                        help="SE multiplier for error bars. [1.0]")
    parser.add_argument("--no-classic", action="store_false", dest="classic",
                        help="Non-classic mode: individual-column scatter, column-name x-axis, "
                             "backward sweep right→left.")
    args = parser.parse_args()

    print(f"Loading T3 col-sweep summaries from {args.results_dir}...")
    rep_data = load_t3_files(args.results_dir)
    if not rep_data:
        raise RuntimeError(f"No col-sweep summary files found in {args.results_dir}.")
    print()

    if args.rep:
        if args.rep not in rep_data:
            raise ValueError(
                f"Rep '{args.rep}' not found. Available: {list(rep_data)}"
            )
        rep = args.rep
    else:
        rep = next(iter(rep_data))
        print(f"Using rep: {rep}")

    df = rep_data[rep]

    for metric in ALL_METRICS:
        col = f"{metric}_mean"
        if col in df.columns:
            full_mean = df.loc[df["col_config"] == "full", col]
            back7_mean = df.loc[df["col_config"] == "back7", col]
            full_str = f"{float(full_mean.iloc[0]):.4f}" if not full_mean.empty else "N/A"
            back7_str = f"{float(back7_mean.iloc[0]):.4f}" if not back7_mean.empty else "N/A"
            print(f"  {METRIC_LABELS[metric]:18s}  fwd(full)={full_str}  back(back7)={back7_str}")
    print()

    metrics = args.metric
    nrows, ncols = get_layout(len(metrics))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))

    if nrows == 1 and ncols == 1:
        axes_flat = [axes]
    elif nrows == 1:
        axes_flat = list(axes)
    else:
        axes_flat = [ax for row in axes for ax in row]

    se_mult = args.se_multiplier if args.error_bars else 0.0
    for ax, metric in zip(axes_flat, metrics):
        draw_panel(ax, df, metric, se_multiplier=se_mult, classic=args.classic)

    for ax in axes_flat[len(metrics):]:
        ax.set_visible(False)

    handles = []
    if not args.classic:
        handles.append(mlines.Line2D([], [], color="steelblue", marker="o",
                                     linestyle="none", alpha=0.5, markersize=6,
                                     label="Single column"))
    handles += [
        mlines.Line2D([], [], color="steelblue", linewidth=1.8, label="Forward sweep"),
        mlines.Line2D([], [], color="darkorange", linewidth=1.8, label="Backward sweep"),
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), bbox_to_anchor=(0.5, -0.04), frameon=False)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")

    # ------------------------------------------------------------------
    # Numeric tables (one per plotted metric)
    # ------------------------------------------------------------------
    back_labels = BACK_AT_X_CLASSIC if args.classic else BACK_AT_X
    idx = df.set_index("col_config")
    print()
    for metric in metrics:
        mean_col = f"{metric}_mean"
        se_col   = f"{metric}_se"
        rows = []
        for x, fwd_lbl, back_lbl in zip(range(1, N + 1), FWD_AT_X, back_labels):
            def _val(lbl, col):
                return round(float(idx.loc[lbl, col]), 4) if lbl in idx.index else float("nan")
            rows.append({
                "x":         x,
                "fwd_config":  fwd_lbl,
                "fwd_mean":    _val(fwd_lbl,  mean_col),
                "fwd_se":      _val(fwd_lbl,  se_col),
                "back_config": back_lbl,
                "back_mean":   _val(back_lbl, mean_col),
                "back_se":     _val(back_lbl, se_col),
            })
        tbl = pd.DataFrame(rows).set_index("x")
        print(f"=== {METRIC_LABELS[metric]} ===")
        print(tbl.to_string())
        print()


if __name__ == "__main__":
    main()
