"""
Summarise T1 preds CSV into T2 per-fold metrics and T3 rolled-up summaries.

Detects sweep type from prediction column names:
  prob_stack_k{K}  → k-sweep
  prob_{col_label} → col-sweep  (full/title/location/etc/fwd{n}/back{n})
  prob_rf          → none (structured-only baseline)

T2 metrics CSV  (results/t2_metrics/{stem}_metrics.csv):
  k-sweep    : group_id, repeat_id, fold_id, n_test, n_pos, rep, k,
               auc, ap, logloss, brier, bss
  col-sweep  : group_id, repeat_id, fold_id, n_test, n_pos, rep, col_config,
               auc, ap, logloss, brier, bss
  none       : same as k-sweep but with k=NaN and rep='none'

T3 rolled-up summary  (results/t3_rolled/{stem}_summary.csv):
  k-sweep    : rep, k, n_folds, auc_mean, auc_se, ap_mean, ap_se,
               logloss_mean, logloss_se, brier_mean, brier_se, bss_mean, bss_se
  col-sweep  : rep, col_config, sweep_type, n_cols, n_folds, auc_mean, auc_se, ...

SE = std(ddof=1) / sqrt(n_folds).

BSS per fold = 1 - brier / (prev * (1 - prev)), prev = n_pos / n_test.

CLI:
  python summarise_results.py [INPUT]          # directory or file; default: results/t1_preds/
  python summarise_results.py INPUT_PREDS [--tier {t2,t3,both}] [--rep NAME]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
METRICS = ["auc", "ap", "logloss", "brier", "bss"]

CANONICAL_COLS = [
    "title", "location", "department", "company_profile",
    "description", "requirements", "benefits",
]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_sweep_type(prob_cols: list[str]) -> str:
    """Return 'k_sweep', 'col_sweep', or 'none'."""
    if any(c == "prob_rf" for c in prob_cols):
        return "none"
    if any(re.fullmatch(r"prob_stack_k\d+", c) for c in prob_cols):
        return "k_sweep"
    return "col_sweep"


def parse_k(col: str) -> int:
    """Extract k from 'prob_stack_k{K}'."""
    return int(re.fullmatch(r"prob_stack_k(\d+)", col).group(1))


def parse_col_config(label: str) -> tuple[str, int]:
    """Return (sweep_type, n_cols) for a col-sweep config label.

    sweep_type values: 'individual', 'full', 'fwd', 'back'
    n_cols: canonical column count (1-7)
    """
    if label == "full":
        return "full", len(CANONICAL_COLS)
    if label in CANONICAL_COLS:
        return "individual", 1
    m_fwd = re.fullmatch(r"fwd(\d+)", label)
    if m_fwd:
        return "fwd", int(m_fwd.group(1))
    m_back = re.fullmatch(r"back(\d+)", label)
    if m_back:
        return "back", int(m_back.group(1))
    # Unrecognised label: treat as individual at position 0
    return "individual", 0


# ---------------------------------------------------------------------------
# Per-fold metrics
# ---------------------------------------------------------------------------

def fold_metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    n_pos = int(y.sum())
    n_test = len(y)
    prev = n_pos / n_test
    bs = float(brier_score_loss(y, prob))
    bs_ref = prev * (1.0 - prev)
    return {
        "auc":     float(roc_auc_score(y, prob)) if 0 < n_pos < n_test else float("nan"),
        "ap":      float(average_precision_score(y, prob)) if 0 < n_pos < n_test else float("nan"),
        "logloss": float(log_loss(y, prob, labels=[0, 1])),
        "brier":   bs,
        "bss":     1.0 - bs / bs_ref if bs_ref > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# T2 computation
# ---------------------------------------------------------------------------

def compute_t2(preds: pd.DataFrame, rep: str, sweep_type: str) -> pd.DataFrame:
    prob_cols = [c for c in preds.columns if c.startswith("prob_")]
    rows: list[dict] = []

    for (group_id, repeat_id, fold_id), grp in preds.groupby(
        ["group_id", "repeat_id", "fold_id"]
    ):
        y = grp["y_true"].values
        n_test = len(y)
        n_pos = int(y.sum())
        base = {
            "group_id": group_id,
            "repeat_id": repeat_id,
            "fold_id": fold_id,
            "n_test": n_test,
            "n_pos": n_pos,
            "rep": rep,
        }

        if sweep_type == "k_sweep":
            for col in prob_cols:
                k = parse_k(col)
                m = fold_metrics(y, grp[col].values)
                rows.append({**base, "k": k, **m})

        elif sweep_type == "col_sweep":
            for col in prob_cols:
                label = col[len("prob_"):]
                m = fold_metrics(y, grp[col].values)
                rows.append({**base, "col_config": label, **m})

        else:  # none
            m = fold_metrics(y, grp["prob_rf"].values)
            rows.append({**base, "k": float("nan"), **m})

    col_order: list[str]
    if sweep_type == "col_sweep":
        col_order = ["group_id", "repeat_id", "fold_id", "n_test", "n_pos",
                     "rep", "col_config"] + METRICS
    else:
        col_order = ["group_id", "repeat_id", "fold_id", "n_test", "n_pos",
                     "rep", "k"] + METRICS

    return pd.DataFrame(rows)[col_order]


# ---------------------------------------------------------------------------
# T3 roll-up
# ---------------------------------------------------------------------------

def _rollup_group(grp: pd.DataFrame) -> dict:
    n_folds = len(grp)
    out: dict = {"n_folds": n_folds}
    for m in METRICS:
        vals = grp[m].dropna().values
        mean = float(vals.mean()) if len(vals) > 0 else float("nan")
        se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        out[f"{m}_mean"] = mean
        out[f"{m}_se"] = se
    return out


def compute_t3_k_sweep(t2: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (rep, k), grp in t2.groupby(["rep", "k"], dropna=False):
        row = {"rep": rep, "k": k}
        row.update(_rollup_group(grp))
        rows.append(row)
    cols = ["rep", "k", "n_folds"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "se")]
    return pd.DataFrame(rows)[cols].sort_values(["rep", "k"]).reset_index(drop=True)


def compute_t3_col_sweep(t2: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (rep, col_config), grp in t2.groupby(["rep", "col_config"]):
        sweep_type, n_cols = parse_col_config(col_config)
        row = {"rep": rep, "col_config": col_config, "sweep_type": sweep_type, "n_cols": n_cols}
        row.update(_rollup_group(grp))
        rows.append(row)
    cols = ["rep", "col_config", "sweep_type", "n_cols", "n_folds"] + [
        f"{m}_{s}" for m in METRICS for s in ("mean", "se")
    ]
    return pd.DataFrame(rows)[cols].sort_values(["rep", "sweep_type", "n_cols"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_T1_DIR = SCRIPT_DIR / "results" / "t1_preds"


def process_one(preds_path: Path, tier: str, rep_override: str | None) -> None:
    """Summarise a single T1 preds CSV into T2 and/or T3 outputs."""
    stem = preds_path.stem
    if stem.endswith("_preds"):
        stem = stem[:-6]

    rep = rep_override
    if rep is None:
        parts = stem.split("_")
        if len(parts) >= 3:
            rep_parts = parts[2:]
            if rep_parts and rep_parts[-1] == "colsweep":
                rep_parts = rep_parts[:-1]
            rep = "_".join(rep_parts)
        else:
            rep = stem

    print(f"Input    : {preds_path.name}")
    print(f"Stem     : {stem}")
    print(f"Rep      : {rep}")

    preds = pd.read_csv(preds_path)
    prob_cols = [c for c in preds.columns if c.startswith("prob_")]
    sweep_type = detect_sweep_type(prob_cols)
    print(f"Sweep    : {sweep_type}")
    print(f"Configs  : {len(prob_cols)}")
    print(f"Rows     : {len(preds)}")
    print()

    t2_dir = SCRIPT_DIR / "results" / "t2_metrics"
    t3_dir = SCRIPT_DIR / "results" / "t3_rolled"
    t2_dir.mkdir(parents=True, exist_ok=True)
    t3_dir.mkdir(parents=True, exist_ok=True)

    t2_path = t2_dir / f"{stem}_metrics.csv"
    t3_path = t3_dir / f"{stem}_summary.csv"

    if tier in ("t2", "both"):
        print("Computing T2 per-fold metrics...", end=" ", flush=True)
        t2 = compute_t2(preds, rep, sweep_type)
        t2.to_csv(t2_path, index=False)
        print(f"done  ({len(t2)} rows)")
        print(f"  -> {t2_path}")
        print()
    else:
        if t2_path.exists():
            t2 = pd.read_csv(t2_path)
        else:
            print("T2 file not found; computing it now (not saving)...")
            t2 = compute_t2(preds, rep, sweep_type)

    if tier in ("t3", "both"):
        print("Computing T3 rolled-up summary...", end=" ", flush=True)
        if sweep_type == "col_sweep":
            t3 = compute_t3_col_sweep(t2)
        else:
            t3 = compute_t3_k_sweep(t2)
        t3.to_csv(t3_path, index=False)
        print(f"done  ({len(t3)} rows)")
        print(f"  -> {t3_path}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise T1 preds CSV(s) into T2 metrics and T3 rolled-up summaries."
    )
    parser.add_argument(
        "input_preds", type=Path, nargs="?", default=DEFAULT_T1_DIR,
        help="T1 preds CSV, or a directory of CSVs. "
             f"[{DEFAULT_T1_DIR.relative_to(SCRIPT_DIR)}]",
    )
    parser.add_argument("--tier", choices=["t2", "t3", "both"], default="both",
                        help="Which tiers to write. [both]")
    parser.add_argument("--rep", type=str, default=None,
                        help="Override the rep name (single-file mode only).")
    args = parser.parse_args()

    target = args.input_preds.resolve()
    if not target.exists():
        raise FileNotFoundError(f"Input not found: {target}")

    if target.is_dir():
        files = sorted(target.glob("*_preds.csv"))
        if not files:
            raise FileNotFoundError(f"No *_preds.csv files found in {target}")
        if args.rep:
            print("Warning: --rep is ignored in directory mode.\n")
        print(f"Batch mode: {len(files)} file(s) in {target}\n")
        for i, f in enumerate(files, 1):
            print(f"[{i}/{len(files)}] ", end="")
            process_one(f, args.tier, None)
    else:
        process_one(target, args.tier, args.rep)


if __name__ == "__main__":
    main()
