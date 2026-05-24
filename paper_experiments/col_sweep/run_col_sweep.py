"""
Fraud CV column sweep: single k, no stacking, one embedding file per config.

Replaces the k-sweep dimension with a column-sweep dimension. Each config
corresponds to one embedding file (produced by generate_embeddings.py
with --sweeps), fed through GMM(k) then a single RF alongside structured
features. No stacking meta-learner.

Column configs (19 total, deduplicating symlinked overlaps):
  full      : all 7 columns in canonical order
  {col}     : each of 7 columns individually
  fwd{n}    : forward sweep n=2..6  (title → ... → COLS[n-1])
  back{n}   : backward sweep n=2..7 (benefits → ... → COLS[7-n])

For tfidf, column texts are concatenated in feed order and TF-IDF+SVD is fit
per fold (no pre-computed cache). Forward and backward sweep text orderings
differ for neural reps but are identical for TF-IDF (bag-of-words).

Output in results/t1_preds/:
  fraud_{cv}_{rep}_colsweep_preds.csv
    columns: orig_idx, filtered_idx, group_id, repeat_id, fold_id,
             industry, y_true, prob_{label} for each config
  fraud_{cv}_{rep}_colsweep_metrics.csv
    columns: group_id, repeat_id, fold_id, n_test, n_pos, n_neg,
             auc_{label}, logloss_{label}, brier_{label} for each config
  fraud_{cv}_{rep}_colsweep_config.json

CV methods:
  logo  : LeaveOneGroupOut by industry (drops rows with missing industry)
  rskf  : RepeatedStratifiedKFold stratified on class label (uses all rows)

  LOGO:  group_id = industry name (or _OTHER_); repeat_id = 0 always
  RSKF:  group_id = r{repeat}_f{fold}; industry column preserved in preds

CLI args:
  --rep {minilm,bge-large,openai-small,openai-large,tfidf}           [minilm]
  --cv-method {logo,rskf}                                            [rskf]
  --n-repeats INT     RSKF repeats                                   [10]
  --n-folds INT       RSKF folds per repeat                          [10]
  --k INT             GMM n_components (single value)                [10]
  --cov STR           GMM covariance_type                            [spherical]
  --svd-dim INT       TruncatedSVD dim (tfidf only)                  [384]
  --n-trees-text INT  RF trees per config pipeline                   [700]
  --min-group-size INT  LOGO: merge industries below this size       [10]
  --n-jobs INT        joblib outer parallelism                       [-1]
  --checkpoint-every INT                                             [5]
  --seed INT                                                         [42]
  --output PATH       preds CSV (metrics + config derived from name)
  --data-dir PATH     directory with fake_job_postings.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from tabullm import GMMFeatureExtractor, load_fraud

# ---------------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_embeddings import TEXT_COLUMNS, get_cache_path, get_sweep_cache_path  # noqa: E402

FRAUD_TEXT_COLS = TEXT_COLUMNS["fraud"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OTHER_GROUP_LABEL = "_OTHER_"
LOW_CARD_COLS = ["employment_type", "required_experience", "required_education"]
MED_CARD_COLS = ["industry", "function"]
DEFAULT_K = 10
DEFAULT_COV = "spherical"
DEFAULT_SVD_DIM = 384
DEFAULT_N_TREES_TEXT = 700
TFIDF_KWARGS = dict(ngram_range=(1, 2), min_df=3, max_features=20000)
NEURAL_REPS = ["minilm", "bge-large", "openai-small", "openai-large"]
SUPPORTED_REPS = NEURAL_REPS + ["tfidf"]


# ---------------------------------------------------------------------------
# Column configs
# ---------------------------------------------------------------------------

@dataclass
class ColConfig:
    label: str        # used in output column names
    cols: list[str]   # columns in feed order (canonical for fwd, reversed for back)


def build_col_configs() -> list[ColConfig]:
    """Return the 19 unique column configurations for the sweep."""
    COLS = FRAUD_TEXT_COLS
    N = len(COLS)
    configs: list[ColConfig] = []

    configs.append(ColConfig("full", list(COLS)))

    for col in COLS:
        configs.append(ColConfig(col, [col]))

    # Forward n=2..N-1 (skip n=1=title and n=N=full, already added above)
    for n in range(2, N):
        configs.append(ColConfig(f"fwd{n}", list(COLS[:n])))

    # Backward n=2..N (skip n=1=benefits, already added; back-N ≠ full)
    for n in range(2, N + 1):
        configs.append(ColConfig(f"back{n}", list(reversed(COLS[N - n:]))))

    return configs


def get_config_path(rep: str, cfg: ColConfig) -> Path:
    if cfg.label == "full":
        return get_cache_path("fraud", rep)
    if len(cfg.cols) == 1:
        return get_cache_path("fraud", rep, cfg.cols)
    if cfg.label.startswith("fwd"):
        return get_sweep_cache_path("fraud", rep, "fwd", cfg.cols)
    return get_sweep_cache_path("fraud", rep, "back", cfg.cols)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def concat_texts(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return df[cols].fillna("").astype(str).apply(" ".join, axis=1).tolist()


def fit_tfidf_svd(
    texts_train: list[str],
    texts_test: list[str],
    svd_dim: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    vec = TfidfVectorizer(**TFIDF_KWARGS)
    tfidf_tr = vec.fit_transform(texts_train)
    tfidf_te = vec.transform(texts_test)
    n_components = min(svd_dim, tfidf_tr.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    return svd.fit_transform(tfidf_tr), svd.transform(tfidf_te)


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def _structured_transformer(binary_cols: list[str], seed: int) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("binary", "passthrough", binary_cols),
            (
                "low_card",
                Pipeline([
                    ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                    ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
                ]),
                LOW_CARD_COLS,
            ),
            (
                "med_card",
                Pipeline([
                    ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                    ("target", TargetEncoder(
                        cv=5, smooth="auto", target_type="binary", random_state=seed,
                    )),
                ]),
                MED_CARD_COLS,
            ),
        ],
        remainder="drop",
    )


def build_config_pipeline(
    emb_col_names: list[str],
    binary_cols: list[str],
    k: int,
    cov_type: str,
    n_trees: int,
    seed: int,
) -> Pipeline:
    """GMM on embedding cols + structured features -> single RF."""
    return Pipeline([
        (
            "features",
            ColumnTransformer(
                transformers=[
                    (
                        "gmm",
                        GMMFeatureExtractor(
                            n_components=k,
                            covariance_type=cov_type,
                            random_state=seed,
                        ),
                        emb_col_names,
                    ),
                    ("binary", "passthrough", binary_cols),
                    (
                        "low_card",
                        Pipeline([
                            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                            ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
                        ]),
                        LOW_CARD_COLS,
                    ),
                    (
                        "med_card",
                        Pipeline([
                            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                            ("target", TargetEncoder(
                                cv=5, smooth="auto", target_type="binary", random_state=seed,
                            )),
                        ]),
                        MED_CARD_COLS,
                    ),
                ],
                remainder="drop",
            ),
        ),
        ("rf", RandomForestClassifier(n_estimators=n_trees, n_jobs=1, random_state=seed)),
    ])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())
    auc = float(roc_auc_score(y_true, prob)) if (n_pos > 0 and n_neg > 0) else float("nan")
    return {
        "auc": auc,
        "logloss": float(log_loss(y_true, prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, prob)),
    }


# ---------------------------------------------------------------------------
# Per-fold worker
# ---------------------------------------------------------------------------

def run_one_fold(
    fold_spec: dict,
    X: pd.DataFrame,
    y: pd.Series,
    col_embs_full: dict[str, np.ndarray] | None,
    metadata: dict,
    rep: str,
    col_configs: list[ColConfig],
    k: int,
    cov: str,
    svd_dim: int,
    n_trees_text: int,
    seed: int,
    surviving: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    # Suppress GaussianMixture convergence warnings in each worker process.
    warnings.filterwarnings("ignore", message=".*did not converge.*",
                            module="sklearn.mixture._base")

    group_id = fold_spec["group_id"]
    repeat_id = fold_spec["repeat_id"]
    fold_id = fold_spec["fold_id"]
    train_idx = fold_spec["train_idx"]
    test_idx = fold_spec["test_idx"]

    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    binary_cols: list[str] = metadata["binary_columns"]

    metrics_row: dict = {
        "group_id": group_id,
        "repeat_id": repeat_id,
        "fold_id": fold_id,
        "n_test": len(y_te),
        "n_pos": int(y_te.sum()),
        "n_neg": int((y_te == 0).sum()),
    }
    preds: dict = {
        "orig_idx": surviving[test_idx],
        "filtered_idx": test_idx,
        "group_id": group_id,
        "repeat_id": repeat_id,
        "fold_id": fold_id,
        "industry": X_te["industry"].values,
        "y_true": y_te.values,
    }

    for cfg in col_configs:
        if rep in NEURAL_REPS:
            emb_tr = col_embs_full[cfg.label][train_idx]
            emb_te = col_embs_full[cfg.label][test_idx]
        else:  # tfidf
            emb_tr, emb_te = fit_tfidf_svd(
                concat_texts(X_tr, cfg.cols),
                concat_texts(X_te, cfg.cols),
                svd_dim, seed,
            )

        dim = emb_tr.shape[1]
        emb_col_names = [f"emb_{i}" for i in range(dim)]

        X_aug_tr = pd.concat(
            [X_tr.reset_index(drop=True), pd.DataFrame(emb_tr, columns=emb_col_names)], axis=1
        )
        X_aug_te = pd.concat(
            [X_te.reset_index(drop=True), pd.DataFrame(emb_te, columns=emb_col_names)], axis=1
        )

        pipe = build_config_pipeline(emb_col_names, binary_cols, k, cov, n_trees_text, seed)
        pipe.fit(X_aug_tr, y_tr)
        prob = pipe.predict_proba(X_aug_te)[:, 1]

        preds[f"prob_{cfg.label}"] = prob
        for metric, val in compute_metrics(y_te.values, prob).items():
            metrics_row[f"{metric}_{cfg.label}"] = val

    return metrics_row, pd.DataFrame(preds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fraud CV column sweep: single k, no stacking."
    )
    parser.add_argument("--rep", choices=SUPPORTED_REPS, default="minilm")
    parser.add_argument("--cv-method", choices=["logo", "rskf"], default="rskf")
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="GMM n_components (single value). [10]")
    parser.add_argument("--cov", type=str, default=DEFAULT_COV)
    parser.add_argument("--svd-dim", type=int, default=DEFAULT_SVD_DIM)
    parser.add_argument("--n-trees-text", type=int, default=DEFAULT_N_TREES_TEXT,
                        help="RF trees per config pipeline. [700]")
    parser.add_argument("--min-group-size", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    col_configs = build_col_configs()

    default_stem = f"fraud_{args.cv_method}_{args.rep}_colsweep"
    default_dir = Path(__file__).resolve().parent.parent / "results" / "t1_preds"
    if args.output:
        out_preds = args.output
        stem = out_preds.stem
        if stem.endswith("_preds"):
            stem = stem[:-6]
    else:
        stem = default_stem
        out_preds = default_dir / f"{stem}_preds.csv"
    out_metrics = out_preds.parent / f"{stem}_metrics.csv"
    out_config = out_preds.parent / f"{stem}_config.json"
    out_preds.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fraud CV col-sweep: rep={args.rep}  cv={args.cv_method}  "
          f"k={args.k}  seed={args.seed}  n_jobs={args.n_jobs}")
    print(f"  {len(col_configs)} col configs  cov={args.cov}  "
          f"n_trees_text={args.n_trees_text}")
    if args.rep == "tfidf":
        print(f"  tfidf: svd_dim={args.svd_dim}  {TFIDF_KWARGS}")
    if args.cv_method == "rskf":
        print(f"  rskf: {args.n_repeats} repeats × {args.n_folds} folds "
              f"= {args.n_repeats * args.n_folds} total folds")
    print()

    config = {
        "rep": args.rep,
        "cv_method": args.cv_method,
        "n_repeats": args.n_repeats,
        "n_folds": args.n_folds,
        "k": args.k,
        "cov": args.cov,
        "svd_dim": args.svd_dim,
        "n_trees_text": args.n_trees_text,
        "min_group_size": args.min_group_size,
        "n_jobs": args.n_jobs,
        "checkpoint_every": args.checkpoint_every,
        "seed": args.seed,
        "col_configs": [{"label": c.label, "cols": c.cols} for c in col_configs],
        "output_preds": str(out_preds),
        "output_metrics": str(out_metrics),
        "data_dir": str(args.data_dir) if args.data_dir else None,
    }
    with open(out_config, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {out_config}")
    print()

    X, y, metadata = load_fraud(data_dir=args.data_dir)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if args.cv_method == "logo":
        mask = X["industry"].notna()
        surviving = np.where(mask.values)[0]
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)

        sizes = X["industry"].value_counts()
        small = sizes[sizes < args.min_group_size].index.tolist()
        industry_group = X["industry"].where(~X["industry"].isin(small), other=OTHER_GROUP_LABEL)
        X = X.copy()
        X["industry_group"] = industry_group
        groups = X["industry_group"].values
        other_rows = int(X["industry"].isin(small).sum())
        n_groups = len(np.unique(groups))

        print(f"  n={len(X)} (dropped {(~mask).sum()} missing-industry rows) | "
              f"fraud_rate={y.mean():.4f}")
        print(f"  {n_groups} folds: {n_groups - 1} named + "
              f"_OTHER_ ({other_rows} rows, min_group_size={args.min_group_size})")
        print()

        logo = LeaveOneGroupOut()
        fold_specs = [
            {"group_id": groups[test_idx][0], "repeat_id": 0, "fold_id": i,
             "train_idx": train_idx, "test_idx": test_idx}
            for i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups))
        ]
    else:
        surviving = np.arange(len(X))
        print(f"  n={len(X)}  fraud_rate={y.mean():.4f}")
        print(f"  rskf: {args.n_repeats} × {args.n_folds} = "
              f"{args.n_repeats * args.n_folds} folds")
        print()

        rskf = RepeatedStratifiedKFold(
            n_splits=args.n_folds, n_repeats=args.n_repeats, random_state=args.seed
        )
        fold_specs = [
            {"group_id": f"r{i // args.n_folds}_f{i % args.n_folds}",
             "repeat_id": i // args.n_folds, "fold_id": i % args.n_folds,
             "train_idx": train_idx, "test_idx": test_idx}
            for i, (train_idx, test_idx) in enumerate(rskf.split(X, y))
        ]

    # Load embedding caches (neural reps only)
    col_embs_full: dict[str, np.ndarray] | None = None
    if args.rep in NEURAL_REPS:
        print(f"Loading {args.rep} embedding caches ({len(col_configs)} configs)...")
        col_embs_full = {}
        for cfg in col_configs:
            path = get_config_path(args.rep, cfg)
            arr = np.load(path)[surviving]
            col_embs_full[cfg.label] = arr
            print(f"  {cfg.label:20s}: {path.name}  shape={arr.shape}")
        print()
    elif args.rep == "tfidf":
        print("TF-IDF: fit per fold (no cache).")
        print()

    # Progress header
    print(f"Running {len(fold_specs)} folds (checkpoint every {args.checkpoint_every})...")
    print(f"  {'group_id':<30} {'n_te':>5} {'n+':>4}  {'auc_full':>9}")
    print("  " + "-" * 54)

    all_metrics: list[dict] = []
    all_preds: list[pd.DataFrame] = []

    gen = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
        delayed(run_one_fold)(
            fold_spec, X, y, col_embs_full, metadata,
            args.rep, col_configs, args.k, args.cov, args.svd_dim,
            args.n_trees_text, args.seed, surviving,
        )
        for fold_spec in fold_specs
    )

    for i, (metrics_row, preds_df) in enumerate(gen, 1):
        all_metrics.append(metrics_row)
        all_preds.append(preds_df)

        auc_key = "auc_full"
        auc_val = metrics_row.get(auc_key, float("nan"))
        auc_str = f"{auc_val:.4f}" if not np.isnan(auc_val) else "   NaN"
        print(
            f"  {metrics_row['group_id']:<30} {metrics_row['n_test']:>5} "
            f"{metrics_row['n_pos']:>4}  {auc_str:>9}  [{i}/{len(fold_specs)}]"
        )

        if i % args.checkpoint_every == 0:
            (
                pd.concat(all_preds, ignore_index=True)
                .sort_values(["repeat_id", "fold_id", "orig_idx"])
                .reset_index(drop=True)
                .to_csv(out_preds, index=False)
            )
            (
                pd.DataFrame(all_metrics)
                .sort_values(["repeat_id", "fold_id"])
                .reset_index(drop=True)
                .to_csv(out_metrics, index=False)
            )
            print(f"  [checkpoint] {i}/{len(fold_specs)} → {out_preds.name}")

    print()

    df_preds = (
        pd.concat(all_preds, ignore_index=True)
        .sort_values(["repeat_id", "fold_id", "orig_idx"])
        .reset_index(drop=True)
    )
    df_preds.to_csv(out_preds, index=False)

    df_metrics = (
        pd.DataFrame(all_metrics)
        .sort_values(["repeat_id", "fold_id"])
        .reset_index(drop=True)
    )
    df_metrics.to_csv(out_metrics, index=False)

    print(f"Preds   saved to {out_preds}")
    print(f"Metrics saved to {out_metrics}")
    print()

    # Aggregate summary
    weights = (df_metrics["n_pos"] * df_metrics["n_neg"]).astype(float)
    w_sum = weights.sum()
    n_valid = int((weights > 0).sum())

    labels = [c.label for c in col_configs]
    print(f"Aggregate (weight=n_pos*n_neg, {n_valid} non-degenerate folds):")
    print(f"  {'config':<22}  {'wtd AUC':>8}  {'wtd logloss':>11}  {'wtd brier':>10}")
    print("  " + "-" * 58)
    for lbl in labels:
        auc_col = f"auc_{lbl}"
        w_valid = weights.where(df_metrics[auc_col].notna(), 0.0)
        wtd_auc = (
            float((df_metrics[auc_col].fillna(0) * w_valid).sum() / w_valid.sum())
            if w_valid.sum() > 0 else float("nan")
        )
        wtd_ll = float((df_metrics[f"logloss_{lbl}"] * weights).sum() / w_sum) if w_sum > 0 else float("nan")
        wtd_br = float((df_metrics[f"brier_{lbl}"] * weights).sum() / w_sum) if w_sum > 0 else float("nan")
        print(f"  {lbl:<22}  {wtd_auc:>8.4f}  {wtd_ll:>11.4f}  {wtd_br:>10.4f}")


if __name__ == "__main__":
    main()
