"""
Fraud CV k-sweep: single-rep, multi-K, LOGO or RSKF.

Extends compute_fraud_logo_stack_k_sweep.py with:
  - 'none' representation: structured features only, single RF, no stacking
  - RepeatedStratifiedKFold (--cv-method rskf) alongside LOGO
  - Configurable RF tree counts (--n-trees-text, --n-trees-none)
  - Per-fold metrics CSV output alongside preds

Supported representations:
  minilm, bge-large, openai-small, openai-large, tfidf  — as before
  none — structured features only; stacking and k-sweep are skipped;
          single RF uses --n-trees-none trees

CV methods:
  logo  : LeaveOneGroupOut by industry (drops rows with missing industry)
  rskf  : RepeatedStratifiedKFold stratified on class label (uses all rows)

RF tree counts:
  --n-trees-text INT  trees per stacking base estimator              [100]
  --n-trees-none INT  trees for the no-rep RF                        [700]
  Rationale: 7 text columns × 100 = 700 trees — matched compute budget.

Output files (default paths in results/t1_preds/):
  fraud_{cv}_{rep}_preds.csv
    columns: orig_idx, filtered_idx, group_id, repeat_id, fold_id,
             industry, y_true,
             prob_stack_k{K}... for text reps  |  prob_rf for none
  fraud_{cv}_{rep}_metrics.csv
    columns: group_id, repeat_id, fold_id, n_test, n_pos, n_neg,
             auc_k{K}, logloss_k{K}, brier_k{K}...  |  auc_rf, ... for none
  fraud_{cv}_{rep}_config.json — CLI args for the run

  LOGO:  group_id = industry name (or _OTHER_); repeat_id = 0 always
  RSKF:  group_id = r{repeat}_f{fold}; industry column preserved in preds

Aggregate summary printed to stdout only.

Reproducibility: fold construction uses --seed alone, before any rep-specific
code, so splits are identical across reps for the same seed + CV settings.

CLI args:
  --rep {minilm,bge-large,openai-small,openai-large,tfidf,none}     [minilm]
  --cv-method {logo,rskf}                                           [rskf]
  --n-repeats INT     RSKF repeats                                  [10]
  --n-folds INT       RSKF folds per repeat                         [10]
  --k-values STR      comma-separated GMM n_components              [5,10,20,50,100]
                      (ignored when --rep none)
  --cov STR           GMM covariance_type (ignored when --rep none) [spherical]
  --svd-dim INT       TruncatedSVD dim (tfidf only)                 [384]
  --n-trees-text INT  RF trees per stacking base estimator          [100]
  --n-trees-none INT  RF trees for no-rep pipeline                  [700]
  --min-group-size INT  LOGO: merge industries below this size      [10]
  --n-jobs INT        joblib outer parallelism                      [-1]
  --checkpoint-every INT                                            [5]
  --seed INT                                                        [42]
  --output PATH       preds CSV (metrics + config derived from name)
  --data-dir PATH     directory with fake_job_postings.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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

from generate_embeddings import TEXT_COLUMNS, get_cache_path  # noqa: E402

FRAUD_TEXT_COLS = TEXT_COLUMNS["fraud"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OTHER_GROUP_LABEL = "_OTHER_"
LOW_CARD_COLS = ["employment_type", "required_experience", "required_education"]
MED_CARD_COLS = ["industry", "function"]
DEFAULT_COV = "spherical"
DEFAULT_SVD_DIM = 384
DEFAULT_N_TREES_TEXT = 100
DEFAULT_N_TREES_NONE = 700
TFIDF_KWARGS = dict(ngram_range=(1, 2), min_df=3, max_features=20000)
SUPPORTED_REPS = ["minilm", "bge-large", "openai-small", "openai-large", "tfidf", "none"]

# ---------------------------------------------------------------------------
# Helpers: text preprocessing
# ---------------------------------------------------------------------------

def col_to_texts(df: pd.DataFrame, col: str) -> list[str]:
    return df[col].fillna("").astype(str).tolist()


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
# Helpers: pipeline construction
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
                        cv=5, smooth="auto", target_type="binary",
                        random_state=seed,
                    )),
                ]),
                MED_CARD_COLS,
            ),
        ],
        remainder="drop",
    )


def build_structured_pipeline(binary_cols: list[str], n_trees: int, seed: int) -> Pipeline:
    """Structured features only -> RF. Used when --rep none."""
    return Pipeline([
        ("features", _structured_transformer(binary_cols, seed)),
        ("rf", RandomForestClassifier(n_estimators=n_trees, n_jobs=1, random_state=seed)),
    ])


def build_col_pipeline(
    emb_col_names: list[str],
    binary_cols: list[str],
    k: int,
    cov_type: str,
    n_trees: int,
    seed: int,
) -> Pipeline:
    """Embedding cols + structured -> GMM -> RF. Used as stacking base estimator."""
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
                                cv=5, smooth="auto", target_type="binary",
                                random_state=seed,
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


def build_augmented_dfs(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    col_tr: dict[str, np.ndarray],
    col_te: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Append per-column embedding arrays as named columns."""
    parts_tr = [X_tr.reset_index(drop=True)]
    parts_te = [X_te.reset_index(drop=True)]
    per_col_names: dict[str, list[str]] = {}

    for col in FRAUD_TEXT_COLS:
        arr_tr, arr_te = col_tr[col], col_te[col]
        names = [f"emb_{col}_{i}" for i in range(arr_tr.shape[1])]
        per_col_names[col] = names
        parts_tr.append(pd.DataFrame(arr_tr, columns=names))
        parts_te.append(pd.DataFrame(arr_te, columns=names))

    return (
        pd.concat(parts_tr, axis=1),
        pd.concat(parts_te, axis=1),
        per_col_names,
    )


# ---------------------------------------------------------------------------
# Helpers: metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())
    auc = float(roc_auc_score(y_true, prob)) if (n_pos > 0 and n_neg > 0) else float("nan")
    ll = float(log_loss(y_true, prob, labels=[0, 1]))
    br = float(brier_score_loss(y_true, prob))
    return {"auc": auc, "logloss": ll, "brier": br}


# ---------------------------------------------------------------------------
# Per-fold worker (module-level for joblib pickling)
# ---------------------------------------------------------------------------

def run_one_fold(
    fold_spec: dict,
    X: pd.DataFrame,
    y: pd.Series,
    col_embs_full: dict[str, np.ndarray] | None,
    metadata: dict,
    rep: str,
    k_values: list[int],
    cov: str,
    svd_dim: int,
    n_trees_text: int,
    n_trees_none: int,
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

    X_tr = X.iloc[train_idx]
    X_te = X.iloc[test_idx]
    y_tr = y.iloc[train_idx]
    y_te = y.iloc[test_idx]
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

    if rep == "none":
        pipe = build_structured_pipeline(binary_cols, n_trees_none, seed)
        pipe.fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_te)[:, 1]
        preds["prob_rf"] = prob
        for metric, val in compute_metrics(y_te.values, prob).items():
            metrics_row[f"{metric}_rf"] = val

    else:
        # Build representations once per fold
        if rep in ("minilm", "bge-large", "openai-small", "openai-large"):
            col_tr = {c: col_embs_full[c][train_idx] for c in FRAUD_TEXT_COLS}
            col_te = {c: col_embs_full[c][test_idx] for c in FRAUD_TEXT_COLS}
        else:  # tfidf
            col_tr, col_te = {}, {}
            for c in FRAUD_TEXT_COLS:
                r_tr, r_te = fit_tfidf_svd(
                    col_to_texts(X_tr, c), col_to_texts(X_te, c), svd_dim, seed
                )
                col_tr[c] = r_tr
                col_te[c] = r_te

        X_aug_tr, X_aug_te, per_col_names = build_augmented_dfs(X_tr, X_te, col_tr, col_te)

        for k in k_values:
            base_estimators = [
                (col, build_col_pipeline(
                    per_col_names[col], binary_cols, k, cov, n_trees_text, seed
                ))
                for col in FRAUD_TEXT_COLS
            ]
            stacking = StackingClassifier(
                estimators=base_estimators,
                final_estimator=LogisticRegression(max_iter=1000, random_state=seed),
                cv=5,
                n_jobs=1,
            )
            stacking.fit(X_aug_tr, y_tr)
            prob = stacking.predict_proba(X_aug_te)[:, 1]
            preds[f"prob_stack_k{k}"] = prob
            for metric, val in compute_metrics(y_te.values, prob).items():
                metrics_row[f"{metric}_k{k}"] = val

    return metrics_row, pd.DataFrame(preds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fraud CV k-sweep: single-rep, multi-K, LOGO or RSKF."
    )
    parser.add_argument("--rep", choices=SUPPORTED_REPS, default="minilm")
    parser.add_argument("--cv-method", choices=["logo", "rskf"], default="rskf")
    parser.add_argument("--n-repeats", type=int, default=10,
                        help="RSKF: number of repeats. [10]")
    parser.add_argument("--n-folds", type=int, default=10,
                        help="RSKF: number of folds per repeat. [10]")
    parser.add_argument(
        "--k-values", type=str, default="5,10,20,50,100",
        help="Comma-separated GMM n_components (ignored when --rep none). [5,10,20,50,100]",
    )
    parser.add_argument("--cov", type=str, default=DEFAULT_COV,
                        help="GMM covariance_type (ignored when --rep none). [spherical]")
    parser.add_argument("--svd-dim", type=int, default=DEFAULT_SVD_DIM)
    parser.add_argument("--n-trees-text", type=int, default=DEFAULT_N_TREES_TEXT,
                        help="RF trees per stacking base estimator. [100]")
    parser.add_argument("--n-trees-none", type=int, default=DEFAULT_N_TREES_NONE,
                        help="RF trees for no-rep (structured-only) pipeline. [700]")
    parser.add_argument("--min-group-size", type=int, default=10,
                        help="LOGO: merge industries smaller than this into _OTHER_. [10]")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Preds CSV path. Metrics CSV and config JSON are derived from the stem.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Directory containing fake_job_postings.csv. Defaults to "
             "~/.tabullm/fraud/ (tabullm default).",
    )
    args = parser.parse_args()

    k_values = [int(x.strip()) for x in args.k_values.split(",")]

    # Output paths derived from a common stem
    tfidf_suffix = (
        f"_svd{args.svd_dim}"
        if args.rep == "tfidf" and args.svd_dim != DEFAULT_SVD_DIM
        else ""
    )
    default_stem = f"fraud_{args.cv_method}_{args.rep}{tfidf_suffix}"
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

    # Banner
    print(f"Fraud CV k-sweep: rep={args.rep}  cv={args.cv_method}  "
          f"seed={args.seed}  n_jobs={args.n_jobs}")
    if args.rep == "none":
        print(f"  structured only: n_trees_none={args.n_trees_none}")
    else:
        print(f"  k_values={k_values}  cov={args.cov}  n_trees_text={args.n_trees_text}")
        if args.rep == "tfidf":
            print(f"  tfidf: svd_dim={args.svd_dim}  {TFIDF_KWARGS}")
    if args.cv_method == "rskf":
        print(f"  rskf: {args.n_repeats} repeats × {args.n_folds} folds "
              f"= {args.n_repeats * args.n_folds} total folds")
    print()

    # Save config before any heavy work
    config = {
        "rep": args.rep,
        "cv_method": args.cv_method,
        "n_repeats": args.n_repeats,
        "n_folds": args.n_folds,
        "k_values": k_values,
        "cov": args.cov,
        "svd_dim": args.svd_dim,
        "n_trees_text": args.n_trees_text,
        "n_trees_none": args.n_trees_none,
        "min_group_size": args.min_group_size,
        "n_jobs": args.n_jobs,
        "checkpoint_every": args.checkpoint_every,
        "seed": args.seed,
        "output_preds": str(out_preds),
        "output_metrics": str(out_metrics),
        "data_dir": str(args.data_dir) if args.data_dir else None,
    }
    with open(out_config, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {out_config}")
    print()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    X, y, metadata = load_fraud(data_dir=args.data_dir)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # ------------------------------------------------------------------
    # CV setup and fold construction
    # Fold construction happens here, before embedding loading, so that
    # splits are determined by seed alone and are identical across reps.
    # ------------------------------------------------------------------
    if args.cv_method == "logo":
        mask = X["industry"].notna()
        surviving = np.where(mask.values)[0]
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)

        sizes = X["industry"].value_counts()
        small = sizes[sizes < args.min_group_size].index.tolist()
        industry_group = X["industry"].where(
            ~X["industry"].isin(small), other=OTHER_GROUP_LABEL
        )
        X = X.copy()
        X["industry_group"] = industry_group
        groups = X["industry_group"].values
        other_rows = int(X["industry"].isin(small).sum())
        n_groups = len(np.unique(groups))

        print(f"  n={len(X)} (dropped {(~mask).sum()} missing-industry rows) | "
              f"fraud_rate={y.mean():.4f}")
        print(f"  {n_groups} folds: {n_groups - 1} named industries + "
              f"_OTHER_ ({other_rows} rows from {len(small)} small industries, "
              f"min_group_size={args.min_group_size})")
        print()

        logo = LeaveOneGroupOut()
        fold_specs = [
            {
                "group_id": groups[test_idx][0],
                "repeat_id": 0,
                "fold_id": i,
                "train_idx": train_idx,
                "test_idx": test_idx,
            }
            for i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups))
        ]

    else:  # rskf
        surviving = np.arange(len(X))

        print(f"  n={len(X)}  fraud_rate={y.mean():.4f}")
        print(f"  rskf: {args.n_repeats} repeats × {args.n_folds} folds "
              f"= {args.n_repeats * args.n_folds} total folds")
        print()

        rskf = RepeatedStratifiedKFold(
            n_splits=args.n_folds,
            n_repeats=args.n_repeats,
            random_state=args.seed,
        )
        fold_specs = [
            {
                "group_id": f"r{i // args.n_folds}_f{i % args.n_folds}",
                "repeat_id": i // args.n_folds,
                "fold_id": i % args.n_folds,
                "train_idx": train_idx,
                "test_idx": test_idx,
            }
            for i, (train_idx, test_idx) in enumerate(rskf.split(X, y))
        ]

    # ------------------------------------------------------------------
    # Load per-column embedding caches (text reps only)
    # ------------------------------------------------------------------
    col_embs_full: dict[str, np.ndarray] | None = None

    if args.rep in ("minilm", "bge-large", "openai-small", "openai-large"):
        print(f"Loading {args.rep} per-column embedding caches...")
        col_embs_full = {}
        for col in FRAUD_TEXT_COLS:
            path = get_cache_path("fraud", args.rep, [col])
            arr = np.load(path)[surviving]
            col_embs_full[col] = arr
            print(f"  {col:20s}: {path.name}  shape={arr.shape}")
        print()
    elif args.rep == "tfidf":
        print("TF-IDF: representations will be fit per fold (no cache).")
        print()
    else:  # none
        print("No text representation: structured features only.")
        print()

    # ------------------------------------------------------------------
    # Progress header
    # ------------------------------------------------------------------
    metric_cols = ["auc_rf"] if args.rep == "none" else [f"auc_k{k}" for k in k_values]
    hdr = "  ".join(f"{c:>10}" for c in metric_cols)
    print(f"Running {len(fold_specs)} folds (checkpoint every {args.checkpoint_every})...")
    print()
    print(f"  {'group_id':<30} {'n_te':>5} {'n+':>4}  {hdr}")
    print("  " + "-" * (42 + 12 * len(metric_cols)))

    # ------------------------------------------------------------------
    # Parallel execution with live progress
    # ------------------------------------------------------------------
    all_metrics: list[dict] = []
    all_preds: list[pd.DataFrame] = []

    gen = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
        delayed(run_one_fold)(
            fold_spec, X, y, col_embs_full, metadata,
            args.rep, k_values, args.cov, args.svd_dim,
            args.n_trees_text, args.n_trees_none, args.seed,
            surviving,
        )
        for fold_spec in fold_specs
    )

    def fmt(v: float) -> str:
        return f"{v:.4f}" if not np.isnan(v) else "  NaN "

    for i, (metrics_row, preds_df) in enumerate(gen, 1):
        all_metrics.append(metrics_row)
        all_preds.append(preds_df)

        auc_vals = "  ".join(
            f"{fmt(metrics_row.get(c, float('nan'))):>10}" for c in metric_cols
        )
        print(
            f"  {metrics_row['group_id']:<30} {metrics_row['n_test']:>5} "
            f"{metrics_row['n_pos']:>4}  {auc_vals}  [{i}/{len(fold_specs)}]"
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
            print(f"  [checkpoint] {i}/{len(fold_specs)} folds → {out_preds.name}")

    print()

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
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

    print(f"OOS predictions saved to {out_preds}")
    print(f"Per-fold metrics  saved to {out_metrics}")
    print()

    # ------------------------------------------------------------------
    # Aggregate summary (stdout only)
    # ------------------------------------------------------------------
    weights = (df_metrics["n_pos"] * df_metrics["n_neg"]).astype(float)
    w_sum = weights.sum()
    n_valid = int((weights > 0).sum())

    tags   = ["rf"]                          if args.rep == "none" else [f"k{k}"       for k in k_values]
    labels = ["structured_rf"]               if args.rep == "none" else [f"stack_k{k}" for k in k_values]

    print(f"Aggregate (weight = n_pos * n_neg, {n_valid} non-degenerate folds):")
    print(f"  {'pipeline':<18}  {'wtd AUC':>8}  {'wtd log_loss':>12}  {'wtd brier':>10}")
    print("  " + "-" * 56)
    for tag, label in zip(tags, labels):
        auc_col = f"auc_{tag}"
        ll_col  = f"logloss_{tag}"
        br_col  = f"brier_{tag}"
        w_valid = weights.where(df_metrics[auc_col].notna(), 0.0)
        wtd_auc = float((df_metrics[auc_col].fillna(0) * w_valid).sum() / w_valid.sum()) if w_valid.sum() > 0 else float("nan")
        wtd_ll  = float((df_metrics[ll_col]  * weights).sum() / w_sum) if w_sum > 0 else float("nan")
        wtd_br  = float((df_metrics[br_col]  * weights).sum() / w_sum) if w_sum > 0 else float("nan")
        print(f"  {label:<18}  {wtd_auc:>8.4f}  {wtd_ll:>12.4f}  {wtd_br:>10.4f}")


if __name__ == "__main__":
    main()
