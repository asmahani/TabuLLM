# Copyright (c) 2024 Alireza S. Mahani and Mansour T.A. Sharabiani
# Licensed under the MIT License. See LICENSE file in the project root.

from pathlib import Path
import pandas as pd
import urllib.request

_ZENODO_URL = "https://zenodo.org/records/18884002/files/fake_job_postings.csv"
_CSV_FILENAME = "fake_job_postings.csv"
_DEFAULT_DATA_DIR = Path.home() / ".tabullm" / "fraud"


def load_fraud(data_dir=None, return_metadata=True):
    """
    Load and preprocess the Real or Fake Job Posting Prediction dataset.

    If the dataset is not found locally, downloads it automatically from Zenodo
    (no credentials required). If the download fails, a FileNotFoundError is
    raised with manual download instructions.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing ``fake_job_postings.csv`` (or where it will be
        downloaded). Defaults to ``~/.tabullm/fraud/``.
    return_metadata : bool, default=True
        If True, returns ``(X, y, metadata)``. If False, returns ``(X, y)``.

    Returns
    -------
    X : pandas.DataFrame, shape (n_samples, 15)
        Features: 7 text columns, 3 binary columns, 5 categorical columns.
    y : pandas.Series, shape (n_samples,)
        Target variable (``fraudulent``: 0 = legitimate, 1 = fraudulent).
    metadata : dict
        Dataset metadata including column categorization, class distribution,
        and missing value summary. Only returned when ``return_metadata=True``.

    Notes
    -----
    The dataset is highly imbalanced (~4.84% fraud). Consider using
    ``class_weight="balanced"`` in downstream classifiers.

    Dataset: Real or Fake Job Posting Prediction (Vidros et al., 2017)
    Source: https://doi.org/10.5281/zenodo.18884002
    License: ODbL v1.0
    """
    data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
    csv_file = data_dir / _CSV_FILENAME

    if not csv_file.exists():
        _download_fraud(data_dir, csv_file)

    df = pd.read_csv(csv_file)

    text_columns = [
        'title',
        'location',
        'department',
        'company_profile',
        'description',
        'requirements',
        'benefits',
    ]
    binary_columns = [
        'telecommuting',
        'has_company_logo',
        'has_questions',
    ]
    categorical_columns = [
        'employment_type',
        'required_experience',
        'required_education',
        'industry',
        'function',
    ]
    target_column = 'fraudulent'
    excluded_columns = {
        'job_id': 'Identifier (not predictive)',
        'salary_range': 'High missingness (84%) exceeds 50% threshold',
    }

    feature_columns = text_columns + binary_columns + categorical_columns

    unaccounted = set(df.columns) - set(feature_columns) - {target_column} - set(excluded_columns)
    if unaccounted:
        print(f"Warning: Unaccounted columns: {unaccounted}")

    missing_cols = [c for c in feature_columns + [target_column] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    if not return_metadata:
        return X, y

    metadata = {
        'dataset_name': 'fraud',
        'task_type': 'binary_classification',
        'target_name': target_column,
        'class_distribution': {
            'legitimate (0)': int((y == 0).sum()),
            'fraudulent (1)': int((y == 1).sum()),
            'fraud_rate': float(y.mean()),
        },
        'n_samples': len(X),
        'n_features': len(feature_columns),
        'text_columns': text_columns,
        'binary_columns': binary_columns,
        'categorical_columns': categorical_columns,
        'numeric_columns': [],
        'excluded_columns': excluded_columns,
        'missing_summary': {
            col: {
                'count': int(X[col].isnull().sum()),
                'percent': float(100 * X[col].isnull().sum() / len(X)),
            }
            for col in feature_columns if X[col].isnull().sum() > 0
        },
        'preprocessing_rules': {
            'text': 'NaN -> empty string',
            'binary': 'Keep as-is (0/1)',
            'categorical': 'NaN -> "missing" category (in Pipeline)',
            'target': 'Binary (0/1), no transformation',
        },
        'notes': 'Highly imbalanced (4.84% fraud) - use class_weight="balanced"',
    }

    return X, y, metadata


def _download_fraud(data_dir: Path, csv_file: Path):
    """Download fraud dataset from Zenodo (no credentials required)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading fraud dataset from Zenodo to {data_dir} ...")
    try:
        urllib.request.urlretrieve(_ZENODO_URL, csv_file)
    except Exception as exc:
        raise FileNotFoundError(
            f"Auto-download from Zenodo failed: {exc}\n\n"
            f"Manual download: https://doi.org/10.5281/zenodo.18884002\n"
            f"Place fake_job_postings.csv at: {csv_file}"
        ) from None
    print("Download complete.")