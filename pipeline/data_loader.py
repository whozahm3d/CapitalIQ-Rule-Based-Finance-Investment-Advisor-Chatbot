"""
pipeline/data_loader.py
DecodeLabs AI Project 2 — Data Classification
Author: Ali Ahmad | DecodeLabs Batch 2026

Handles dataset loading with HuggingFace primary source and sklearn fallback,
column normalisation, species name mapping, and SHA-256 data versioning hash.
"""

import hashlib
import datetime
import pandas as pd

FEATURE_NAMES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
TARGET_NAMES  = ['setosa', 'versicolor', 'virginica']


def load_data() -> tuple:
    """
    Load the Iris dataset. Tries HuggingFace first; falls back to sklearn.

    Returns:
        tuple: (clean_df: pd.DataFrame, data_hash: str, source_used: str)
            clean_df   — DataFrame with columns FEATURE_NAMES + ['target', 'species']
            data_hash  — SHA-256 hex digest of the raw CSV representation
            source_used — 'huggingface' or 'sklearn'

    Raises:
        RuntimeError: if both sources fail to load.
    """
    df = None
    source_used = None

    # ── Primary: HuggingFace ────────────────────────────────────────────────
    try:
        from datasets import load_dataset
        ds = load_dataset("scikit-learn/iris", split="train")
        df = ds.to_pandas()
        source_used = 'huggingface'
        print("[DataLoader] Source: HuggingFace (scikit-learn/iris)")
    except Exception as hf_err:
        print(f"[DataLoader] HuggingFace load failed ({hf_err}). Falling back to sklearn.")

    # ── Fallback: sklearn ───────────────────────────────────────────────────
    if df is None:
        try:
            from sklearn.datasets import load_iris
            iris = load_iris(as_frame=True)
            df = iris.frame.copy()
            source_used = 'sklearn'
            print("[DataLoader] Source: sklearn built-in dataset")
        except Exception as sk_err:
            raise RuntimeError(
                f"[DataLoader] Both data sources failed. sklearn error: {sk_err}"
            )

    # ── Normalise column names ──────────────────────────────────────────────
    df = _normalise_columns(df)

    # ── Map numeric target → species string ────────────────────────────────
    df = _map_species(df)

    # ── Compute SHA-256 hash ────────────────────────────────────────────────
    data_hash, hash_timestamp = _compute_hash(df)

    print(f"[DataLoader] DataFrame shape : {df.shape}")
    print(f"[DataLoader] Hash computed at: {hash_timestamp}")

    return df, data_hash, source_used


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise column names to FEATURE_NAMES + 'target'.

    Handles both HuggingFace naming (SepalLengthCm style or feature_0..3 style)
    and sklearn naming (sepal length (cm) style).

    Args:
        df: Raw DataFrame from either source.

    Returns:
        pd.DataFrame with standardised column names.
    """
    cols = list(df.columns)

    # sklearn as_frame=True uses 'sepal length (cm)' style + 'target'
    sklearn_map = {
        'sepal length (cm)': 'sepal_length',
        'sepal width (cm)':  'sepal_width',
        'petal length (cm)': 'petal_length',
        'petal width (cm)':  'petal_width',
    }

    # HuggingFace scikit-learn/iris uses these column names
    hf_map = {
        'SepalLengthCm': 'sepal_length',
        'SepalWidthCm':  'sepal_width',
        'PetalLengthCm': 'petal_length',
        'PetalWidthCm':  'petal_width',
        'Species':       'species_raw',
    }

    # Alternative HF column set
    hf_map2 = {
        'sepal.length': 'sepal_length',
        'sepal.width':  'sepal_width',
        'petal.length': 'petal_length',
        'petal.width':  'petal_width',
        'variety':      'species_raw',
    }

    # Check which mapping applies
    if all(k in cols for k in sklearn_map):
        df = df.rename(columns=sklearn_map)
    elif all(k in cols for k in hf_map):
        df = df.rename(columns=hf_map)
    elif all(k in cols for k in hf_map2):
        df = df.rename(columns=hf_map2)
    else:
        # Generic fallback: assume first 4 numeric cols are features, last is target
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        for i, fname in enumerate(FEATURE_NAMES):
            if i < len(numeric_cols):
                df = df.rename(columns={numeric_cols[i]: fname})

    return df


def _map_species(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has both a numeric 'target' column and a string 'species' column.

    Args:
        df: DataFrame after column normalisation.

    Returns:
        pd.DataFrame with 'target' (int 0/1/2) and 'species' (str) columns.
    """
    # If sklearn-style numeric target exists
    if 'target' in df.columns and 'species' not in df.columns:
        df['species'] = df['target'].map(
            {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        )

    # If HF-style string species exists
    elif 'species_raw' in df.columns:
        # Normalise species strings
        species_norm = (
            df['species_raw']
            .astype(str)
            .str.lower()
            .str.replace('iris-', '', regex=False)
            .str.strip()
        )
        species_to_int = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
        df['target']  = species_norm.map(species_to_int)
        df['species'] = species_norm
        df = df.drop(columns=['species_raw'])

    elif 'species' in df.columns and 'target' not in df.columns:
        species_norm = (
            df['species']
            .astype(str)
            .str.lower()
            .str.replace('iris-', '', regex=False)
            .str.strip()
        )
        species_to_int = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
        df['target']  = species_norm.map(species_to_int)
        df['species'] = species_norm

    # Keep only necessary columns
    keep = FEATURE_NAMES + ['target', 'species']
    available = [c for c in keep if c in df.columns]
    df = df[available].reset_index(drop=True)

    return df


def _compute_hash(df: pd.DataFrame) -> tuple:
    """
    Compute SHA-256 hash of the DataFrame CSV representation.

    Args:
        df: Clean DataFrame.

    Returns:
        tuple: (data_hash: str, timestamp: str)
            data_hash — 64-character hex string
            timestamp — ISO-format string of hash computation time
    """
    df_bytes   = df.to_csv(index=False).encode('utf-8')
    data_hash  = hashlib.sha256(df_bytes).hexdigest()
    timestamp  = datetime.datetime.now().isoformat()
    return data_hash, timestamp
