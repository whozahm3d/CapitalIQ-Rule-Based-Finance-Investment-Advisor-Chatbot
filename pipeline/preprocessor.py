"""
pipeline/preprocessor.py
DecodeLabs AI Project 2 — Data Classification
Author: Ali Ahmad | DecodeLabs Batch 2026

Handles X/y splitting, stratified train-test split, StandardScaler
with leakage guard, IQR outlier detection, and shape validation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
RANDOM_STATE  = 42
TEST_SIZE     = 0.20


def split_features_target(df: pd.DataFrame) -> tuple:
    """
    Split DataFrame into feature matrix X and target vector y.

    Args:
        df: Clean DataFrame with FEATURE_NAMES + 'species' columns.

    Returns:
        tuple: (X: pd.DataFrame, y: pd.Series)

    Raises:
        KeyError: if required columns are missing from df.
    """
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    if 'species' not in df.columns:
        raise KeyError("Missing target column 'species'")

    X = df[FEATURE_NAMES].copy()
    y = df['species'].copy()
    return X, y


def stratified_split(X: pd.DataFrame, y: pd.Series,
                     test_size: float = TEST_SIZE,
                     random_state: int = RANDOM_STATE) -> tuple:
    """
    Perform stratified train-test split preserving class proportions.

    Args:
        X            : Feature matrix.
        y            : Target vector.
        test_size    : Fraction of data reserved for testing (default 0.20).
        random_state : Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        shuffle=True,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Fit StandardScaler on training data ONLY, then transform both sets.
    Enforces strict leakage guard — scaler never sees X_test during fit.

    Args:
        X_train: Training feature matrix (unscaled).
        X_test : Test feature matrix (unscaled).

    Returns:
        tuple: (X_train_scaled: np.ndarray,
                X_test_scaled: np.ndarray,
                scaler: fitted StandardScaler)
    """
    scaler = StandardScaler()

    # Fit ONLY on training data — critical anti-leakage rule
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform test data using statistics from training data
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers per feature per class using the IQR method.
    Bounds: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]

    Args:
        df: Clean DataFrame with FEATURE_NAMES + 'species'.

    Returns:
        pd.DataFrame with columns:
            feature | class | n_outliers | outlier_indices
    """
    records = []

    for feature in FEATURE_NAMES:
        for cls in df['species'].unique():
            subset = df[df['species'] == cls][feature]

            q1  = subset.quantile(0.25)
            q3  = subset.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outlier_mask    = (subset < lower) | (subset > upper)
            outlier_indices = subset[outlier_mask].index.tolist()

            records.append({
                'feature':         feature,
                'class':           cls,
                'n_outliers':      int(outlier_mask.sum()),
                'outlier_indices': outlier_indices,
            })

    summary_df = pd.DataFrame(records)
    return summary_df


def validate_shapes(X_train: pd.DataFrame, X_test: pd.DataFrame,
                    y_train: pd.Series, y_test: pd.Series,
                    X_train_scaled: np.ndarray,
                    X_test_scaled: np.ndarray) -> None:
    """
    Assert that shapes of all split and scaled arrays are consistent.

    Args:
        X_train        : Unscaled training features.
        X_test         : Unscaled test features.
        y_train        : Training labels.
        y_test         : Test labels.
        X_train_scaled : Scaled training features.
        X_test_scaled  : Scaled test features.

    Raises:
        AssertionError: if any shape mismatch is detected.
    """
    assert X_train.shape[0] == y_train.shape[0], \
        f"X_train rows ({X_train.shape[0]}) != y_train rows ({y_train.shape[0]})"
    assert X_test.shape[0] == y_test.shape[0], \
        f"X_test rows ({X_test.shape[0]}) != y_test rows ({y_test.shape[0]})"
    assert X_train_scaled.shape == X_train.shape, \
        f"X_train_scaled shape mismatch"
    assert X_test_scaled.shape == X_test.shape, \
        f"X_test_scaled shape mismatch"
    assert X_train.shape[1] == len(FEATURE_NAMES), \
        f"Expected {len(FEATURE_NAMES)} features, got {X_train.shape[1]}"
