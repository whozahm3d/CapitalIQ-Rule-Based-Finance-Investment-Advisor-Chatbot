"""
tests/test_pipeline.py
DecodeLabs AI Project 2 — Data Classification
Author: Ali Ahmad | DecodeLabs Batch 2026

22 unit tests covering data_loader, preprocessor, trainer, and evaluator.
Run with:  pytest tests/test_pipeline.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import pandas as pd

from sklearn.datasets       import load_iris
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.tree           import DecisionTreeClassifier
from sklearn.svm            import SVC
from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import VotingClassifier
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler

from pipeline.data_loader  import _normalise_columns, _map_species, _compute_hash
from pipeline.preprocessor import (split_features_target, stratified_split,
                                    scale_features, detect_outliers_iqr,
                                    validate_shapes)
from pipeline.trainer       import (train_knn, train_decision_tree, train_svm,
                                    train_logistic_regression, train_all_models,
                                    build_voting_classifier, build_pipeline)
from pipeline.evaluator     import (compute_metrics, cross_validate_all,
                                    statistical_significance_test,
                                    weighted_f1, compute_weighted_f1,
                                    compute_calibration_score)

FEATURE_NAMES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
TARGET_NAMES  = ['setosa', 'versicolor', 'virginica']


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def clean_df():
    """Fixture: clean Iris DataFrame from sklearn."""
    iris = load_iris(as_frame=True)
    df   = iris.frame.copy()
    df   = df.rename(columns={
        'sepal length (cm)': 'sepal_length',
        'sepal width (cm)':  'sepal_width',
        'petal length (cm)': 'petal_length',
        'petal width (cm)':  'petal_width',
    })
    df['species'] = df['target'].map(
        {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    )
    return df[FEATURE_NAMES + ['target', 'species']]


@pytest.fixture(scope='module')
def split_data(clean_df):
    """Fixture: fully split and scaled data."""
    X, y = split_features_target(clean_df)
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
    return X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler


@pytest.fixture(scope='module')
def trained_knn(split_data):
    """Fixture: trained KNN model."""
    _, _, _, _, X_train_s, _, _ = split_data
    _, _, y_train, _, _, _, _   = split_data
    return train_knn(X_train_s, y_train, n_neighbors=5)


@pytest.fixture(scope='module')
def all_models(split_data):
    """Fixture: all 5 trained models."""
    _, _, y_train, _, X_train_s, _, _ = split_data
    return train_all_models(X_train_s, y_train, optimal_k=5)


# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — data_loader  (Tests 1–5)
# ══════════════════════════════════════════════════════════════════════════════

class TestDataLoader:

    def test_01_df_shape(self, clean_df):
        """Dataset must have exactly 150 rows and 6 columns."""
        assert clean_df.shape == (150, 6), \
            f"Expected (150, 6), got {clean_df.shape}"

    def test_02_feature_columns_present(self, clean_df):
        """All four feature columns must exist."""
        for col in FEATURE_NAMES:
            assert col in clean_df.columns, f"Missing column: {col}"

    def test_03_no_null_values(self, clean_df):
        """DataFrame must contain zero null values."""
        assert clean_df.isnull().sum().sum() == 0, "Null values found in DataFrame"

    def test_04_class_balance(self, clean_df):
        """Each species must have exactly 50 samples (balanced dataset)."""
        counts = clean_df['species'].value_counts()
        for species in TARGET_NAMES:
            assert counts[species] == 50, \
                f"Expected 50 samples for {species}, got {counts.get(species, 0)}"

    def test_05_hash_is_sha256(self, clean_df):
        """SHA-256 hash must be a 64-character hex string."""
        data_hash, _ = _compute_hash(clean_df)
        assert len(data_hash) == 64,          "Hash must be 64 characters"
        assert all(c in '0123456789abcdef'
                   for c in data_hash.lower()), "Hash must be hexadecimal"


# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — preprocessor  (Tests 6–11)
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessor:

    def test_06_split_shapes(self, split_data):
        """Train set must be ~80%, test set ~20% of 150 samples."""
        X_train, X_test, y_train, y_test, _, _, _ = split_data
        assert X_train.shape[0] + X_test.shape[0] == 150
        assert abs(X_test.shape[0] - 30) <= 2, \
            f"Expected ~30 test samples, got {X_test.shape[0]}"

    def test_07_stratification_maintained(self, split_data):
        """Each class must appear in test set with ~equal representation."""
        _, _, _, y_test, _, _, _ = split_data
        counts = y_test.value_counts()
        for cls in TARGET_NAMES:
            assert cls in counts, f"{cls} missing from test set"
            assert 8 <= counts[cls] <= 12, \
                f"Stratification off for {cls}: {counts[cls]} samples"

    def test_08_scaler_no_leakage(self, split_data):
        """Scaler mean and variance must be computed from training data only."""
        X_train, _, _, _, X_train_s, _, scaler = split_data
        expected_mean = X_train.mean().values
        np.testing.assert_allclose(
            scaler.mean_, expected_mean, atol=1e-6,
            err_msg="Scaler mean does not match training data mean"
        )

    def test_09_scaled_mean_near_zero(self, split_data):
        """Scaled training features must have mean ≈ 0 (within 0.01)."""
        _, _, _, _, X_train_s, _, _ = split_data
        col_means = X_train_s.mean(axis=0)
        np.testing.assert_allclose(
            col_means, np.zeros(4), atol=0.01,
            err_msg=f"Scaled train means not near zero: {col_means}"
        )

    def test_10_scaled_std_near_one(self, split_data):
        """Scaled training features must have std ≈ 1 (within 0.05)."""
        _, _, _, _, X_train_s, _, _ = split_data
        col_stds = X_train_s.std(axis=0)
        np.testing.assert_allclose(
            col_stds, np.ones(4), atol=0.05,
            err_msg=f"Scaled train stds not near one: {col_stds}"
        )

    def test_11_outlier_detection_output_format(self, clean_df):
        """IQR outlier detection must return DataFrame with correct columns."""
        result = detect_outliers_iqr(clean_df)
        required_cols = {'feature', 'class', 'n_outliers', 'outlier_indices'}
        assert required_cols.issubset(set(result.columns)), \
            f"Missing columns: {required_cols - set(result.columns)}"
        assert len(result) == len(FEATURE_NAMES) * len(TARGET_NAMES), \
            f"Expected {len(FEATURE_NAMES) * len(TARGET_NAMES)} rows, got {len(result)}"


# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — trainer  (Tests 12–17)
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainer:

    def test_12_knn_type(self, trained_knn):
        """train_knn must return a KNeighborsClassifier."""
        assert isinstance(trained_knn, KNeighborsClassifier)

    def test_13_knn_accuracy_above_90(self, split_data, trained_knn):
        """KNN (k=5) must achieve ≥ 90% accuracy on the test set."""
        _, X_test_s, _, y_test, _, _, _ = split_data
        _, _, _, y_test, _, X_test_s, _ = split_data
        score = trained_knn.score(X_test_s, y_test)
        assert score >= 0.90, f"KNN accuracy too low: {score:.4f}"

    def test_14_all_models_keys(self, all_models):
        """train_all_models must return exactly 5 models with correct keys."""
        expected_keys = {'KNN', 'Decision Tree', 'SVM',
                         'Logistic Regression', 'Ensemble'}
        assert set(all_models.keys()) == expected_keys, \
            f"Model keys mismatch: {set(all_models.keys())}"

    def test_15_voting_classifier_type(self):
        """build_voting_classifier must return a VotingClassifier."""
        vc = build_voting_classifier()
        assert isinstance(vc, VotingClassifier)
        assert vc.voting == 'soft', "VotingClassifier must use soft voting"

    def test_16_pipeline_structure(self):
        """build_pipeline must produce Pipeline with 'scaler' and 'classifier' steps."""
        model    = KNeighborsClassifier(n_neighbors=5)
        pipeline = build_pipeline(model)
        assert isinstance(pipeline, Pipeline), "Must return sklearn Pipeline"
        assert 'scaler'     in dict(pipeline.steps), "Pipeline missing 'scaler' step"
        assert 'classifier' in dict(pipeline.steps), "Pipeline missing 'classifier' step"
        assert isinstance(pipeline.named_steps['scaler'], StandardScaler)

    def test_17_all_models_have_predict_proba(self, all_models):
        """All 5 models must have predict_proba (required for ROC and SHAP)."""
        for name, model in all_models.items():
            assert hasattr(model, 'predict_proba'), \
                f"{name} is missing predict_proba method"


# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — evaluator  (Tests 18–22)
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluator:

    def test_18_metrics_keys(self, all_models, split_data):
        """compute_metrics must return all 5 required metric keys."""
        _, _, _, y_test, _, X_test_s, _ = split_data
        metrics = compute_metrics(all_models['KNN'], X_test_s, y_test)
        required = {'accuracy', 'f1', 'precision', 'recall', 'roc_auc'}
        assert required == set(metrics.keys()), \
            f"Metric keys mismatch: {set(metrics.keys())}"

    def test_19_metrics_range(self, all_models, split_data):
        """All metric values must be in [0, 1]."""
        _, _, _, y_test, _, X_test_s, _ = split_data
        for name, model in all_models.items():
            metrics = compute_metrics(model, X_test_s, y_test)
            for k, v in metrics.items():
                assert 0.0 <= v <= 1.0, \
                    f"{name} — {k} out of range: {v}"

    def test_20_weighted_f1_range(self, all_models, split_data):
        """weighted_f1 must return a value in [0, 1]."""
        _, _, _, y_test, _, X_test_s, _ = split_data
        score = compute_weighted_f1(all_models['KNN'], X_test_s, y_test)
        assert 0.0 <= score <= 1.0, f"Weighted F1 out of range: {score}"

    def test_21_cv_scores_length(self, all_models, split_data):
        """cross_validate_all must return n_splits scores per model."""
        _, _, y_train, _, X_train_s, _, _ = split_data
        # Use only KNN to keep test fast
        from sklearn.base import clone
        n_splits = 3
        cv_scores = cross_validate_all(
            {'KNN': clone(all_models['KNN'])},
            X_train_s, y_train,
            n_splits=n_splits
        )
        assert len(cv_scores['KNN']) == n_splits, \
            f"Expected {n_splits} CV scores, got {len(cv_scores['KNN'])}"

    def test_22_calibration_score_is_float(self, all_models, split_data):
        """compute_calibration_score must return a non-negative float."""
        _, _, _, y_test, _, X_test_s, _ = split_data
        score = compute_calibration_score(all_models['KNN'], X_test_s, y_test)
        assert isinstance(score, float), f"Calibration score must be float, got {type(score)}"
        assert score >= 0.0, f"Calibration score must be non-negative: {score}"
