"""
pipeline/evaluator.py
DecodeLabs AI Project 2 — Data Classification
Author: Ali Ahmad | DecodeLabs Batch 2026

Evaluation utilities: metrics, cross-validation, statistical significance,
custom scoring, calibration quality, and CSV experiment logging.
"""

import os
import csv
import datetime
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, make_scorer,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from scipy import stats

TARGET_NAMES = ['setosa', 'versicolor', 'virginica']
RANDOM_STATE = 42
N_SPLITS     = 5
LOG_DIR      = os.path.join(os.getcwd(), 'logs')
LOG_FILE     = os.path.join(LOG_DIR, 'experiment_log.csv')


# ── Core metrics ─────────────────────────────────────────────────────────────

def compute_metrics(model, X_test: np.ndarray, y_test) -> dict:
    """
    Compute the 5 standard evaluation metrics for a single model.

    Args:
        model : Fitted sklearn estimator with predict() and predict_proba().
        X_test: Scaled test feature matrix.
        y_test: True test labels.

    Returns:
        dict with keys: accuracy, f1, precision, recall, roc_auc.

    Raises:
        AttributeError: if model does not support predict_proba.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Binarize for multiclass ROC-AUC
    y_bin = label_binarize(y_test, classes=TARGET_NAMES)

    metrics = {
        'accuracy':  round(accuracy_score(y_test, y_pred),  4),
        'f1':        round(f1_score(y_test, y_pred,
                                    average='macro',
                                    zero_division=0),         4),
        'precision': round(precision_score(y_test, y_pred,
                                           average='macro',
                                           zero_division=0),  4),
        'recall':    round(recall_score(y_test, y_pred,
                                        average='macro',
                                        zero_division=0),     4),
        'roc_auc':   round(roc_auc_score(y_bin, y_proba,
                                         multi_class='ovr',
                                         average='macro'),    4),
    }
    return metrics


def full_report(model_name: str, model,
                X_test: np.ndarray, y_test) -> dict:
    """
    Compute metrics and print a full classification report with separators.

    Args:
        model_name: Display name string.
        model     : Fitted estimator.
        X_test    : Scaled test features.
        y_test    : True labels.

    Returns:
        dict: same as compute_metrics().
    """
    metrics = compute_metrics(model, X_test, y_test)
    y_pred  = model.predict(X_test)

    sep = "=" * 60
    print(sep)
    print(f"  MODEL: {model_name}")
    print(sep)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['f1']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print()
    print(classification_report(y_test, y_pred,
                                 target_names=TARGET_NAMES,
                                 zero_division=0))
    print(sep)
    return metrics


# ── Cross-validation ─────────────────────────────────────────────────────────

def cross_validate_all(models: dict,
                       X_train_scaled: np.ndarray,
                       y_train,
                       n_splits: int = N_SPLITS) -> dict:
    """
    Run StratifiedKFold cross-validation on all models.

    Args:
        models        : Dict {model_name: unfitted_estimator}.
                        NOTE: pass UNFITTED clones for CV integrity.
        X_train_scaled: Full training feature matrix (scaled).
        y_train       : Full training labels.
        n_splits      : Number of CV folds (default 5).

    Returns:
        dict: {model_name: np.ndarray of fold F1 scores (length n_splits)}
    """
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    cv_scores = {}

    for name, model in models.items():
        scores = cross_val_score(
            model, X_train_scaled, y_train,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1
        )
        cv_scores[name] = scores
        print(f"  {name:<25} CV F1: {scores.mean():.4f} ± {scores.std():.4f}")

    return cv_scores


# ── Statistical significance ─────────────────────────────────────────────────

def statistical_significance_test(cv_scores: dict) -> tuple:
    """
    Perform paired t-test between the best and second-best base model.

    Args:
        cv_scores: Dict {model_name: np.ndarray of fold scores}.

    Returns:
        tuple: (t_stat: float, p_value: float,
                best_model: str, second_model: str)
    """
    # Sort by mean score descending
    sorted_models = sorted(
        cv_scores.items(),
        key=lambda x: x[1].mean(),
        reverse=True
    )

    best_name,   best_scores   = sorted_models[0]
    second_name, second_scores = sorted_models[1]

    t_stat, p_value = stats.ttest_rel(best_scores, second_scores)

    return float(t_stat), float(p_value), best_name, second_name


# ── Custom weighted F1 scorer ────────────────────────────────────────────────

def weighted_f1(y_true, y_pred) -> float:
    """
    Custom domain-aware scorer applying higher penalty for
    Versicolor/Virginica confusion than for Setosa misclassification.

    Penalty weights: setosa=1.0, versicolor=2.0, virginica=2.0

    Args:
        y_true: array-like of true labels (strings from TARGET_NAMES).
        y_pred: array-like of predicted labels.

    Returns:
        float: weighted F1 score penalising high-confusion class pairs.

    Raises:
        ValueError: if label sets do not match TARGET_NAMES.
    """
    weights = {'setosa': 1.0, 'versicolor': 2.0, 'virginica': 2.0}

    try:
        sample_weights = np.array([weights[str(t)] for t in y_true])
    except KeyError as e:
        raise ValueError(
            f"Unexpected label {e}. Labels must be in {TARGET_NAMES}."
        )

    return f1_score(
        y_true, y_pred,
        average='weighted',
        sample_weight=sample_weights,
        zero_division=0
    )


def make_weighted_scorer():
    """
    Wrap weighted_f1 in sklearn's make_scorer for use with GridSearchCV.

    Returns:
        sklearn scorer object.
    """
    return make_scorer(weighted_f1)


def compute_weighted_f1(model, X_test: np.ndarray, y_test) -> float:
    """
    Compute the domain-aware weighted F1 for a fitted model on test data.

    Args:
        model : Fitted estimator.
        X_test: Scaled test features.
        y_test: True labels.

    Returns:
        float: weighted F1 score.
    """
    y_pred = model.predict(X_test)
    return weighted_f1(y_test, y_pred)


# ── Calibration quality ───────────────────────────────────────────────────────

def compute_calibration_score(model, X_test: np.ndarray, y_test) -> float:
    """
    Compute the mean absolute deviation of a model's calibration curves
    from the perfect diagonal across all 3 classes (OvR).

    A lower score indicates better calibration.

    Args:
        model : Fitted estimator with predict_proba().
        X_test: Scaled test features.
        y_test: True labels.

    Returns:
        float: mean deviation from perfect calibration diagonal.
    """
    y_proba  = model.predict_proba(X_test)
    y_bin    = label_binarize(y_test, classes=TARGET_NAMES)
    n_bins   = 10
    deviations = []

    for class_idx in range(len(TARGET_NAMES)):
        prob_pos = y_proba[:, class_idx]
        y_binary = y_bin[:, class_idx]

        try:
            frac_pos, mean_pred = calibration_curve(
                y_binary, prob_pos, n_bins=n_bins
            )
            # Mean absolute deviation from the diagonal (perfect calibration)
            dev = np.mean(np.abs(frac_pos - mean_pred))
            deviations.append(dev)
        except ValueError:
            # Happens when a bin has no samples — skip that bin safely
            pass

    return float(np.mean(deviations)) if deviations else 1.0


# ── Experiment logging ────────────────────────────────────────────────────────

def log_experiment(log_entry: dict, log_file: str = LOG_FILE) -> None:
    """
    Append an experiment log entry to a CSV file.
    Creates the file with headers on first run; appends thereafter.

    Args:
        log_entry: Dict containing all experiment metadata and metrics.
        log_file : Path to the CSV log file.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    file_exists = os.path.isfile(log_file)

    with open(log_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(log_entry.keys()))

        if not file_exists:
            writer.writeheader()   # Write column headers on first run

        writer.writerow(log_entry)

    print(f"[Evaluator] Experiment logged to: {log_file}")
