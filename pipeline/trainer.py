"""
pipeline/trainer.py
DecodeLabs AI Project 2 — Data Classification
Author: Ali Ahmad | DecodeLabs Batch 2026

One function per model, train_all_models(), build_pipeline(),
and build_voting_classifier().
"""

import numpy as np
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.tree          import DecisionTreeClassifier
from sklearn.svm           import SVC
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import VotingClassifier
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


# ── Individual model trainers ────────────────────────────────────────────────

def train_knn(X_train: np.ndarray, y_train,
              n_neighbors: int = 5) -> KNeighborsClassifier:
    """
    Train a K-Nearest Neighbours classifier.

    Args:
        X_train    : Scaled training feature matrix.
        y_train    : Training labels.
        n_neighbors: Number of neighbours (default 5).

    Returns:
        Fitted KNeighborsClassifier.
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    print(f"[Trainer] KNN (k={n_neighbors}) trained.")
    return model


def train_decision_tree(X_train: np.ndarray, y_train,
                        max_depth: int = 5) -> DecisionTreeClassifier:
    """
    Train a Decision Tree classifier.

    Args:
        X_train  : Scaled training feature matrix.
        y_train  : Training labels.
        max_depth: Maximum tree depth (default 5).

    Returns:
        Fitted DecisionTreeClassifier.
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    print(f"[Trainer] Decision Tree (max_depth={max_depth}) trained.")
    return model


def train_svm(X_train: np.ndarray, y_train) -> SVC:
    """
    Train a Support Vector Machine classifier with RBF kernel.
    probability=True is mandatory for ROC, calibration, SHAP, and confidence.

    Args:
        X_train: Scaled training feature matrix.
        y_train: Training labels.

    Returns:
        Fitted SVC.
    """
    model = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    print("[Trainer] SVM (RBF kernel, probability=True) trained.")
    return model


def train_logistic_regression(X_train: np.ndarray, y_train) -> LogisticRegression:
    """
    Train a Logistic Regression classifier.

    Args:
        X_train: Scaled training feature matrix.
        y_train: Training labels.

    Returns:
        Fitted LogisticRegression.
    """
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    print("[Trainer] Logistic Regression (max_iter=1000) trained.")
    return model


# ── Composite trainers ───────────────────────────────────────────────────────

def train_all_models(X_train: np.ndarray, y_train,
                     optimal_k: int = 5) -> dict:
    """
    Train all 4 base models and the soft-voting ensemble.

    Args:
        X_train  : Scaled training feature matrix.
        y_train  : Training labels.
        optimal_k: Optimal k for KNN (from elbow method).

    Returns:
        dict: {model_name: fitted_model} for all 5 models.
            Keys: 'KNN', 'Decision Tree', 'SVM',
                  'Logistic Regression', 'Ensemble'
    """
    base_models = {
        'KNN':                 train_knn(X_train, y_train, n_neighbors=optimal_k),
        'Decision Tree':       train_decision_tree(X_train, y_train),
        'SVM':                 train_svm(X_train, y_train),
        'Logistic Regression': train_logistic_regression(X_train, y_train),
    }

    # Build and fit the ensemble
    voting_clf = build_voting_classifier(optimal_k=optimal_k)
    voting_clf.fit(X_train, y_train)
    print("[Trainer] Soft Voting Ensemble trained.")

    models = {**base_models, 'Ensemble': voting_clf}
    print(f"[Trainer] All {len(models)} models trained successfully.")
    return models


def build_voting_classifier(optimal_k: int = 5) -> VotingClassifier:
    """
    Build an unfitted soft-voting VotingClassifier from all 4 base estimators.

    Args:
        optimal_k: Optimal k for KNN estimator within ensemble.

    Returns:
        Unfitted VotingClassifier with soft voting strategy.
    """
    voting_clf = VotingClassifier(
        estimators=[
            ('knn', KNeighborsClassifier(n_neighbors=optimal_k)),
            ('dt',  DecisionTreeClassifier(max_depth=5,
                                           random_state=RANDOM_STATE)),
            ('svm', SVC(kernel='rbf', probability=True,
                        random_state=RANDOM_STATE)),
            ('lr',  LogisticRegression(max_iter=1000,
                                       random_state=RANDOM_STATE)),
        ],
        voting='soft'   # soft voting uses predicted probabilities
    )
    return voting_clf


def build_pipeline(best_model_unfitted) -> Pipeline:
    """
    Build a sklearn Pipeline that chains StandardScaler with a classifier.
    The pipeline operates on raw (unscaled) data.

    Args:
        best_model_unfitted: An unfitted sklearn estimator (e.g. KNN, SVM).

    Returns:
        Unfitted sklearn Pipeline([('scaler', StandardScaler()),
                                    ('classifier', best_model_unfitted)])

    Raises:
        TypeError: if best_model_unfitted is already fitted (has coef_ or
                   n_samples_seen_ attribute — heuristic check).
    """
    pipeline = Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', best_model_unfitted),
    ])
    return pipeline
