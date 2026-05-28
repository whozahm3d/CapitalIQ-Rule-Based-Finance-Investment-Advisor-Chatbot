"""
pipeline/visualizer.py
DecodeLabs AI Project 2 — Data Classification
Author: Ali Ahmad | DecodeLabs Batch 2026

All plotting functions used throughout the notebook.
Every function saves its figure to assets/ and optionally shows it inline.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.metrics        import confusion_matrix, roc_curve, auc
from sklearn.preprocessing  import label_binarize
from sklearn.calibration    import calibration_curve
from sklearn.tree           import plot_tree

TARGET_NAMES = ['setosa', 'versicolor', 'virginica']
PALETTE      = {'setosa': '#2196F3', 'versicolor': '#FF9800', 'virginica': '#4CAF50'}
ASSETS_DIR   = os.path.join(os.getcwd(), 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Shared helper ─────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str) -> str:
    """Save figure to assets/ at 150 dpi and return the full path."""
    path = os.path.join(ASSETS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[Visualizer] Saved: {path}")
    return path


# ── 1. EDA ────────────────────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame) -> str:
    """
    Bar chart of sample counts per Iris species.

    Args:
        df: Clean DataFrame with 'species' column.

    Returns:
        str: Path to saved figure.
    """
    counts = df['species'].value_counts().reindex(TARGET_NAMES)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[s] for s in counts.index], width=0.5)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title('Class Distribution — Iris Dataset', fontsize=14, fontweight='bold')
    ax.set_xlabel('Species', fontsize=12)
    ax.set_ylabel('Sample Count', fontsize=12)
    ax.set_ylim(0, counts.max() + 10)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return _save(fig, 'class_distribution.png')


def plot_feature_distributions(df: pd.DataFrame) -> str:
    """
    4-panel KDE + rug plot of each feature, coloured by species.

    Args:
        df: Clean DataFrame with FEATURE_NAMES + 'species'.

    Returns:
        str: Path to saved figure.
    """
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    labels   = ['Sepal Length (cm)', 'Sepal Width (cm)',
                 'Petal Length (cm)', 'Petal Width (cm)']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, feat, label in zip(axes, features, labels):
        for species in TARGET_NAMES:
            data = df[df['species'] == species][feat]
            sns.kdeplot(data, ax=ax, label=species,
                        color=PALETTE[species], fill=True, alpha=0.3)
            ax.plot(data, np.zeros_like(data) - 0.02,
                    '|', color=PALETTE[species], alpha=0.5)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('Feature Distributions by Species', fontsize=14, fontweight='bold')
    fig.tight_layout()
    return _save(fig, 'feature_distributions.png')


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """
    Annotated Pearson correlation heatmap of the 4 features.

    Args:
        df: Clean DataFrame.

    Returns:
        str: Path to saved figure.
    """
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    corr     = df[features].corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, ax=ax,
                cbar_kws={'shrink': 0.8},
                linewidths=0.5)
    ax.set_title('Pearson Correlation Heatmap', fontsize=13, fontweight='bold')
    ax.set_xticklabels(['Sepal L', 'Sepal W', 'Petal L', 'Petal W'], rotation=30)
    ax.set_yticklabels(['Sepal L', 'Sepal W', 'Petal L', 'Petal W'], rotation=0)
    fig.tight_layout()
    return _save(fig, 'correlation_heatmap.png')


def plot_pairplot(df: pd.DataFrame) -> str:
    """
    Seaborn pairplot of all 4 features coloured by species.
    Diagonal: KDE. Off-diagonal: scatter.

    Args:
        df: Clean DataFrame.

    Returns:
        str: Path to saved figure.
    """
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    g = sns.pairplot(
        df[features + ['species']],
        hue='species',
        palette=PALETTE,
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 30},
        corner=False
    )
    g.figure.suptitle('Pairplot — All Feature Combinations',
                       y=1.02, fontsize=13, fontweight='bold')
    return _save(g.figure, 'pairplot.png')


def plot_boxplots(df: pd.DataFrame) -> str:
    """
    4-panel box plots of each feature grouped by species.
    Overlaid strip plots show individual data points.

    Args:
        df: Clean DataFrame.

    Returns:
        str: Path to saved figure.
    """
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    labels   = ['Sepal Length (cm)', 'Sepal Width (cm)',
                 'Petal Length (cm)', 'Petal Width (cm)']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, feat, label in zip(axes, features, labels):
        sns.boxplot(data=df, x='species', y=feat,
                    palette=PALETTE, order=TARGET_NAMES,
                    width=0.5, linewidth=1.2, ax=ax)
        sns.stripplot(data=df, x='species', y=feat,
                      palette=PALETTE, order=TARGET_NAMES,
                      jitter=True, alpha=0.4, size=3, ax=ax)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('Feature Box Plots by Species (with IQR outlier reference)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    return _save(fig, 'boxplots.png')


def plot_violin(df: pd.DataFrame) -> str:
    """
    4-panel violin plots showing distribution shape per species.

    Args:
        df: Clean DataFrame.

    Returns:
        str: Path to saved figure.
    """
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    labels   = ['Sepal Length (cm)', 'Sepal Width (cm)',
                 'Petal Length (cm)', 'Petal Width (cm)']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, feat, label in zip(axes, features, labels):
        sns.violinplot(data=df, x='species', y=feat,
                       palette=PALETTE, order=TARGET_NAMES,
                       inner='box', linewidth=1, ax=ax)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('Violin Plots — Distribution Shape by Species',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    return _save(fig, 'violin_plots.png')


# ── 2. Preprocessing ─────────────────────────────────────────────────────────

def plot_scaling_comparison(X_train: pd.DataFrame,
                             X_train_scaled: np.ndarray) -> str:
    """
    Side-by-side box plots of raw vs StandardScaler-scaled features.

    Args:
        X_train       : Unscaled training features (pd.DataFrame).
        X_train_scaled: Scaled training features (np.ndarray).

    Returns:
        str: Path to saved figure.
    """
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    labels   = ['Sepal L', 'Sepal W', 'Petal L', 'Petal W']

    raw_df    = pd.DataFrame(X_train.values,     columns=features)
    scaled_df = pd.DataFrame(X_train_scaled,     columns=features)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    raw_df.boxplot(ax=axes[0], rot=20)
    axes[0].set_title('Raw Features (Unscaled)', fontsize=12, fontweight='bold')
    axes[0].set_xticklabels(labels)

    scaled_df.boxplot(ax=axes[1], rot=20)
    axes[1].set_title('Standard Scaled Features (μ=0, σ=1)',
                      fontsize=12, fontweight='bold')
    axes[1].set_xticklabels(labels)

    fig.suptitle('Feature Scaling Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    return _save(fig, 'scaling_comparison.png')


# ── 3. KNN Elbow ─────────────────────────────────────────────────────────────

def plot_knn_elbow(k_values: list, error_rates: list,
                   optimal_k: int) -> str:
    """
    KNN elbow curve — error rate vs k with optimal k annotated.

    Args:
        k_values   : List of k values tested.
        error_rates: Corresponding mean CV error rates.
        optimal_k  : The selected optimal k (highlighted in red).

    Returns:
        str: Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k_values, error_rates, 'o-', color='#1565C0',
            linewidth=2, markersize=6, label='CV Error Rate')
    ax.axvline(x=optimal_k, color='#E53935', linestyle='--',
               linewidth=1.5, label=f'Optimal k = {optimal_k}')
    ax.scatter([optimal_k], [error_rates[k_values.index(optimal_k)]],
               color='#E53935', s=120, zorder=5)

    ax.annotate(f'k = {optimal_k}',
                xy=(optimal_k, error_rates[k_values.index(optimal_k)]),
                xytext=(optimal_k + 1, error_rates[k_values.index(optimal_k)] + 0.005),
                fontsize=10, color='#E53935',
                arrowprops=dict(arrowstyle='->', color='#E53935'))

    ax.set_title('KNN Elbow Curve — Choosing Optimal k', fontsize=13, fontweight='bold')
    ax.set_xlabel('Number of Neighbours (k)', fontsize=11)
    ax.set_ylabel('CV Error Rate (1 − F1 macro)', fontsize=11)
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return _save(fig, 'knn_elbow.png')


# ── 4. Model Comparison ───────────────────────────────────────────────────────

def plot_model_comparison(results_df: pd.DataFrame) -> str:
    """
    Grouped bar chart comparing all models across 5 metrics.

    Args:
        results_df: DataFrame with models as index and metric columns.

    Returns:
        str: Path to saved figure.
    """
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    n_models  = len(results_df)
    n_metrics = len(metrics)
    x         = np.arange(n_models)
    width     = 0.15
    colors    = ['#1E88E5', '#43A047', '#FB8C00', '#E53935', '#8E24AA']

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - n_metrics / 2) * width + width / 2
        bars = ax.bar(x + offset, results_df[metric], width,
                      label=metric.replace('_', ' ').title(),
                      color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(results_df.index, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Model Performance Comparison — All Metrics',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return _save(fig, 'model_comparison.png')


def plot_cv_boxplot(cv_scores: dict) -> str:
    """
    Box plot of cross-validation F1 scores across folds for each model.

    Args:
        cv_scores: Dict {model_name: np.ndarray of fold scores}.

    Returns:
        str: Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    data   = [scores for scores in cv_scores.values()]
    labels = list(cv_scores.keys())

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2))

    colors = ['#1E88E5', '#43A047', '#FB8C00', '#E53935', '#8E24AA']
    for patch, color in zip(bp['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('F1 Score (macro)', fontsize=11)
    ax.set_title('Cross-Validation Score Distribution per Model',
                 fontsize=13, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return _save(fig, 'cv_boxplot.png')


# ── 5. Confusion Matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(model, X_test: np.ndarray,
                           y_test, model_name: str) -> str:
    """
    Annotated confusion matrix heatmap with per-cell percentages.

    Args:
        model      : Fitted estimator.
        X_test     : Scaled test features.
        y_test     : True labels.
        model_name : Display name for title and filename.

    Returns:
        str: Path to saved figure.
    """
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred, labels=TARGET_NAMES)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=TARGET_NAMES,
                yticklabels=TARGET_NAMES,
                linewidths=0.5, ax=ax, cbar=False)

    # Overlay raw counts + percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j + 0.5, i + 0.4, str(cm[i, j]),
                    ha='center', va='center', fontsize=13,
                    fontweight='bold', color=color)
            ax.text(j + 0.5, i + 0.65, f'({cm_pct[i,j]:.1f}%)',
                    ha='center', va='center', fontsize=8, color=color)

    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title(f'Confusion Matrix — {model_name}',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    safe_name = model_name.lower().replace(' ', '_')
    return _save(fig, f'confusion_matrix_{safe_name}.png')


# ── 6. ROC Curve ─────────────────────────────────────────────────────────────

def plot_roc_curves(model, X_test: np.ndarray,
                    y_test, model_name: str) -> str:
    """
    One-vs-Rest multiclass ROC curves (one line per class) with AUC in legend.

    Args:
        model      : Fitted estimator with predict_proba().
        X_test     : Scaled test features.
        y_test     : True labels.
        model_name : Display name.

    Returns:
        str: Path to saved figure.
    """
    y_proba = model.predict_proba(X_test)
    y_bin   = label_binarize(y_test, classes=TARGET_NAMES)
    colors  = ['#1E88E5', '#FB8C00', '#43A047']

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (cls, color) in enumerate(zip(TARGET_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc_val     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{cls} (AUC = {auc_val:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.6, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'ROC Curves (OvR) — {model_name}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    safe_name = model_name.lower().replace(' ', '_')
    return _save(fig, f'roc_curve_{safe_name}.png')


# ── 7. Decision Tree ─────────────────────────────────────────────────────────

def plot_decision_tree(dt_model, feature_names: list,
                       class_names: list) -> str:
    """
    Render the fitted Decision Tree as a matplotlib figure.

    Args:
        dt_model     : Fitted DecisionTreeClassifier.
        feature_names: List of feature name strings.
        class_names  : List of class name strings.

    Returns:
        str: Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(dt_model,
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              fontsize=9,
              ax=ax)
    ax.set_title('Decision Tree — Full Structure',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    return _save(fig, 'decision_tree.png')


# ── 8. Feature Importance ────────────────────────────────────────────────────

def plot_feature_importance(dt_model, feature_names: list) -> str:
    """
    Horizontal bar chart of Decision Tree Gini feature importances.

    Args:
        dt_model     : Fitted DecisionTreeClassifier.
        feature_names: List of feature name strings.

    Returns:
        str: Path to saved figure.
    """
    importances = dt_model.feature_importances_
    sorted_idx  = np.argsort(importances)
    colors      = ['#1565C0', '#1E88E5', '#42A5F5', '#90CAF9']

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        [feature_names[i] for i in sorted_idx],
        importances[sorted_idx],
        color=[colors[i] for i in sorted_idx]
    )

    for bar, val in zip(bars, importances[sorted_idx]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=10)

    ax.set_xlabel('Gini Importance', fontsize=11)
    ax.set_title('Decision Tree — Feature Importances',
                 fontsize=13, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return _save(fig, 'feature_importance.png')


# ── 9. Calibration ────────────────────────────────────────────────────────────

def plot_calibration_curves(models: dict,
                             X_test: np.ndarray, y_test) -> str:
    """
    Reliability diagrams (calibration curves) for all models.

    Args:
        models: Dict {model_name: fitted_estimator}.
        X_test: Scaled test features.
        y_test: True labels.

    Returns:
        str: Path to saved figure.
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    colors = ['#1E88E5', '#43A047', '#FB8C00', '#E53935', '#8E24AA']
    y_bin  = label_binarize(y_test, classes=TARGET_NAMES)

    for ax, (name, model), color in zip(axes, models.items(), colors):
        y_proba = model.predict_proba(X_test)
        for class_idx, cls in enumerate(TARGET_NAMES):
            try:
                frac, pred = calibration_curve(
                    y_bin[:, class_idx],
                    y_proba[:, class_idx],
                    n_bins=8
                )
                ax.plot(pred, frac, 's-', label=cls,
                        color=list(PALETTE.values())[class_idx], linewidth=1.5)
            except ValueError:
                pass

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect')
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Mean Predicted Probability', fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel('Fraction of Positives', fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle('Calibration Curves (Reliability Diagrams)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    return _save(fig, 'calibration_curves.png')


# ── 10. SHAP ─────────────────────────────────────────────────────────────────

def plot_shap_summary(shap_values, X_test_scaled: np.ndarray,
                      feature_names: list, model_name: str) -> str:
    """
    SHAP beeswarm summary plot for the best model.

    Args:
        shap_values  : SHAP values array from KernelExplainer
                       shape (n_classes, n_samples, n_features) or
                       (n_samples, n_features).
        X_test_scaled: Scaled test feature array.
        feature_names: Feature name list.
        model_name   : Display name for title.

    Returns:
        str: Path to saved figure.
    """
    import shap

    fig, axes = plt.subplots(1, len(TARGET_NAMES),
                              figsize=(6 * len(TARGET_NAMES), 5))

    for i, (cls, ax) in enumerate(zip(TARGET_NAMES, axes)):
        plt.sca(ax)
        sv = shap_values[i] if isinstance(shap_values, list) else shap_values[:, :, i]
        shap.summary_plot(
            sv,
            X_test_scaled,
            feature_names=feature_names,
            show=False,
            plot_type='dot',
            color_bar=False
        )
        ax.set_title(f'SHAP — {cls}', fontsize=10, fontweight='bold')

    fig.suptitle(f'SHAP Feature Importance — {model_name}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    return _save(fig, f'shap_summary_{model_name.lower().replace(" ","_")}.png')


# ── 11. Learning Curve ────────────────────────────────────────────────────────

def plot_learning_curve(train_sizes: np.ndarray,
                         train_scores: np.ndarray,
                         val_scores: np.ndarray,
                         model_name: str) -> str:
    """
    Learning curve: training and validation F1 vs training set size.
    Shaded regions show ±1 std across CV folds.

    Args:
        train_sizes : Array of training set sizes used.
        train_scores: 2D array (n_sizes × n_folds) of train scores.
        val_scores  : 2D array (n_sizes × n_folds) of validation scores.
        model_name  : Display name.

    Returns:
        str: Path to saved figure.
    """
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes, train_mean, 'o-', color='#1E88E5',
            linewidth=2, label='Training Score')
    ax.fill_between(train_sizes,
                    train_mean - train_std,
                    train_mean + train_std,
                    color='#1E88E5', alpha=0.15)

    ax.plot(train_sizes, val_mean, 's-', color='#E53935',
            linewidth=2, label='Validation Score')
    ax.fill_between(train_sizes,
                    val_mean - val_std,
                    val_mean + val_std,
                    color='#E53935', alpha=0.15)

    ax.set_xlabel('Training Set Size', fontsize=11)
    ax.set_ylabel('F1 Score (macro)', fontsize=11)
    ax.set_title(f'Learning Curve — {model_name}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    safe_name = model_name.lower().replace(' ', '_')
    return _save(fig, f'learning_curve_{safe_name}.png')
