# 🌸 DecodeLabs AI — Project 2: Data Classification Using AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![DecodeLabs](https://img.shields.io/badge/DecodeLabs-Batch%202026-purple)

> **Industrial Training Kit | Batch 2026 | Powered by DecodeLabs**

---

## 👤 Author
**Ali Ahmad** — AI Engineering Intern, DecodeLabs Batch 2026
GitHub: [whozahm3d](https://github.com/whozahm3d)

---

## 📄 Abstract

This project implements a complete, production-grade supervised learning pipeline
for multi-class data classification using the Iris benchmark dataset. It covers
every stage of the ML engineering lifecycle — from raw data ingestion with
source fallback logic, through exploratory analysis, preprocessing, model
training, cross-validation, statistical significance testing, and explainability
— culminating in a persisted sklearn Pipeline and a real-time interactive
prediction interface.

The project goes significantly beyond the DecodeLabs Project 2 specification.
Rather than training a single KNN classifier, it trains and rigorously compares
four algorithms (KNN, Decision Tree, SVM, Logistic Regression) plus a soft-voting
ensemble, evaluates each across five metrics, runs stratified k-fold
cross-validation with paired t-test significance testing, and applies SHAP
explainability to the best model. All code is modular, documented, and organized
as a deployable Python package.

As a portfolio artifact, this notebook demonstrates mastery of the full
supervised learning workflow — data validation, feature scaling without leakage,
hyperparameter tuning, domain-aware evaluation, and experiment reproducibility —
skills that transfer directly to production ML engineering roles.

---

## 📋 Table of Contents

- [Project Architecture](#-project-architecture)
- [Dataset](#-dataset)
- [Results](#-results)
- [Visualizations](#-visualizations)
- [Reproducibility](#-reproducibility)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Known Limitations](#-known-limitations)
- [Future Work](#-future-work)
- [License](#-license)

---

## 🏗️ Project Architecture

Raw Data (HuggingFace primary / sklearn fallback)
↓
SHA-256 Data Hash (versioning integrity)
↓
Data Validation (shape · nulls · classes · duplicates)
↓
Exploratory Data Analysis
(distributions · correlations · outliers · pairplot · violin)
↓
Dimensionality Reduction (PCA + t-SNE — separability check)
↓
Preprocessing
StandardScaler — fit on X_train ONLY → transform X_train + X_test
↓
Train / Test Split (80/20 · stratified · shuffled · seed=42)
↓
┌──────────────────────────────────────────────────────┐
│  KNN (optimal k)  │  Decision Tree  │  SVM  │  LR   │
│                     + VotingClassifier (soft)        │
└──────────────────────────────────────────────────────┘
↓
Stratified K-Fold Cross-Validation (5 folds · f1_macro)
↓
Paired t-Test — statistical significance between top 2 models
↓
GridSearchCV — hyperparameter tuning on best model
↓
Full Evaluation
(Accuracy · F1 · Precision · Recall · ROC-AUC per model)
↓
Visualizations (14 plot types → assets/)
↓
SHAP Explainability + Calibration + Learning Curves
↓
Class Imbalance Demonstration (accuracy mirage proof)
↓
sklearn Pipeline Object (scaler + classifier chained)
↓
Model Persistence (joblib) + Experiment Log (CSV)
↓
Interactive Prediction Interface (ipywidgets · confidence threshold)
↓
Requirements Freeze (exact environment snapshot)

---

## 📊 Dataset

| Property | Value |
|---|---|
| Name | Iris Benchmark |
| Source (primary) | HuggingFace `scikit-learn/iris` |
| Source (fallback) | `sklearn.datasets.load_iris` |
| Samples | 150 (balanced — 50 per class) |
| Classes | 3 — Setosa, Versicolor, Virginica |
| Features | 4 — sepal length, sepal width, petal length, petal width |
| Missing Values | None |
| Split | 80% train / 20% test (stratified) |

---

## 📈 Results

### Model Comparison (Test Set)

| Model | Accuracy | F1 (macro) | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| KNN (optimal k) | ~0.967 | ~0.967 | ~0.968 | ~0.967 | ~0.997 |
| Decision Tree | ~0.933 | ~0.933 | ~0.935 | ~0.933 | ~0.978 |
| SVM (RBF) | ~0.967 | ~0.967 | ~0.968 | ~0.967 | ~0.998 |
| Logistic Regression | ~0.967 | ~0.967 | ~0.968 | ~0.967 | ~0.997 |
| Voting Ensemble | ~0.967 | ~0.967 | ~0.968 | ~0.967 | ~0.998 |

### Cross-Validation Results (5-Fold Stratified · F1 macro)

| Model | Mean F1 | Std F1 |
|---|---|---|
| KNN (optimal k) | ~0.960 | ±0.030 |
| Decision Tree | ~0.940 | ±0.040 |
| SVM (RBF) | ~0.973 | ±0.025 |
| Logistic Regression | ~0.960 | ±0.030 |
| Voting Ensemble | ~0.967 | ±0.025 |

### Statistical Significance
Paired t-test between best and second-best model.
If p < 0.05: difference is statistically significant.
Winner declared with p-value printed in Cell 32.

---

## 🖼️ Visualizations

| Plot | Description |
|---|---|
| Class Distribution | Bar chart of sample count per species |
| Feature KDE | Kernel density + rug plots per feature |
| Correlation Heatmap | Pearson r matrix across all 4 features |
| Pairplot | All feature combinations colored by species |
| Boxplots | IQR bounds with outlier markers per feature |
| Violin Plots | Distribution shape + IQR per feature per class |
| PCA + t-SNE | 2D projections showing natural class separability |
| Elbow Curve | KNN error rate vs k with optimal k marked |
| CV Bar Chart | Mean ± std F1 per model with error bars |
| Model Comparison | Grouped bars — 5 metrics × 5 models |
| Confusion Matrices | Annotated heatmap grid per model |
| ROC Curves | Per-class curves with AUC for best model |
| Feature Importance | Permutation importance horizontal bars per model |
| Learning Curve | Train vs validation score with overfitting diagnosis |

---

## 🔁 Reproducibility

```bash
git clone https://github.com/whozahm3d/decodelabs-ai-project2.git
cd decodelabs-ai-project2
pip install -r requirements.txt
jupyter notebook notebooks/ai_classification_project2.ipynb
# Kernel → Restart & Run All
```

All randomness seeded with `RANDOM_STATE = 42`.
Data hash logged per run to `logs/experiment_log.csv`.

---

## 📁 Project Structure

```
decodelabs-ai-project2/
│
├── notebooks/
│   └── ai_classification_project2.ipynb   ← 44-cell main notebook
│
├── pipeline/                               ← Modular Python package
│   ├── __init__.py
│   ├── data_loader.py                      ← HuggingFace + sklearn fallback, SHA-256
│   ├── preprocessor.py                     ← Split, scale, IQR outlier detection
│   ├── trainer.py                          ← 4 base models + Ensemble + Pipeline
│   ├── evaluator.py                        ← Metrics, CV, t-test, weighted F1, log
│   └── visualizer.py                       ← All 11 plot functions → assets/
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py                    ← 22 unit tests (pytest)
│
├── models/                                 ← Saved artifacts (generated at runtime)
│   ├── best_model_pipeline.pkl
│   └── model_card.txt
│
├── assets/                                 ← All plots (generated at runtime)
├── logs/                                   ← Experiment CSV log (generated at runtime)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone / extract the project
```bash
unzip decodelabs-ai-project2.zip
cd decodelabs-ai-project2
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run unit tests
```bash
pytest tests/test_pipeline.py -v
```
All 22 tests should pass with `PASSED` status.

### 5. Launch the notebook
```bash
cd notebooks
jupyter notebook ai_classification_project2.ipynb
```
Run cells top-to-bottom (`Kernel → Restart & Run All`).

---

## 📊 Pipeline Architecture

```
INPUT                  PROCESS                      OUTPUT
─────                  ───────                      ──────
Iris Dataset     →     Train-Test Split (80/20)  →  Confusion Matrix
(HuggingFace /         StandardScaler             →  F1 Score
 sklearn)              KNN Elbow Method           →  ROC-AUC Curves
                       KNN / DT / SVM / LR        →  SHAP Beeswarm
                       Soft Voting Ensemble        →  Model Card
                       5-Fold CV + t-test         →  Experiment Log
```

---

## 🤖 Models Trained

| Model | Key Parameters | Role |
|-------|---------------|------|
| **KNN** | k = optimal (elbow) | Primary classifier |
| **Decision Tree** | max_depth=5 | Interpretable baseline |
| **SVM** | RBF kernel, probability=True | Kernel-based |
| **Logistic Regression** | max_iter=1000 | Linear baseline |
| **Voting Ensemble** | soft voting (all 4) | Best-of-all |

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Fraction of correct predictions |
| F1 (macro) | Harmonic mean of Precision & Recall |
| Precision | Quality of positive predictions |
| Recall | Coverage of actual positives |
| ROC-AUC (OvR) | Discrimination across all thresholds |
| Weighted F1 | Domain-aware (Versicolor/Virginica penalty ×2) |
| Calibration | Mean deviation from perfect probability calibration |

---

## 🧪 Running Tests

```bash
pytest tests/test_pipeline.py -v --tb=short
```

**22 tests across 4 sections:**
- `TestDataLoader` (5 tests) — shape, columns, nulls, balance, hash
- `TestPreprocessor` (6 tests) — splits, stratification, leakage, scaling
- `TestTrainer` (6 tests) — types, accuracy, keys, pipeline structure
- `TestEvaluator` (5 tests) — metric keys, ranges, CV length, calibration

---

## 🔮 Explainability

SHAP `KernelExplainer` is used for model-agnostic feature attribution.  
Background: 20 training samples | Evaluation: full test set (50 nsamples per call).

> ⏳ Expect ~30–60s runtime for SHAP cell on standard hardware.

---

## 💾 Artifacts

After running the notebook:

| Artifact | Location | Description |
|----------|----------|-------------|
| `best_model_pipeline.pkl` | `models/` | Scaler + best model (joblib) |
| `model_card.txt` | `models/` | Standardised model summary |
| `experiment_log.csv` | `logs/` | Append-only run history |
| `*.png` | `assets/` | All 14+ saved visualisations |

---

## 📚 Concepts Demonstrated

- Supervised Learning (classification)
- Stratified train/test split & anti-leakage scaling
- Hyperparameter tuning (elbow method)
- Multi-model comparison and ensemble methods
- Evaluation beyond accuracy (F1, ROC-AUC)
- Statistical significance testing (paired t-test)
- Explainability (SHAP KernelExplainer)
- Model calibration analysis
- Production-ready sklearn Pipeline
- Unit testing with pytest

---

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| scikit-learn | ≥1.4.0 | ML models · metrics · pipeline |
| pandas | ≥2.0.0 | Data manipulation |
| numpy | ≥1.26.0 | Numerical computing |
| matplotlib | ≥3.8.0 | Base plotting |
| seaborn | ≥0.13.0 | Statistical visualizations |
| datasets (HuggingFace) | ≥2.18.0 | Primary data source |
| scipy | ≥1.12.0 | Statistical significance testing |
| shap | ≥0.44.0 | Model explainability |
| joblib | ≥1.3.0 | Model serialization |
| ipywidgets | ≥8.0.0 | Interactive prediction UI |
| pytest | ≥8.0.0 | Unit testing |
| Jupyter Notebook | ≥7.0.0 | Development environment |

---

## ⚠️ Known Limitations

- **KNN does not scale** — inference is O(n × d) at prediction time. Unsuitable for datasets beyond ~100k rows without approximation methods like FAISS.
- **Iris is a benchmark, not a real problem** — near-perfect class separability (especially Setosa) is not representative of real-world noise, missing values, or class overlap.
- **No Bayesian hyperparameter search** — GridSearchCV covers a fixed param grid. Production pipelines would use Optuna or Ray Tune for smarter search.
- **Model not validated on external data** — all evaluation is in-distribution. A held-out external dataset would be needed to confirm generalization.

---

## 🚀 Future Work

- **Project 3 — Deep Learning & CNNs** — move from tabular classification to image classification using convolutional neural networks on a real image dataset.
- **Scalable inference** — replace KNN with a neural network classifier and serve via FastAPI REST endpoint for production deployment.
- **Proper experiment tracking** — integrate MLflow or Weights & Biases to replace the manual CSV logger with a full experiment management system.
- **AutoML comparison** — benchmark the manual pipeline against TPOT or H2O AutoML to quantify the value of human-guided feature engineering.

---

## 🏢 About DecodeLabs

DecodeLabs is an industrial AI training organisation based in Greater Lucknow, India.  
📞 +91 89330 06408 | ✉ decodelabs.tech@gmail.com | 🌐 www.decodelabs.tech

---

*Built with ❤️ as part of DecodeLabs AI Internship — Batch 2026*
