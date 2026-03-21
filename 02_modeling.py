"""
Mental-Health Polypharmacy DDI Prediction
==========================================
NOTEBOOK 2 — ML Modeling, Evaluation & Comparison

Loads the prepared mental-health subset from 01_data_engineering.py,
runs EDA, trains three models, evaluates with clinical-context metrics,
and generates comparison plots.

Author: tarajeeclarke
Course: HADM 283 - AI in Healthcare, Hofstra University
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)
import pickle
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────

def load_subset(filepath: str = "twosides_mental_health_subset.csv") -> pd.DataFrame:
    """Load the prepared mental-health DDI subset."""
    print(f"Loading subset: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame):
    """Visualize class distribution and PRR statistics."""
    print("\n" + "=" * 50)
    print("Exploratory Data Analysis")
    print("=" * 50)

    print("\nPRR Summary Statistics:")
    print(df["PRR_float"].describe())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("EDA — Mental-Health DDI Subset", fontsize=14, fontweight="bold")

    # Class distribution
    label_counts = df["label"].value_counts().sort_index()
    axes[0].bar(
        ["No Signal (0)", "Signal (1)"],
        label_counts.values,
        color=["#4c72b0", "#dd8452"],
        edgecolor="white",
    )
    axes[0].set_title("Label Distribution (Class Imbalance)")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(label_counts.values):
        axes[0].text(i, v + 20, f"{v:,}", ha="center", fontsize=11)

    # PRR distribution (log scale for clarity)
    axes[1].hist(
        df["PRR_float"].clip(upper=20),
        bins=50,
        color="#55a868",
        edgecolor="white",
        alpha=0.85,
    )
    axes[1].axvline(x=1.0, color="red", linestyle="--", linewidth=1.5, label="PRR = 1 threshold")
    axes[1].set_title("PRR Distribution (clipped at 20)")
    axes[1].set_xlabel("PRR Value")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("eda_mental_health_ddi.png", dpi=150)
    plt.show()
    print("EDA plot saved.")


# ─────────────────────────────────────────────────────────────
# 3. PREPROCESSING & SPLIT
# ─────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame):
    """
    Select features (drug name pair), create stratified train/test split.
    Returns X_train, X_test, y_train, y_test.
    """
    features = ["drug_1_concept_name", "drug_2_concept_name"]
    target   = "label"

    df_model = df[features + [target]].dropna()
    X = df_model[features]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Train label distribution:\n{y_train.value_counts()}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────────────────────

def build_pipeline(classifier) -> Pipeline:
    """Wrap OneHotEncoder + classifier in a sklearn Pipeline."""
    return Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ("model",   classifier),
    ])


def train_models(X_train, y_train) -> dict:
    """Train all three models and return fitted pipelines."""
    models = {
        "LR Baseline":         build_pipeline(LogisticRegression(max_iter=500, random_state=42)),
        "LR Balanced":         build_pipeline(LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)),
        "Random Forest Balanced": build_pipeline(RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)),
    }

    for name, pipeline in models.items():
        print(f"\nTraining: {name}...")
        pipeline.fit(X_train, y_train)
        print(f"  Done.")

    return models


# ─────────────────────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_models(models: dict, X_test, y_test):
    """
    Evaluate all models. Print classification reports, plot confusion
    matrices, plot combined ROC curve, and return summary table.
    """
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    results = []
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

    for name, pipeline in models.items():
        y_pred  = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(f"\n── {name} ──")
        print(f"  Accuracy: {acc:.4f} | ROC AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["No Signal", "Signal"]))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["No Signal", "Signal"]).plot(
            cmap="Blues", ax=ax_cm
        )
        ax_cm.set_title(f"Confusion Matrix — {name}")
        plt.tight_layout()
        plt.savefig(f"cm_{name.lower().replace(' ', '_')}.png", dpi=150)
        plt.close(fig_cm)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

        results.append({"Model": name, "Accuracy": round(acc, 4), "ROC AUC": round(auc, 4)})

    # ROC plot
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax_roc.set_title("ROC Curves — All Models")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(alpha=0.3)
    plt.tight_layout()
    fig_roc.savefig("roc_curves.png", dpi=150)
    plt.show()
    print("ROC curve plot saved.")

    summary = pd.DataFrame(results)
    print("\n── Model Comparison Summary ──")
    print(summary.to_string(index=False))

    return summary


# ─────────────────────────────────────────────────────────────
# 6. ERROR ANALYSIS
# ─────────────────────────────────────────────────────────────

def error_analysis(best_pipeline, X_test: pd.DataFrame, y_test: pd.Series, n: int = 10):
    """
    Inspect false positives and false negatives from the best model.
    Shows PRR context for misclassified cases.
    """
    print("\n" + "=" * 50)
    print("Error Analysis — Best Model (LR Balanced)")
    print("=" * 50)

    y_pred = best_pipeline.predict(X_test)
    results = X_test.copy()
    results["true_label"]      = y_test.values
    results["predicted_label"] = y_pred

    fp = results[(results["true_label"] == 0) & (results["predicted_label"] == 1)]
    fn = results[(results["true_label"] == 1) & (results["predicted_label"] == 0)]

    print(f"\nFalse Positives (predicted Signal, actually No Signal): {len(fp)}")
    print(fp.head(n).to_string(index=False))

    print(f"\nFalse Negatives (predicted No Signal, actually Signal): {len(fn)}")
    print(fn.head(n).to_string(index=False))


# ─────────────────────────────────────────────────────────────
# 7. SAVE BEST MODEL
# ─────────────────────────────────────────────────────────────

def save_model(pipeline, name: str = "lr_balanced_pipeline.pkl"):
    """Serialize the best model pipeline to disk."""
    with open(name, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved: {name}")


# ─────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Mental-Health DDI — ML Modeling Pipeline")
    print("=" * 60)

    import os
    if not os.path.exists("twosides_mental_health_subset.csv"):
        print(
            "\n[ERROR] 'twosides_mental_health_subset.csv' not found.\n"
            "Run 01_data_engineering.py first to generate this file."
        )
        return

    df                          = load_subset()
    run_eda(df)
    X_train, X_test, y_train, y_test = prepare_features(df)
    models                      = train_models(X_train, y_train)
    summary                     = evaluate_models(models, X_test, y_test)
    error_analysis(models["LR Balanced"], X_test, y_test)
    save_model(models["LR Balanced"])

    print("\nModeling pipeline complete. All plots saved to working directory.")


if __name__ == "__main__":
    main()
