"""Model Evaluation & Selection Module for Customer Churn Prediction.

Author: MSc AI & Data Analytics Intern
Project: Customer Churn Prediction Using Machine Learning
Dataset: IBM Telco Customer Churn

This module:
1. Computes comprehensive classification metrics:
   - Accuracy
   - Precision
   - Recall (Sensitivity)
   - F1-Score
   - ROC-AUC (Area Under ROC Curve)
   - Confusion Matrix (TP, TN, FP, FN)
2. Compares Logistic Regression, Random Forest, and XGBoost in an academic evaluation table.
3. Generates and saves publication-quality evaluation figures:
   - Confusion matrix subplots for each model
   - Multi-model ROC Curve comparison
   - Side-by-side metric comparison bar charts
   - Feature importance bar chart for top churn drivers
4. Formulates the business and analytical rationale for prioritizing Recall over naive Accuracy.
5. Programmatically selects the best-performing model based on balanced criteria (Recall, ROC-AUC, F1, Accuracy)
   and serializes the pipeline to models/best_model.pkl.
"""

import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def evaluate_single_model(
    model_name: str,
    pipeline: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Compute all evaluation metrics for a single fitted pipeline.

    Args:
        model_name: Name of the model.
        pipeline: Fitted sklearn Pipeline.
        X_test: Test features.
        y_test: Ground-truth test binary labels.

    Returns:
        dict containing scalar metrics, predictions, probabilities, and confusion matrix.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "Confusion_Matrix": cm,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def compare_models(
    fitted_pipelines: dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate and compare all candidate models.

    Args:
        fitted_pipelines: dict of fitted pipelines.
        X_test: Test feature matrix.
        y_test: Test true labels.

    Returns:
        tuple: (comparison_df, evaluation_results_dict)
    """
    results = {}
    summary_rows = []

    print(f"\n{'='*75}\n[MODEL EVALUATION ON UNSEEN TEST SET]\n{'='*75}")
    for name, pipeline in fitted_pipelines.items():
        res = evaluate_single_model(name, pipeline, X_test, y_test)
        results[name] = res
        summary_rows.append({
            "Model": name,
            "Accuracy": res["Accuracy"],
            "Precision": res["Precision"],
            "Recall": res["Recall"],
            "F1-Score": res["F1-Score"],
            "ROC-AUC": res["ROC-AUC"],
        })

    comparison_df = pd.DataFrame(summary_rows).set_index("Model")
    # Sort primarily by ROC-AUC and Recall
    comparison_df["Composite_Score"] = (
        0.35 * comparison_df["Recall"]
        + 0.35 * comparison_df["ROC-AUC"]
        + 0.20 * comparison_df["F1-Score"]
        + 0.10 * comparison_df["Accuracy"]
    )
    comparison_df.sort_values(by="Composite_Score", ascending=False, inplace=True)

    print(comparison_df.to_string(formatters={
        "Accuracy": "{:.4f}".format,
        "Precision": "{:.4f}".format,
        "Recall": "{:.4f}".format,
        "F1-Score": "{:.4f}".format,
        "ROC-AUC": "{:.4f}".format,
        "Composite_Score": "{:.4f}".format,
    }))
    print(f"{'='*75}")

    return comparison_df, results


def explain_recall_importance() -> None:
    """Print academic & industry rationale for prioritizing Recall over Accuracy."""
    text = """
================================================================================
BUSINESS & STATISTICAL RATIONALE: WHY RECALL MATTERS IN CHURN PREDICTION
================================================================================
In commercial subscription systems, class distribution is inherently imbalanced
(~26% Churn vs. ~74% Retained). Relying on naive Accuracy is deceptive:
a dummy model predicting 'No Churn' for all customers achieves ~74% accuracy,
yet captures 0% of churners!

Evaluation Metric Trade-off:
- False Positive (FP): Model flags a loyal customer as churn-risk.
  Cost: Minor outreach expense (e.g. email, proactive customer support, 10% loyalty discount).
- False Negative (FN): Model fails to flag an actual churner.
  Cost: Loss of the customer permanently! Customer Acquisition Cost (CAC) is 5x to 7x
  higher than Customer Retention Cost. Lost Customer Lifetime Value (CLV) is substantial.

Therefore, RECALL (Sensitivity = TP / [TP + FN]) directly quantifies how many
at-risk customers the business successfully catches before they cancel.
A higher Recall minimizes catastrophic False Negatives. Combining Recall with ROC-AUC
ensures strong probability discrimination across all operating thresholds.
================================================================================
"""
    print(text)


def plot_confusion_matrices(results: dict, y_test: pd.Series) -> None:
    """Plot confusion matrices for all evaluated models side by side."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for idx, (name, res) in enumerate(results.items()):
        cm = res["Confusion_Matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            xticklabels=["Stayed (0)", "Churned (1)"],
            yticklabels=["Stayed (0)", "Churned (1)"],
        )
        axes[idx].set_title(
            f"{name}\nRecall: {res['Recall']:.2%} | Acc: {res['Accuracy']:.2%}",
            fontsize=12,
            fontweight="bold",
        )
        axes[idx].set_xlabel("Predicted Label", fontsize=11)
        axes[idx].set_ylabel("True Label", fontsize=11)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] {plot_path}")


def plot_roc_curves(results: dict, y_test: pd.Series) -> None:
    """Plot combined ROC Curves for all evaluated models."""
    plt.figure(figsize=(9, 7))
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    for idx, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {res['ROC-AUC']:.3f})",
            color=colors[idx % len(colors)],
            linewidth=2.5,
        )

    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.500)", linewidth=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, "roc_curves_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] {plot_path}")


def plot_model_metrics_comparison(comparison_df: pd.DataFrame) -> None:
    """Plot comparison bar chart across Accuracy, Precision, Recall, F1, and ROC-AUC."""
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    plot_df = comparison_df[metrics_to_plot].reset_index()
    plot_melted = pd.melt(plot_df, id_vars="Model", var_name="Metric", value_name="Score")

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=plot_melted,
        x="Metric",
        y="Score",
        hue="Model",
        palette="tab10",
    )
    plt.title("Model Performance Benchmark Across Core Metrics", fontsize=14, fontweight="bold")
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
    plt.xlabel("Evaluation Metric", fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(title="Algorithm", loc="lower right", fontsize=10)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height + 0.02),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "model_metrics_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] {plot_path}")


def plot_feature_importance(best_name: str, best_pipeline: object) -> None:
    """Extract and visualize feature importance from the best model."""
    from src.train_models import get_feature_names_from_preprocessor

    preprocessor = best_pipeline.named_steps["preprocessor"]
    classifier = best_pipeline.named_steps["classifier"]
    feature_names = get_feature_names_from_preprocessor(preprocessor)

    importances = None
    metric_label = "Feature Importance"

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        metric_label = "Gini / Gain Importance"
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
        metric_label = "Absolute Logistic Coefficient |β|"

    if importances is not None and len(importances) == len(feature_names):
        feat_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values(by="Importance", ascending=False).head(15)

        plt.figure(figsize=(10, 7))
        sns.barplot(
            data=feat_df,
            x="Importance",
            y="Feature",
            palette="viridis",
            hue="Feature",
            legend=False,
        )
        plt.title(f"Top 15 Churn Predictive Features ({best_name})", fontsize=14, fontweight="bold")
        plt.xlabel(metric_label, fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.tight_layout()

        plot_path = os.path.join(PLOTS_DIR, "feature_importance.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[SAVED] {plot_path}")


def select_and_save_best_model(
    comparison_df: pd.DataFrame,
    fitted_pipelines: dict[str, object],
) -> tuple[str, object]:
    """Select the best model based on balanced criteria and serialize to disk."""
    best_name = comparison_df.index[0]
    best_pipeline = fitted_pipelines[best_name]

    save_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump({
        "model_name": best_name,
        "pipeline": best_pipeline,
        "comparison_metrics": comparison_df.loc[best_name].to_dict(),
    }, save_path)

    print(f"\n[SELECTION] Best model identified: '{best_name}'")
    print(f"            Recall:   {comparison_df.loc[best_name, 'Recall']:.4f}")
    print(f"            ROC-AUC:  {comparison_df.loc[best_name, 'ROC-AUC']:.4f}")
    print(f"            F1-Score: {comparison_df.loc[best_name, 'F1-Score']:.4f}")
    print(f"            Accuracy: {comparison_df.loc[best_name, 'Accuracy']:.4f}")
    print(f"[SAVED] Full serialized pipeline saved to: {save_path}")

    return best_name, best_pipeline


def run_full_evaluation(fitted_pipelines: dict, X_test: pd.DataFrame, y_test: pd.Series):
    """Execute complete evaluation workflow, plot generation, and best model persistence."""
    explain_recall_importance()
    comparison_df, results = compare_models(fitted_pipelines, X_test, y_test)

    # Plot evaluation diagnostics
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    plot_model_metrics_comparison(comparison_df)

    # Best model selection & persistence
    best_name, best_pipeline = select_and_save_best_model(comparison_df, fitted_pipelines)
    plot_feature_importance(best_name, best_pipeline)

    return comparison_df, best_name, best_pipeline


if __name__ == "__main__":
    from src.data_preprocessing import get_preprocessed_data
    from src.train_models import train_all_models

    print("[RUNNING] Commencing End-to-End Model Training & Evaluation Pipeline...")
    X_train, X_test, y_train, y_test, preprocessor, df_clean = get_preprocessed_data()
    fitted_pipelines = train_all_models(X_train, y_train)
    comparison_df, best_name, best_pipeline = run_full_evaluation(fitted_pipelines, X_test, y_test)
    print("\n[COMPLETED] Pipeline execution finished successfully.")
