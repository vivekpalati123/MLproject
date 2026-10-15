"""
evaluate.py - Model evaluation and visualisation.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,
)
from src.utils import get_logger, ensure_dirs, save_json

logger = get_logger("evaluate")


# ── Binary evaluation ──────────────────────────────────────────────────────
def evaluate_binary(model, X_test: pd.DataFrame, y_test: pd.Series, config: dict) -> dict:
    """Evaluate binary classifier on held-out test set."""
    out_dir = config["outputs"]["evaluation_dir"]
    ensure_dirs(out_dir)

    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            pass

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "confusion_matrix": cm.tolist(),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            metrics["pr_auc"] = float(average_precision_score(y_test, y_proba))
        except Exception:
            pass

    report_str = classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"], zero_division=0)
    metrics["classification_report"] = report_str

    save_json(metrics, os.path.join(out_dir, "binary_metrics.json"))
    pd.DataFrame([{k: v for k, v in metrics.items() if not isinstance(v, (list, str))}]).to_csv(
        os.path.join(out_dir, "binary_metrics.csv"), index=False
    )

    # ── Confusion matrix plot ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BENIGN", "ATTACK"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Binary Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(out_dir, "binary_confusion_matrix.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    metrics["cm_plot"] = p

    # ── ROC curve ─────────────────────────────────────────────────────────
    if y_proba is not None:
        try:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, color="#e74c3c", lw=2,
                    label=f"ROC AUC = {metrics.get('roc_auc', 0):.4f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve – Binary", fontsize=13, fontweight="bold")
            ax.legend()
            plt.tight_layout()
            roc_path = os.path.join(out_dir, "binary_roc_curve.png")
            fig.savefig(roc_path, dpi=120)
            plt.close(fig)
            metrics["roc_plot"] = roc_path
        except Exception as e:
            logger.warning("ROC curve failed: %s", e)

    logger.info("Binary evaluation: Accuracy=%.4f, F1=%.4f, Recall=%.4f",
                metrics["accuracy"], metrics["f1"], metrics["recall"])
    return metrics


# ── Multiclass evaluation ──────────────────────────────────────────────────
def evaluate_multiclass(model, label_encoder, X_test: pd.DataFrame, y_test: pd.Series, config: dict) -> dict:
    """Evaluate multiclass classifier."""
    out_dir = config["outputs"]["evaluation_dir"]
    ensure_dirs(out_dir)

    y_encoded = label_encoder.transform(y_test)
    y_pred_enc = model.predict(X_test)

    # Map back to original classes
    known_indices = np.arange(len(label_encoder.classes_))
    valid_mask = np.isin(y_pred_enc, known_indices)
    y_pred_enc_safe = y_pred_enc.copy()

    classes = list(label_encoder.classes_)
    y_pred_labels = label_encoder.inverse_transform(y_pred_enc_safe)
    y_true_labels = label_encoder.inverse_transform(y_encoded)

    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=classes)

    metrics = {
        "accuracy": float(accuracy_score(y_true_labels, y_pred_labels)),
        "macro_precision": float(precision_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)),
        "classes": classes,
        "confusion_matrix": cm.tolist(),
    }

    report_str = classification_report(y_true_labels, y_pred_labels, zero_division=0)
    metrics["classification_report"] = report_str

    save_json(metrics, os.path.join(out_dir, "multiclass_metrics.json"))
    pd.DataFrame([{k: v for k, v in metrics.items()
                   if not isinstance(v, (list, str))}]).to_csv(
        os.path.join(out_dir, "multiclass_metrics.csv"), index=False
    )

    # ── Multiclass confusion matrix ────────────────────────────────────────
    n_classes = len(classes)
    fig_size = max(8, n_classes * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        ax=ax, linewidths=0.3,
    )
    ax.set_title("Multiclass Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    p = os.path.join(out_dir, "multiclass_confusion_matrix.png")
    fig.savefig(p, dpi=100)
    plt.close(fig)
    metrics["cm_plot"] = p

    logger.info(
        "Multiclass evaluation: Accuracy=%.4f, Macro-F1=%.4f",
        metrics["accuracy"], metrics["macro_f1"],
    )
    return metrics


# ── Feature importance plot ────────────────────────────────────────────────
def plot_feature_importance(model, feature_cols: list, config: dict) -> str:
    """Plot and save top-20 feature importances."""
    out_dir = config["outputs"]["feature_importance_dir"]
    ensure_dirs(out_dir)

    if not hasattr(model, "feature_importances_"):
        return ""

    importances = model.feature_importances_
    df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df)))[::-1]
    ax.barh(df["feature"][::-1], df["importance"][::-1], color=colors)
    ax.set_title("Top 20 Feature Importances (Binary Model)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    path = os.path.join(out_dir, "feature_importance_model.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    df.to_csv(os.path.join(out_dir, "top20_features.csv"), index=False)
    return path
