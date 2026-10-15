"""
eda.py - Exploratory Data Analysis module.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import get_logger, ensure_dirs, replace_infinities

logger = get_logger("eda")


def run_eda(df: pd.DataFrame, config: dict) -> dict:
    """
    Generate EDA plots and statistics from the cleaned DataFrame.
    Returns a dict with figure paths and key stats.
    """
    out_dir = config["outputs"]["eda_dir"]
    ensure_dirs(out_dir)
    label_col = config["data"]["label_column"]
    benign_label = config["data"]["benign_label"]

    df = replace_infinities(df)
    results = {"plots": {}, "stats": {}}

    if label_col not in df.columns:
        logger.warning("Label column not found; skipping EDA.")
        return results

    # ── 1: Benign vs Attack count bar ──────────────────────────────────────
    binary = df[label_col].apply(lambda x: "Benign" if x == benign_label else "Attack")
    counts = binary.value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", width=0.5)
    for bar, cnt in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{cnt:,}", ha="center", va="bottom", fontsize=11)
    ax.set_title("Benign vs Attack Traffic", fontsize=14, fontweight="bold")
    ax.set_ylabel("Sample Count", fontsize=12)
    plt.tight_layout()
    p = os.path.join(out_dir, "benign_vs_attack.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    results["plots"]["benign_vs_attack"] = p

    # ── 2: Attack family counts ─────────────────────────────────────────────
    attack_df = df[df[label_col] != benign_label]
    if not attack_df.empty:
        atk_counts = attack_df[label_col].value_counts()
        fig, ax = plt.subplots(figsize=(12, 5))
        palette = plt.cm.tab10(np.linspace(0, 1, len(atk_counts)))
        bars = ax.bar(atk_counts.index, atk_counts.values, color=palette, edgecolor="white")
        for bar, cnt in zip(bars, atk_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{cnt:,}", ha="center", va="bottom", fontsize=8)
        ax.set_title("Attack Family Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Attack Type")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        p = os.path.join(out_dir, "attack_families.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        results["plots"]["attack_families"] = p

    # ── 3: Class imbalance bar ─────────────────────────────────────────────
    class_counts = df[label_col].value_counts()
    fig, ax = plt.subplots(figsize=(12, 5))
    palette2 = ["#2ecc71" if l == benign_label else "#e74c3c" for l in class_counts.index]
    ax.bar(class_counts.index, class_counts.values, color=palette2, edgecolor="white")
    ax.set_title("Class Imbalance (All Labels)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    p = os.path.join(out_dir, "class_imbalance.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    results["plots"]["class_imbalance"] = p

    # ── 4: Feature distributions (top 12 numeric by variance) ─────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != label_col]
    if numeric_cols:
        variances = df[numeric_cols].var().sort_values(ascending=False)
        top_cols = variances.head(12).index.tolist()
        n = len(top_cols)
        cols_per_row = 4
        rows = (n + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(16, rows * 3))
        axes = axes.flatten() if n > 1 else [axes]
        for i, col in enumerate(top_cols):
            try:
                df[col].dropna().astype(float).plot(
                    kind="hist", bins=50, ax=axes[i],
                    color="#3498db", edgecolor="white", alpha=0.8,
                )
                axes[i].set_title(col[:22], fontsize=8)
                axes[i].set_xlabel("")
            except Exception:
                pass
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("Feature Distributions (Top 12 by Variance)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        p = os.path.join(out_dir, "feature_distributions.png")
        fig.savefig(p, dpi=100)
        plt.close(fig)
        results["plots"]["feature_distributions"] = p

    # ── 5: Correlation heatmap ─────────────────────────────────────────────
    if len(numeric_cols) >= 2:
        top_var_cols = variances.head(20).index.tolist() if numeric_cols else []
        if len(top_var_cols) >= 2:
            corr = df[top_var_cols].corr()
            fig, ax = plt.subplots(figsize=(13, 10))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(
                corr, mask=mask, cmap="coolwarm", center=0,
                annot=False, linewidths=0.3, ax=ax,
                cbar_kws={"shrink": 0.8},
            )
            ax.set_title("Feature Correlation Heatmap (Top 20 by Variance)", fontsize=13, fontweight="bold")
            plt.tight_layout()
            p = os.path.join(out_dir, "correlation_heatmap.png")
            fig.savefig(p, dpi=100)
            plt.close(fig)
            results["plots"]["correlation_heatmap"] = p

    # ── Key stats ──────────────────────────────────────────────────────────
    results["stats"] = {
        "total_rows": int(len(df)),
        "benign_count": int(binary.value_counts().get("Benign", 0)),
        "attack_count": int(binary.value_counts().get("Attack", 0)),
        "n_attack_types": int(len(attack_df[label_col].unique())) if not attack_df.empty else 0,
        "n_features": len(numeric_cols),
    }

    logger.info("EDA complete. Plots saved to %s", out_dir)
    return results
