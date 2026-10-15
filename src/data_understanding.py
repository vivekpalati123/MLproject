"""
data_understanding.py - Compute and save dataset statistics and visualisations.
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import (
    get_logger, load_config, ensure_dirs, normalize_columns,
    normalize_labels, replace_infinities, save_json, NpEncoder,
)

warnings.filterwarnings("ignore")
logger = get_logger("data_understanding")


def load_raw_data(csv_files: list, sample_size: int = None) -> pd.DataFrame:
    """Load and concatenate CSV files. Optionally sample rows."""
    frames = []
    for f in csv_files:
        try:
            chunk = pd.read_csv(f, low_memory=False)
            chunk = normalize_columns(chunk)
            frames.append(chunk)
            logger.info("Loaded %s rows from %s", len(chunk), f)
        except Exception as e:
            logger.error("Failed to read %s: %s", f, e)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    return df


def compute_understanding(df: pd.DataFrame, config: dict) -> dict:
    """Compute all data-understanding statistics."""
    label_col = config["data"]["label_column"]
    benign_label = config["data"]["benign_label"]

    # Normalise labels
    if label_col in df.columns:
        df[label_col] = normalize_labels(df[label_col])

    df_inf = replace_infinities(df)
    numeric_cols = df_inf.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_inf.select_dtypes(exclude=[np.number]).columns.tolist()

    # Class distribution
    class_dist = {}
    benign_count = 0
    attack_count = 0
    attack_dist = {}
    if label_col in df.columns:
        class_dist = df[label_col].value_counts().to_dict()
        benign_count = int(class_dist.get(benign_label, 0))
        attack_count = int(sum(v for k, v in class_dist.items() if k != benign_label))
        attack_dist = {k: int(v) for k, v in class_dist.items() if k != benign_label}

    # Missing values
    missing = df_inf.isnull().sum()
    missing_dict = missing[missing > 0].to_dict()

    # Infinite values (original df)
    numeric_df = df.select_dtypes(include=[np.number])
    inf_count = int(np.isinf(numeric_df.values).sum())

    # Duplicates
    dup_count = int(df.duplicated().sum())

    # Constant features
    constant_cols = [c for c in numeric_cols if df_inf[c].nunique() <= 1]

    # Descriptive stats
    desc = df_inf[numeric_cols].describe().round(4).to_dict() if numeric_cols else {}

    report = {
        "shape": list(df.shape),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols,
        "n_numeric": len(numeric_cols),
        "n_categorical": len(cat_cols),
        "label_column": label_col,
        "class_distribution": {str(k): int(v) for k, v in class_dist.items()},
        "benign_count": benign_count,
        "attack_count": attack_count,
        "attack_distribution": attack_dist,
        "missing_values": {str(k): int(v) for k, v in missing_dict.items()},
        "total_missing": int(missing.sum()),
        "duplicate_rows": dup_count,
        "infinite_values": inf_count,
        "constant_features": constant_cols,
        "descriptive_stats": desc,
    }
    return report, df


def save_understanding_outputs(report: dict, df: pd.DataFrame, config: dict) -> None:
    """Save report and visualisations."""
    out_dir = config["outputs"]["data_understanding_dir"]
    ensure_dirs(out_dir)

    # JSON report
    save_json(report, os.path.join(out_dir, "data_understanding_report.json"))

    # CSV summary
    summary_rows = [
        {"metric": "Total Rows", "value": report["n_rows"]},
        {"metric": "Total Columns", "value": report["n_cols"]},
        {"metric": "Numeric Features", "value": report["n_numeric"]},
        {"metric": "Categorical Features", "value": report["n_categorical"]},
        {"metric": "Total Missing Values", "value": report["total_missing"]},
        {"metric": "Duplicate Rows", "value": report["duplicate_rows"]},
        {"metric": "Infinite Values", "value": report["infinite_values"]},
        {"metric": "Benign Rows", "value": report["benign_count"]},
        {"metric": "Attack Rows", "value": report["attack_count"]},
    ]
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(out_dir, "dataset_summary.csv"), index=False
    )

    # Class distribution CSV
    if report["class_distribution"]:
        pd.DataFrame(
            list(report["class_distribution"].items()), columns=["Label", "Count"]
        ).to_csv(os.path.join(out_dir, "class_distribution.csv"), index=False)

    label_col = config["data"]["label_column"]
    benign_label = config["data"]["benign_label"]

    # ── Plot 1: Label distribution bar ─────────────────────────────────────
    if report["class_distribution"]:
        fig, ax = plt.subplots(figsize=(12, 5))
        labels = list(report["class_distribution"].keys())
        counts = list(report["class_distribution"].values())
        colors = ["#2ecc71" if l == benign_label else "#e74c3c" for l in labels]
        bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title("Label Distribution", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Label", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        for bar, cnt in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{cnt:,}",
                ha="center", va="bottom", fontsize=8,
            )
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "label_distribution.png"), dpi=120)
        plt.close(fig)

    # ── Plot 2: Benign vs Attack pie ────────────────────────────────────────
    if report["benign_count"] > 0 or report["attack_count"] > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        vals = [report["benign_count"], report["attack_count"]]
        lbls = ["Benign", "Attack"]
        colors_pie = ["#2ecc71", "#e74c3c"]
        wedges, texts, autotexts = ax.pie(
            vals, labels=lbls, colors=colors_pie,
            autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for at in autotexts:
            at.set_fontsize(10)
        ax.set_title("Benign vs Attack Traffic", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "benign_vs_attack.png"), dpi=120)
        plt.close(fig)

    # ── Plot 3: Attack category bar ─────────────────────────────────────────
    if report["attack_distribution"]:
        fig, ax = plt.subplots(figsize=(12, 5))
        atk_labels = list(report["attack_distribution"].keys())
        atk_counts = list(report["attack_distribution"].values())
        ax.bar(atk_labels, atk_counts, color="#e74c3c", edgecolor="white")
        ax.set_title("Attack Category Distribution", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Attack Type", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "attack_distribution.png"), dpi=120)
        plt.close(fig)

    # ── Plot 4: Missing values ──────────────────────────────────────────────
    if report["missing_values"]:
        fig, ax = plt.subplots(figsize=(10, 4))
        mv_cols = list(report["missing_values"].keys())[:20]
        mv_vals = [report["missing_values"][c] for c in mv_cols]
        ax.barh(mv_cols, mv_vals, color="#3498db")
        ax.set_title("Top Missing Values per Column", fontsize=13, fontweight="bold")
        ax.set_xlabel("Missing Count")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "missing_values.png"), dpi=120)
        plt.close(fig)

    # ── Plot 5: Numeric feature distributions (first 12) ───────────────────
    numeric_cols = report["numeric_cols"]
    if numeric_cols:
        sample_cols = [c for c in numeric_cols if c != label_col][:12]
        df_plot = replace_infinities(df)
        n = len(sample_cols)
        if n > 0:
            cols_per_row = 4
            rows = (n + cols_per_row - 1) // cols_per_row
            fig, axes = plt.subplots(rows, cols_per_row, figsize=(16, rows * 3))
            axes = axes.flatten() if n > 1 else [axes]
            for i, col in enumerate(sample_cols):
                try:
                    df_plot[col].dropna().astype(float).plot(
                        kind="hist", bins=40, ax=axes[i],
                        color="#3498db", edgecolor="white", alpha=0.8,
                    )
                    axes[i].set_title(col[:25], fontsize=8)
                    axes[i].set_xlabel("")
                except Exception:
                    pass
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            plt.suptitle("Numeric Feature Distributions (Sample)", fontsize=13, fontweight="bold")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "feature_distributions.png"), dpi=100)
            plt.close(fig)

    # ── Plot 6: Correlation heatmap ─────────────────────────────────────────
    if len(numeric_cols) >= 2:
        df_corr = replace_infinities(df)
        corr_cols = [c for c in numeric_cols if c != label_col][:25]
        corr_matrix = df_corr[corr_cols].dropna().corr()
        fig, ax = plt.subplots(figsize=(14, 11))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, mask=mask, cmap="coolwarm", center=0,
            annot=False, fmt=".2f", linewidths=0.3,
            ax=ax, cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Feature Correlation Heatmap (Top 25 Numeric)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "correlation_heatmap.png"), dpi=100)
        plt.close(fig)

    logger.info("Data-understanding outputs saved to %s", out_dir)


def run_data_understanding(csv_files: list, config: dict, sample_size: int = 200000) -> dict:
    """Full data-understanding pipeline. Returns the report dict."""
    df = load_raw_data(csv_files, sample_size=sample_size)
    if df.empty:
        logger.error("No data loaded.")
        return {}
    report, df = compute_understanding(df, config)
    save_understanding_outputs(report, df, config)
    logger.info("Data understanding complete. Shape: %s", report["shape"])
    return report
