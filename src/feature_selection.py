"""
feature_selection.py - Select top features using model importance.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from src.utils import get_logger, ensure_dirs, save_json

logger = get_logger("feature_selection")


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
    top_n: int = 40,
) -> tuple:
    """
    Fit a quick ExtraTreesClassifier to rank features.
    Returns: (selected_feature_list, importance_df)
    """
    out_dir = config["outputs"]["feature_importance_dir"]
    ensure_dirs(out_dir)

    logger.info("Running feature selection with ExtraTrees...")
    clf = ExtraTreesClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)

    importances = clf.feature_importances_
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    importance_df.to_csv(os.path.join(out_dir, "feature_importances.csv"), index=False)

    top_features = importance_df.head(top_n)["feature"].tolist()
    save_json(
        {"top_features": top_features, "top_n": top_n},
        os.path.join(out_dir, "selected_features.json"),
    )

    # ── Plot top 20 ────────────────────────────────────────────────────────
    plot_df = importance_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(plot_df)))[::-1]
    ax.barh(plot_df["feature"][::-1], plot_df["importance"][::-1], color=colors)
    ax.set_title("Top 20 Feature Importances", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    p = os.path.join(out_dir, "feature_importance_plot.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)

    logger.info("Feature selection done. Top %d features selected.", top_n)
    return top_features, importance_df
