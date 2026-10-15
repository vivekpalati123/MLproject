"""
feature_engineering.py - Feature engineering and selection pipeline.
"""
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from src.utils import (
    get_logger, ensure_dirs, replace_infinities,
    make_binary_labels, save_json,
)

logger = get_logger("feature_engineering")

# Columns to always exclude
EXCLUDE_PATTERNS = [
    "flow id", "source ip", "destination ip", "src ip", "dst ip",
    "source port", "destination port", "src port", "dst port",
    "timestamp", "flow start", "flow end", "fid",
]


def get_feature_cols(df: pd.DataFrame, label_col: str) -> list:
    """Return numeric feature columns, excluding label and identifiers."""
    lower_map = {c.lower(): c for c in df.columns}
    exclude = set()
    for pat in EXCLUDE_PATTERNS:
        if pat in lower_map:
            exclude.add(lower_map[pat])
    exclude.add(label_col)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]
    return feature_cols


def build_features(df: pd.DataFrame, config: dict) -> tuple:
    """
    Build feature matrix X and target series y (binary and multiclass).
    Fits imputer on the full df (training use only).
    Returns: X_imputed (df), y_binary, y_multi, feature_cols, imputer
    """
    label_col = config["data"]["label_column"]
    df = replace_infinities(df)

    feature_cols = get_feature_cols(df, label_col)

    # Remove constant features
    non_const = [c for c in feature_cols if df[c].nunique(dropna=False) > 1]
    removed_const = [c for c in feature_cols if c not in non_const]
    if removed_const:
        logger.info("Removed constant features: %s", removed_const)
    feature_cols = non_const

    X = df[feature_cols].copy()

    # Impute missing values
    imputer = SimpleImputer(strategy=config["preprocessing"]["imputer_strategy"])
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=feature_cols,
        index=df.index,
    )

    # Targets
    y_multi = df[label_col].copy() if label_col in df.columns else None
    y_binary = make_binary_labels(y_multi, config["data"]["benign_label"]) if y_multi is not None else None

    logger.info("Feature engineering done. Features: %d, Rows: %d", len(feature_cols), len(X_imputed))
    return X_imputed, y_binary, y_multi, feature_cols, imputer


def save_feature_list(feature_cols: list, config: dict) -> None:
    """Save the ordered feature list to disk."""
    out_dir = config["outputs"]["feature_importance_dir"]
    ensure_dirs(out_dir)
    save_json({"features": feature_cols, "count": len(feature_cols)},
              os.path.join(out_dir, "feature_list.json"))
    pd.DataFrame({"feature": feature_cols}).to_csv(
        os.path.join(out_dir, "feature_list.csv"), index=False
    )
    logger.info("Feature list (%d features) saved.", len(feature_cols))
