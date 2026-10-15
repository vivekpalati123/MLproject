"""
train_binary.py - Train and save the binary intrusion-detection model.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from src.utils import get_logger, ensure_dirs, save_json

logger = get_logger("train_binary")


def train_binary_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> RandomForestClassifier:
    """Train the RandomForestClassifier binary model."""
    params = config["binary_model"]["params"]
    logger.info("Training binary RandomForestClassifier with params: %s", params)
    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)
    logger.info("Binary model training complete.")
    return clf


def save_binary_artifacts(
    model,
    imputer: SimpleImputer,
    feature_cols: list,
    split_info: dict,
    config: dict,
) -> None:
    """Save model, imputer, feature list, and metadata."""
    models_dir = config["artifacts"]["models_dir"]
    ensure_dirs(models_dir)

    joblib.dump(model, os.path.join(models_dir, "binary_model.joblib"))
    joblib.dump(imputer, os.path.join(models_dir, "binary_imputer.joblib"))

    metadata = {
        "model_type": "RandomForestClassifier",
        "task": "binary",
        "label_mapping": {"0": "BENIGN", "1": "ATTACK"},
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "split_info": split_info,
        "params": config["binary_model"]["params"],
    }
    save_json(metadata, os.path.join(models_dir, "binary_metadata.json"))
    logger.info("Binary model artifacts saved to %s", models_dir)


def load_binary_artifacts(config: dict) -> tuple:
    """
    Load binary model, imputer, and metadata.
    Returns: (model, imputer, metadata) or raises FileNotFoundError.
    """
    models_dir = config["artifacts"]["models_dir"]
    model_path = os.path.join(models_dir, "binary_model.joblib")
    imputer_path = os.path.join(models_dir, "binary_imputer.joblib")
    meta_path = os.path.join(models_dir, "binary_metadata.json")

    for p in [model_path, imputer_path, meta_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Binary model artifact not found: {p}")

    model = joblib.load(model_path)
    imputer = joblib.load(imputer_path)
    from src.utils import load_json
    metadata = load_json(meta_path)
    logger.info("Binary model loaded. Features: %d", metadata["n_features"])
    return model, imputer, metadata
