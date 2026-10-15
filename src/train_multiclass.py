"""
train_multiclass.py - Train and save the multiclass attack-classification model.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from src.utils import get_logger, ensure_dirs, save_json

logger = get_logger("train_multiclass")


def train_multiclass_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> tuple:
    """
    Train ExtraTreesClassifier multiclass model.
    Returns: (model, label_encoder)
    """
    # Remove rare classes (< 2 samples in train)
    class_counts = y_train.value_counts()
    rare = class_counts[class_counts < 2].index.tolist()
    if rare:
        logger.warning("Removing rare classes from multiclass training: %s", rare)
        mask = ~y_train.isin(rare)
        X_train = X_train[mask]
        y_train = y_train[mask]

    if y_train.nunique() < 2:
        raise ValueError("Not enough classes for multiclass training.")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)

    params = config["multiclass_model"]["params"]
    logger.info("Training multiclass ExtraTreesClassifier with params: %s", params)
    clf = ExtraTreesClassifier(**params)
    clf.fit(X_train, y_encoded)
    logger.info("Multiclass model training complete. Classes: %s", list(le.classes_))
    return clf, le


def save_multiclass_artifacts(
    model,
    label_encoder: LabelEncoder,
    imputer: SimpleImputer,
    feature_cols: list,
    config: dict,
) -> None:
    """Save multiclass model, encoder, imputer, and metadata."""
    models_dir = config["artifacts"]["models_dir"]
    ensure_dirs(models_dir)

    joblib.dump(model, os.path.join(models_dir, "multiclass_model.joblib"))
    joblib.dump(label_encoder, os.path.join(models_dir, "multiclass_encoder.joblib"))
    joblib.dump(imputer, os.path.join(models_dir, "multiclass_imputer.joblib"))

    metadata = {
        "model_type": "ExtraTreesClassifier",
        "task": "multiclass",
        "classes": list(label_encoder.classes_),
        "n_classes": len(label_encoder.classes_),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "params": config["multiclass_model"]["params"],
    }
    save_json(metadata, os.path.join(models_dir, "multiclass_metadata.json"))
    logger.info("Multiclass artifacts saved to %s", models_dir)


def load_multiclass_artifacts(config: dict) -> tuple:
    """
    Load multiclass model, encoder, imputer, and metadata.
    Returns: (model, label_encoder, imputer, metadata)
    """
    models_dir = config["artifacts"]["models_dir"]
    model_path = os.path.join(models_dir, "multiclass_model.joblib")
    encoder_path = os.path.join(models_dir, "multiclass_encoder.joblib")
    imputer_path = os.path.join(models_dir, "multiclass_imputer.joblib")
    meta_path = os.path.join(models_dir, "multiclass_metadata.json")

    for p in [model_path, encoder_path, imputer_path, meta_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Multiclass artifact not found: {p}")

    model = joblib.load(model_path)
    le = joblib.load(encoder_path)
    imputer = joblib.load(imputer_path)
    from src.utils import load_json
    metadata = load_json(meta_path)
    return model, le, imputer, metadata
