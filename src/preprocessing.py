"""
preprocessing.py - Train/test split with stratification.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import get_logger, save_json

logger = get_logger("preprocessing")


def split_data(
    X: pd.DataFrame,
    y_binary: pd.Series,
    y_multi: pd.Series,
    config: dict,
) -> tuple:
    """
    Stratified train/test split.
    Returns: X_train, X_test, yb_train, yb_test, ym_train, ym_test
    """
    test_size = config["preprocessing"]["test_size"]
    random_state = config["preprocessing"]["random_state"]

    # Stratify on binary label (more stable)
    try:
        X_train, X_test, yb_train, yb_test, ym_train, ym_test = train_test_split(
            X, y_binary, y_multi,
            test_size=test_size,
            random_state=random_state,
            stratify=y_binary,
        )
    except ValueError as e:
        logger.warning("Stratified split failed (%s). Using random split.", e)
        X_train, X_test, yb_train, yb_test, ym_train, ym_test = train_test_split(
            X, y_binary, y_multi,
            test_size=test_size,
            random_state=random_state,
        )

    logger.info(
        "Split: train=%d, test=%d | Binary - train attacks=%d, test attacks=%d",
        len(X_train), len(X_test),
        int(yb_train.sum()), int(yb_test.sum()),
    )

    split_info = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": test_size,
        "random_state": random_state,
        "train_benign": int((yb_train == 0).sum()),
        "train_attack": int((yb_train == 1).sum()),
        "test_benign": int((yb_test == 0).sum()),
        "test_attack": int((yb_test == 1).sum()),
    }

    return X_train, X_test, yb_train, yb_test, ym_train, ym_test, split_info
