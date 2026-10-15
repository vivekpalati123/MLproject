"""
predict.py - Batch prediction pipeline for uploaded CSV files.
"""
import os
import numpy as np
import pandas as pd
from src.utils import (
    get_logger, normalize_columns, normalize_labels,
    replace_infinities, align_features,
)

logger = get_logger("predict")


def validate_prediction_input(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Validate incoming DataFrame for prediction.
    Returns a validation report dict.
    """
    df = normalize_columns(df)
    present = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in feature_cols and c != "Label"]

    report = {
        "n_rows": len(df),
        "features_present": len(present),
        "features_missing": len(missing),
        "missing_feature_names": missing[:20],
        "extra_columns": extra[:20],
        "pct_features_present": round(len(present) / max(len(feature_cols), 1) * 100, 1),
        "valid": len(present) > 0,
    }
    if len(missing) > len(feature_cols) * 0.5:
        report["warning"] = (
            f"More than 50% of required features are missing ({len(missing)} of {len(feature_cols)}). "
            "Predictions may be unreliable."
        )
    return report


def run_binary_prediction(
    df_input: pd.DataFrame,
    model,
    imputer,
    feature_cols: list,
    label_col: str = "Label",
) -> pd.DataFrame:
    """
    Run binary prediction on an uploaded DataFrame.
    Returns df with added Predicted_Label and Attack_Probability columns.
    """
    df = normalize_columns(df_input.copy())
    df = replace_infinities(df)

    # Drop label column if present
    if label_col in df.columns:
        df = df.drop(columns=[label_col])

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Align features
    df = align_features(df, feature_cols)

    # Impute
    X = pd.DataFrame(
        imputer.transform(df),
        columns=feature_cols,
    )

    y_pred = model.predict(X)
    result_df = df_input.copy()
    result_df["Predicted_Binary"] = y_pred
    result_df["Predicted_Label"] = ["ATTACK" if p == 1 else "BENIGN" for p in y_pred]

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[:, 1]
            result_df["Attack_Probability"] = proba.round(4)
        except Exception:
            pass

    return result_df


def run_multiclass_prediction(
    df_input: pd.DataFrame,
    model,
    label_encoder,
    imputer,
    feature_cols: list,
    label_col: str = "Label",
) -> pd.DataFrame:
    """
    Run multiclass prediction on an uploaded DataFrame.
    Returns df with added Predicted_Category column.
    """
    df = normalize_columns(df_input.copy())
    df = replace_infinities(df)

    if label_col in df.columns:
        df = df.drop(columns=[label_col])

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = align_features(df, feature_cols)

    X = pd.DataFrame(
        imputer.transform(df),
        columns=feature_cols,
    )

    y_pred_enc = model.predict(X)
    y_pred_labels = label_encoder.inverse_transform(y_pred_enc)

    result_df = df_input.copy()
    result_df["Predicted_Category"] = y_pred_labels
    return result_df
