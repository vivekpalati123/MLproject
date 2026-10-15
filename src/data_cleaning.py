"""
data_cleaning.py - Clean and prepare the raw dataset.
"""
import os
import numpy as np
import pandas as pd
from src.utils import (
    get_logger, ensure_dirs, normalize_columns,
    normalize_labels, replace_infinities, save_json,
)

logger = get_logger("data_cleaning")

# Columns that are likely identifiers or leakage candidates
IDENTIFIER_CANDIDATES = [
    "flow id", "source ip", "destination ip", "src ip", "dst ip",
    "source port", "destination port", "src port", "dst port",
    "timestamp", "flow start", "flow end", "fid",
]


def clean_data(df: pd.DataFrame, config: dict) -> tuple:
    """
    Apply cleaning pipeline.
    Returns: (cleaned_df, cleaning_report_dict)
    """
    label_col = config["data"]["label_column"]
    report = {}
    step_log = []

    # Step 0 – initial state
    report["initial_shape"] = list(df.shape)
    step_log.append({"step": "Initial", "rows": df.shape[0], "cols": df.shape[1], "notes": ""})

    # Step 1 – Normalise column names
    df = normalize_columns(df)
    step_log.append({"step": "Normalize column names", "rows": df.shape[0], "cols": df.shape[1], "notes": "Stripped whitespace"})

    # Step 2 – Normalise labels
    if label_col in df.columns:
        df[label_col] = normalize_labels(df[label_col])
        step_log.append({"step": "Normalize labels", "rows": df.shape[0], "cols": df.shape[1], "notes": "Canonical label strings"})
    else:
        logger.warning("Label column '%s' not found.", label_col)

    # Step 3 – Drop fully empty columns
    empty_cols = [c for c in df.columns if df[c].isnull().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        step_log.append({"step": "Drop empty columns", "rows": df.shape[0], "cols": df.shape[1], "notes": f"Removed: {empty_cols}"})

    # Step 4 – Convert all non-label columns to numeric
    feature_cols = [c for c in df.columns if c != label_col]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    step_log.append({"step": "Convert to numeric", "rows": df.shape[0], "cols": df.shape[1], "notes": "Non-numeric → NaN"})

    # Step 5 – Replace infinities
    df = replace_infinities(df)
    step_log.append({"step": "Replace infinities", "rows": df.shape[0], "cols": df.shape[1], "notes": "+/-inf → NaN"})

    # Step 6 – Remove constant features
    feature_cols = [c for c in df.columns if c != label_col]
    const_cols = [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]
    if const_cols:
        df = df.drop(columns=const_cols)
        step_log.append({"step": "Drop constant features", "rows": df.shape[0], "cols": df.shape[1], "notes": f"Removed: {const_cols[:10]}"})

    # Step 7 – Identify and flag identifier columns (do not drop label)
    lower_cols = {c.lower(): c for c in df.columns}
    id_cols_found = [lower_cols[k] for k in IDENTIFIER_CANDIDATES if k in lower_cols and lower_cols[k] != label_col]
    report["identifier_columns"] = id_cols_found
    if id_cols_found:
        df = df.drop(columns=id_cols_found)
        step_log.append({"step": "Drop identifier columns", "rows": df.shape[0], "cols": df.shape[1], "notes": f"Removed: {id_cols_found}"})

    # Step 8 – Remove exact duplicates
    before_dup = df.shape[0]
    df = df.drop_duplicates()
    removed_dup = before_dup - df.shape[0]
    step_log.append({"step": "Remove duplicates", "rows": df.shape[0], "cols": df.shape[1], "notes": f"Removed {removed_dup} duplicates"})

    # Step 9 – Drop rows where label is NaN
    if label_col in df.columns:
        before = df.shape[0]
        df = df.dropna(subset=[label_col])
        step_log.append({"step": "Drop rows with null label", "rows": df.shape[0], "cols": df.shape[1], "notes": f"Removed {before - df.shape[0]} rows"})

    report["final_shape"] = list(df.shape)
    report["steps"] = step_log
    report["rows_removed"] = report["initial_shape"][0] - report["final_shape"][0]
    report["cols_removed"] = report["initial_shape"][1] - report["final_shape"][1]
    report["remaining_missing"] = int(df.isnull().sum().sum())
    report["label_column"] = label_col
    report["final_columns"] = list(df.columns)
    report["class_distribution"] = (
        df[label_col].value_counts().to_dict() if label_col in df.columns else {}
    )

    logger.info(
        "Cleaning done. Shape: %s → %s",
        report["initial_shape"], report["final_shape"]
    )
    return df, report


def save_cleaned_data(df: pd.DataFrame, report: dict, config: dict) -> str:
    """Save cleaned DataFrame and cleaning report."""
    proc_dir = config["data"]["processed_dir"]
    ensure_dirs(proc_dir)

    clean_path = os.path.join(proc_dir, "cleaned_data.csv")
    df.to_csv(clean_path, index=False)
    logger.info("Cleaned data saved to %s", clean_path)

    report_dir = config["artifacts"]["reports_dir"]
    ensure_dirs(report_dir)
    save_json(report, os.path.join(report_dir, "cleaning_report.json"))

    import pandas as pd
    steps_df = pd.DataFrame(report.get("steps", []))
    steps_df.to_csv(os.path.join(report_dir, "cleaning_steps.csv"), index=False)

    return clean_path


def run_data_cleaning(csv_files: list, config: dict, sample_size: int = None) -> tuple:
    """
    Load CSVs, clean, save. Returns (cleaned_df, report).
    """
    from src.data_understanding import load_raw_data
    df = load_raw_data(csv_files, sample_size=sample_size)
    if df.empty:
        return pd.DataFrame(), {}
    df_clean, report = clean_data(df, config)
    save_cleaned_data(df_clean, report, config)
    return df_clean, report
