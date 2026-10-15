"""
utils.py - Shared utility functions for the SecureNet project.
"""
import os
import json
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────
def get_logger(name: str = "securenet") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


logger = get_logger()

# ── Config ─────────────────────────────────────────────────────────────────
def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Directory helpers ───────────────────────────────────────────────────────
def ensure_dirs(*dirs: str) -> None:
    """Create directories if they do not exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ── CSV discovery ───────────────────────────────────────────────────────────
def find_csv_files(root_dir: str) -> list:
    """Recursively find all CSV files under root_dir."""
    csv_files = []
    for path in Path(root_dir).rglob("*.csv"):
        csv_files.append(str(path))
    return sorted(csv_files)


# ── Safe JSON serializer ────────────────────────────────────────────────────
class NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data: dict, path: str) -> None:
    ensure_dirs(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NpEncoder)


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ── Column name normalisation ───────────────────────────────────────────────
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names."""
    df.columns = df.columns.str.strip()
    return df


# ── Label normalisation ─────────────────────────────────────────────────────
LABEL_NORMALISATION_MAP = {
    "benign": "BENIGN",
    "dos hulk": "DoS Hulk",
    "portscan": "PortScan",
    "ddos": "DDoS",
    "dos goldeneye": "DoS GoldenEye",
    "ftppatator": "FTP-Patator",
    "sshpatator": "SSH-Patator",
    "dos slowloris": "DoS Slowloris",
    "dos slowhttptest": "DoS Slowhttptest",
    "bot": "Bot",
    "web attack – brute force": "Web Attack Brute Force",
    "web attack – xss": "Web Attack XSS",
    "web attack – sql injection": "Web Attack SQL Injection",
    "infiltration": "Infiltration",
    "heartbleed": "Heartbleed",
}


def normalize_labels(series: pd.Series) -> pd.Series:
    """Normalise label strings to a canonical form."""
    normalised = series.astype(str).str.strip()
    lower = normalised.str.lower()
    result = normalised.copy()
    for key, val in LABEL_NORMALISATION_MAP.items():
        result = result.where(lower != key, val)
    return result


def make_binary_labels(series: pd.Series, benign_label: str = "BENIGN") -> pd.Series:
    """Convert multi-class labels to binary: 0 = BENIGN, 1 = ATTACK."""
    return (series != benign_label).astype(int)


# ── Infinite-value handling ─────────────────────────────────────────────────
def replace_infinities(df: pd.DataFrame) -> pd.DataFrame:
    """Replace +/- inf with NaN."""
    return df.replace([np.inf, -np.inf], np.nan)


# ── Feature alignment ───────────────────────────────────────────────────────
def align_features(df: pd.DataFrame, expected_features: list) -> pd.DataFrame:
    """
    Align df columns to expected_features.
    Missing columns are filled with 0; extra columns are dropped.
    """
    missing = [c for c in expected_features if c not in df.columns]
    extra = [c for c in df.columns if c not in expected_features]
    if missing:
        logger.warning("Missing features filled with 0: %s", missing)
        for col in missing:
            df[col] = 0.0
    if extra:
        logger.info("Extra columns dropped: %s", extra)
        df = df.drop(columns=extra)
    return df[expected_features]
