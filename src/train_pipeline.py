"""
train_pipeline.py - Full end-to-end training pipeline.
Run this script to train both binary and multiclass models.
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, find_csv_files, get_logger, ensure_dirs
from src.data_understanding import run_data_understanding
from src.data_cleaning import run_data_cleaning
from src.eda import run_eda
from src.feature_engineering import build_features, save_feature_list
from src.feature_selection import select_features
from src.preprocessing import split_data
from src.train_binary import train_binary_model, save_binary_artifacts
from src.train_multiclass import train_multiclass_model, save_multiclass_artifacts
from src.evaluate import evaluate_binary, evaluate_multiclass, plot_feature_importance

logger = get_logger("train_pipeline")


def run_full_pipeline(config_path: str = "config/config.yaml", sample_size: int = None):
    """Run the complete SecureNet training pipeline."""
    config = load_config(config_path)

    # ── Ensure output directories ──────────────────────────────────────────
    for key in ["data_understanding_dir", "eda_dir", "evaluation_dir", "feature_importance_dir"]:
        ensure_dirs(config["outputs"][key])
    ensure_dirs(config["artifacts"]["models_dir"], config["artifacts"]["reports_dir"])
    ensure_dirs(config["data"]["processed_dir"])

    # ── Find CSV files ─────────────────────────────────────────────────────
    raw_dir = config["data"]["raw_dir"]
    csv_files = find_csv_files(raw_dir)
    if not csv_files:
        logger.error(
            "No CSV files found in '%s'. "
            "Please download the CIC-IDS2017 dataset and place CSV files in data/raw/",
            raw_dir,
        )
        return False

    logger.info("Found %d CSV file(s): %s", len(csv_files), csv_files)

    # ── Phase 3: Data understanding ────────────────────────────────────────
    logger.info("=== PHASE 3: Data Understanding ===")
    understanding_report = run_data_understanding(csv_files, config, sample_size=sample_size or 200000)

    # ── Phase 4: Data cleaning ─────────────────────────────────────────────
    logger.info("=== PHASE 4: Data Cleaning ===")
    df_clean, cleaning_report = run_data_cleaning(csv_files, config, sample_size=sample_size)
    if df_clean.empty:
        logger.error("Cleaning produced empty DataFrame. Aborting.")
        return False

    # ── Phase 5: EDA ───────────────────────────────────────────────────────
    logger.info("=== PHASE 5: EDA ===")
    run_eda(df_clean, config)

    # ── Phase 6: Feature engineering ──────────────────────────────────────
    logger.info("=== PHASE 6: Feature Engineering ===")
    X, y_binary, y_multi, feature_cols, imputer = build_features(df_clean, config)
    save_feature_list(feature_cols, config)

    # ── Phase 6b: Feature selection ────────────────────────────────────────
    logger.info("=== PHASE 6b: Feature Selection ===")
    try:
        top_features, importance_df = select_features(X, y_binary, config, top_n=min(40, len(feature_cols)))
        # Use top features for model training
        X = X[top_features]
        feature_cols = top_features
        save_feature_list(feature_cols, config)
    except Exception as e:
        logger.warning("Feature selection failed (%s); using all features.", e)

    # ── Phase 7: Train/test split ──────────────────────────────────────────
    logger.info("=== PHASE 7: Train/Test Split ===")
    X_train, X_test, yb_train, yb_test, ym_train, ym_test, split_info = split_data(
        X, y_binary, y_multi, config
    )

    # Re-fit imputer on training data only
    from sklearn.impute import SimpleImputer
    train_imputer = SimpleImputer(strategy=config["preprocessing"]["imputer_strategy"])
    import pandas as pd
    X_train_imp = pd.DataFrame(train_imputer.fit_transform(X_train), columns=feature_cols)
    X_test_imp  = pd.DataFrame(train_imputer.transform(X_test),  columns=feature_cols)

    # Reset all series indices so positional slicing works consistently
    yb_train = yb_train.reset_index(drop=True)
    yb_test  = yb_test.reset_index(drop=True)
    ym_train = ym_train.reset_index(drop=True)
    ym_test  = ym_test.reset_index(drop=True)

    # ── Phase 8: Binary model ──────────────────────────────────────────────
    logger.info("=== PHASE 8: Binary Classification ===")
    binary_model = train_binary_model(X_train_imp, yb_train, config)
    save_binary_artifacts(binary_model, train_imputer, feature_cols, split_info, config)

    # ── Phase 10: Binary evaluation ────────────────────────────────────────
    logger.info("=== PHASE 10: Binary Evaluation ===")
    binary_metrics = evaluate_binary(binary_model, X_test_imp, yb_test, config)
    plot_feature_importance(binary_model, feature_cols, config)

    # ── Phase 9: Multiclass model ──────────────────────────────────────────
    logger.info("=== PHASE 9: Multiclass Classification ===")
    multi_metrics = {}
    try:
        multi_model, label_encoder = train_multiclass_model(X_train_imp, ym_train, config)
        save_multiclass_artifacts(multi_model, label_encoder, train_imputer, feature_cols, config)

        logger.info("=== PHASE 10: Multiclass Evaluation ===")
        # Filter test rows to known classes — use positional boolean mask
        known_classes = set(label_encoder.classes_)
        mask = ym_test.isin(known_classes).values          # numpy bool array
        if mask.sum() > 0:
            multi_metrics = evaluate_multiclass(
                multi_model, label_encoder,
                X_test_imp.iloc[mask], ym_test.iloc[mask],
                config,
            )
        else:
            logger.warning("No test rows with known multiclass labels.")
    except Exception as e:
        logger.warning("Multiclass training/evaluation failed: %s", e)

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("Binary  → Accuracy: %.4f | F1: %.4f | Recall: %.4f | ROC-AUC: %.4f",
                binary_metrics.get("accuracy", 0),
                binary_metrics.get("f1", 0),
                binary_metrics.get("recall", 0),
                binary_metrics.get("roc_auc", 0))
    if multi_metrics:
        logger.info("Multiclass → Accuracy: %.4f | Macro-F1: %.4f",
                    multi_metrics.get("accuracy", 0),
                    multi_metrics.get("macro_f1", 0))
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SecureNet Training Pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--sample", type=int, default=None,
                        help="Max rows to load (default: all)")
    args = parser.parse_args()
    success = run_full_pipeline(args.config, args.sample)
    sys.exit(0 if success else 1)
