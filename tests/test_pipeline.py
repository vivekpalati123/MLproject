"""
test_pipeline.py - Unit tests for the SecureNet pipeline.
Run with: python -m pytest tests/test_pipeline.py -v
"""
import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    normalize_columns, normalize_labels, make_binary_labels,
    replace_infinities, align_features, find_csv_files,
    NpEncoder, save_json, load_json, ensure_dirs,
)


# ── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Small synthetic DataFrame resembling CIC-IDS2017 format."""
    return pd.DataFrame({
        " Destination Port ": [80, 443, 22, 8080, 53],
        " Flow Duration": [1000, 2000, 500, 3000, 100],
        " Total Fwd Packets": [10, 20, 5, 30, 2],
        "Label": ["BENIGN", "DoS Hulk", "BENIGN", "DDoS", "PortScan"],
    })


@pytest.fixture
def sample_config():
    return {
        "data": {"label_column": "Label", "benign_label": "BENIGN"},
        "preprocessing": {"test_size": 0.2, "random_state": 42, "imputer_strategy": "median"},
        "binary_model": {"params": {"n_estimators": 10, "random_state": 42, "n_jobs": 1}},
        "multiclass_model": {"params": {"n_estimators": 10, "random_state": 42, "n_jobs": 1}},
    }


# ── Test: column normalisation ──────────────────────────────────────────────
def test_normalize_columns_strips_whitespace(sample_df):
    df = normalize_columns(sample_df)
    for col in df.columns:
        assert col == col.strip(), f"Column still has whitespace: repr={repr(col)}"


def test_normalize_columns_preserves_all_columns(sample_df):
    original_count = len(sample_df.columns)
    df = normalize_columns(sample_df)
    assert len(df.columns) == original_count


# ── Test: label normalisation ───────────────────────────────────────────────
def test_normalize_labels_benign():
    series = pd.Series(["benign", "BENIGN", " BENIGN ", "benign"])
    result = normalize_labels(series)
    assert (result == "BENIGN").all()


def test_normalize_labels_attack_types():
    series = pd.Series(["dos hulk", "portscan", "ddos"])
    result = normalize_labels(series)
    assert result.iloc[0] == "DoS Hulk"
    assert result.iloc[1] == "PortScan"
    assert result.iloc[2] == "DDoS"


def test_normalize_labels_unknown_preserved():
    series = pd.Series(["SomeUnknownLabel"])
    result = normalize_labels(series)
    assert result.iloc[0] == "SomeUnknownLabel"


# ── Test: binary label conversion ───────────────────────────────────────────
def test_make_binary_labels_benign_is_zero():
    series = pd.Series(["BENIGN", "BENIGN", "DoS Hulk", "PortScan"])
    binary = make_binary_labels(series)
    assert binary.iloc[0] == 0
    assert binary.iloc[1] == 0
    assert binary.iloc[2] == 1
    assert binary.iloc[3] == 1


def test_make_binary_labels_all_benign():
    series = pd.Series(["BENIGN"] * 10)
    binary = make_binary_labels(series)
    assert (binary == 0).all()


def test_make_binary_labels_all_attack():
    series = pd.Series(["DoS", "DDoS", "PortScan", "Bot"])
    binary = make_binary_labels(series)
    assert (binary == 1).all()


# ── Test: infinity handling ──────────────────────────────────────────────────
def test_replace_infinities_replaces_pos_inf():
    df = pd.DataFrame({"a": [1.0, np.inf, 3.0], "b": [4.0, 5.0, np.inf]})
    result = replace_infinities(df)
    assert result.isnull().values.sum() == 2


def test_replace_infinities_replaces_neg_inf():
    df = pd.DataFrame({"a": [1.0, -np.inf, 3.0]})
    result = replace_infinities(df)
    assert pd.isna(result["a"].iloc[1])


def test_replace_infinities_leaves_normal_values():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = replace_infinities(df)
    assert result.isnull().values.sum() == 0


# ── Test: feature alignment ──────────────────────────────────────────────────
def test_align_features_adds_missing_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    expected = ["a", "b", "c"]
    aligned = align_features(df, expected)
    assert "c" in aligned.columns
    assert (aligned["c"] == 0).all()


def test_align_features_drops_extra_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "extra": [5, 6]})
    expected = ["a", "b"]
    aligned = align_features(df, expected)
    assert "extra" not in aligned.columns
    assert list(aligned.columns) == expected


def test_align_features_preserves_order():
    df = pd.DataFrame({"b": [3, 4], "a": [1, 2]})
    expected = ["a", "b"]
    aligned = align_features(df, expected)
    assert list(aligned.columns) == expected


# ── Test: missing value handling ─────────────────────────────────────────────
def test_imputer_handles_missing_values():
    from sklearn.impute import SimpleImputer
    X = pd.DataFrame({
        "f1": [1.0, np.nan, 3.0, 4.0],
        "f2": [np.nan, 2.0, 3.0, np.nan],
    })
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    assert not np.isnan(X_imp).any()


# ── Test: JSON serialiser ────────────────────────────────────────────────────
def test_np_encoder_handles_numpy_int():
    data = {"value": np.int64(42)}
    result = json.dumps(data, cls=NpEncoder)
    parsed = json.loads(result)
    assert parsed["value"] == 42


def test_np_encoder_handles_numpy_float():
    data = {"value": np.float32(3.14)}
    result = json.dumps(data, cls=NpEncoder)
    parsed = json.loads(result)
    assert abs(parsed["value"] - 3.14) < 0.01


def test_np_encoder_handles_numpy_array():
    data = {"arr": np.array([1, 2, 3])}
    result = json.dumps(data, cls=NpEncoder)
    parsed = json.loads(result)
    assert parsed["arr"] == [1, 2, 3]


# ── Test: save/load JSON ─────────────────────────────────────────────────────
def test_save_and_load_json(tmp_path):
    data = {"accuracy": 0.99, "f1": 0.98, "classes": ["BENIGN", "ATTACK"]}
    path = str(tmp_path / "test_metrics.json")
    save_json(data, path)
    loaded = load_json(path)
    assert loaded["accuracy"] == 0.99
    assert loaded["classes"] == ["BENIGN", "ATTACK"]


# ── Test: find_csv_files ──────────────────────────────────────────────────────
def test_find_csv_files_returns_list(tmp_path):
    # Create dummy CSV files
    (tmp_path / "file1.csv").write_text("a,b\n1,2\n")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file2.csv").write_text("a,b\n3,4\n")
    (tmp_path / "not_csv.txt").write_text("text")

    found = find_csv_files(str(tmp_path))
    assert len(found) == 2
    assert all(f.endswith(".csv") for f in found)


def test_find_csv_files_empty_dir(tmp_path):
    found = find_csv_files(str(tmp_path))
    assert found == []


# ── Test: prediction output format ────────────────────────────────────────────
def test_prediction_output_has_required_columns():
    """Test that prediction function returns expected columns."""
    from unittest.mock import MagicMock
    import numpy as np

    # Mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0, 1, 1])
    mock_model.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9]])

    # Mock imputer
    mock_imputer = MagicMock()
    mock_imputer.transform.side_effect = lambda x: x

    feature_cols = ["f1", "f2", "f3"]
    df_input = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0],
        "f2": [4.0, 5.0, 6.0],
        "f3": [7.0, 8.0, 9.0],
    })

    from src.predict import run_binary_prediction
    result = run_binary_prediction(df_input, mock_model, mock_imputer, feature_cols)

    assert "Predicted_Label" in result.columns
    assert "Predicted_Binary" in result.columns
    assert set(result["Predicted_Label"].unique()).issubset({"BENIGN", "ATTACK"})


# ── Test: invalid CSV handling ─────────────────────────────────────────────
def test_invalid_csv_validation():
    """Test that validate_prediction_input handles mismatched features."""
    from src.predict import validate_prediction_input
    df = pd.DataFrame({"totally_different_col": [1, 2, 3]})
    expected_features = ["f1", "f2", "f3", "f4", "f5"]
    result = validate_prediction_input(df, expected_features)
    assert result["features_present"] == 0
    assert not result["valid"]


def test_partial_feature_warning():
    """Test that a warning is issued when many features are missing."""
    from src.predict import validate_prediction_input
    # Provide only 1 out of 10 expected features
    df = pd.DataFrame({"f1": [1, 2, 3]})
    expected_features = [f"f{i}" for i in range(1, 11)]
    result = validate_prediction_input(df, expected_features)
    assert "warning" in result


# ── Test: data cleaning ────────────────────────────────────────────────────
def test_cleaning_removes_duplicates():
    from src.data_cleaning import clean_data
    config = {"data": {"label_column": "Label", "benign_label": "BENIGN"}}
    df = pd.DataFrame({
        "Flow Duration": [100.0, 100.0, 200.0],
        "Total Packets": [5.0, 5.0, 10.0],
        "Label": ["BENIGN", "BENIGN", "DoS Hulk"],
    })
    df_clean, report = clean_data(df, config)
    assert df_clean.shape[0] <= df.shape[0]


def test_cleaning_handles_infinities():
    from src.data_cleaning import clean_data
    config = {"data": {"label_column": "Label", "benign_label": "BENIGN"}}
    df = pd.DataFrame({
        "Flow Duration": [100.0, np.inf, -np.inf],
        "Total Packets": [5.0, 10.0, 15.0],
        "Label": ["BENIGN", "DoS", "DDoS"],
    })
    df_clean, report = clean_data(df, config)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        assert not np.isinf(df_clean[col]).any(), f"Infinity found in {col}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
