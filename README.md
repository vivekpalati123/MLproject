# 🛡️ SecureNet: AI-Based Network Intrusion Detection and Cyber Threat Intelligence System

An end-to-end machine learning project for detecting network intrusions using the **CIC-IDS2017** dataset.

---

## 📋 Project Introduction

SecureNet is an academic machine learning project that detects and classifies network intrusions
using the CIC-IDS2017 dataset. It provides a complete pipeline from raw data to a polished
Streamlit dashboard with binary and multiclass classification models.

---

## 🎯 Problem Statement

Modern networks face sophisticated cyber threats. Manual analysis of network traffic logs is
impractical at scale. This project trains ML models on extracted network flow features to
automatically classify traffic as **benign** or **attack**, and further identify the specific
attack category (DoS, DDoS, PortScan, etc.).

---

## 🏆 Objectives

1. Understand the CIC-IDS2017 dataset structure and statistics
2. Clean and preprocess raw network flow CSVs
3. Engineer and select predictive features
4. Train a binary intrusion detection model (BENIGN vs ATTACK)
5. Train a multiclass model to identify specific attack types
6. Evaluate models on held-out test data
7. Build an interactive professional dashboard
8. Support CSV batch prediction

---

## 📊 Dataset

| Attribute | Value |
|---|---|
| Name | CIC-IDS2017 |
| Source | Canadian Institute for Cybersecurity |
| Kaggle | https://www.kaggle.com/datasets/cicdataset/cicids2017 |
| Official | https://www.unb.ca/cic/datasets/ids-2017.html |
| Features | 78 network flow features (CICFlowMeter) |
| Attack Types | DoS, DDoS, PortScan, Bot, Web Attacks, Infiltration, Heartbleed |

---

## 📁 Project Structure

```
SecureNet/
│
├── app.py                          # Streamlit dashboard (main entry point)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore
│
├── config/
│   └── config.yaml                 # All configuration parameters
│
├── data/
│   ├── raw/                        # Place CIC-IDS2017 CSV files here
│   ├── processed/                  # Cleaned data (generated)
│   └── sample/                     # Small sample for testing
│
├── src/
│   ├── utils.py                    # Shared utilities
│   ├── data_understanding.py       # Phase 3: Data statistics
│   ├── data_cleaning.py            # Phase 4: Data cleaning
│   ├── eda.py                      # Phase 5: EDA plots
│   ├── feature_engineering.py      # Phase 6: Feature pipeline
│   ├── feature_selection.py        # Feature importance ranking
│   ├── preprocessing.py            # Phase 7: Train/test split
│   ├── train_binary.py             # Phase 8: Binary model
│   ├── train_multiclass.py         # Phase 9: Multiclass model
│   ├── evaluate.py                 # Phase 10: Evaluation metrics
│   ├── predict.py                  # Phase 14: Batch prediction
│   └── train_pipeline.py           # Full pipeline runner
│
├── artifacts/
│   ├── models/                     # Saved .joblib model files
│   └── reports/                    # Cleaning and training reports
│
├── outputs/
│   ├── data_understanding/         # Dataset stats and charts
│   ├── eda/                        # EDA plots
│   ├── evaluation/                 # Metrics and confusion matrices
│   └── feature_importance/         # Feature rankings
│
└── tests/
    └── test_pipeline.py            # Unit tests
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 or newer
- pip

### Windows (Command Prompt or PowerShell)

```cmd
cd "path\to\security net"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### PyCharm

1. Open PyCharm → **File** → **Open** → select the `security net` folder
2. PyCharm will detect the project
3. Go to **File** → **Settings** → **Project** → **Python Interpreter**
4. Click the gear icon → **Add** → **Virtual Environment** → **New** → OK
5. Open the Terminal at the bottom and run:
   ```
   pip install -r requirements.txt
   ```

---

## 📥 Dataset Setup

1. Visit https://www.kaggle.com/datasets/cicdataset/cicids2017
2. Sign in / create a free Kaggle account
3. Click **Download** to download the ZIP file
4. Extract the ZIP
5. Copy all `.csv` files into the `data/raw/` folder:

```
data/
└── raw/
    ├── Monday-WorkingHours.pcap_ISCX.csv
    ├── Tuesday-WorkingHours.pcap_ISCX.csv
    ├── Wednesday-workingHours.pcap_ISCX.csv
    ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
    ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    └── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

> **Tip:** You can start with just one or two CSV files to test the pipeline quickly.

---

## 🚀 How to Run Training

From the project root folder (where `app.py` is located):

```cmd
python src/train_pipeline.py
```

**Optional: limit rows for faster testing**
```cmd
python src/train_pipeline.py --sample 50000
```

Training will:
1. Load and understand the dataset
2. Clean the data
3. Run EDA
4. Engineer and select features
5. Split into train/test sets
6. Train binary RandomForestClassifier
7. Train multiclass ExtraTreesClassifier
8. Evaluate both models
9. Save all artifacts and reports

---

## 🌐 How to Launch the Dashboard

```cmd
streamlit run app.py
```

Then open your browser at: **http://localhost:8501**

**In PyCharm:**
1. Open Terminal (bottom panel)
2. Run: `streamlit run app.py`
3. Click the URL that appears

---

## 🧪 Running Tests

```cmd
python -m pytest tests/test_pipeline.py -v
```

Or in PyCharm: right-click `tests/test_pipeline.py` → **Run pytest**

---

## 🤖 ML Algorithms

| Task | Algorithm | Key Parameters |
|---|---|---|
| Binary (BENIGN/ATTACK) | RandomForestClassifier | 100 trees, max_depth=20, balanced |
| Multiclass (Attack types) | ExtraTreesClassifier | 100 trees, max_depth=20, balanced |
| Feature Selection | ExtraTreesClassifier | 50 trees, max_depth=10 |
| Imputation | SimpleImputer | strategy=median |

---

## 📈 Evaluation Metrics

### Binary Classification
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Confusion Matrix (TP, TN, FP, FN)
- Classification Report

### Multiclass Classification
- Accuracy
- Macro Precision, Macro Recall, Macro F1
- Weighted F1
- Per-class Classification Report
- Confusion Matrix

---

## 📤 Expected Output Files

After running the pipeline:

```
artifacts/models/
  binary_model.joblib
  binary_imputer.joblib
  binary_metadata.json
  multiclass_model.joblib
  multiclass_encoder.joblib
  multiclass_imputer.joblib
  multiclass_metadata.json

artifacts/reports/
  cleaning_report.json
  cleaning_steps.csv

outputs/data_understanding/
  data_understanding_report.json
  dataset_summary.csv
  class_distribution.csv
  label_distribution.png
  benign_vs_attack.png
  attack_distribution.png
  feature_distributions.png
  correlation_heatmap.png

outputs/eda/
  benign_vs_attack.png
  attack_families.png
  class_imbalance.png
  feature_distributions.png
  correlation_heatmap.png

outputs/evaluation/
  binary_metrics.json
  binary_metrics.csv
  binary_confusion_matrix.png
  binary_roc_curve.png
  multiclass_metrics.json
  multiclass_confusion_matrix.png

outputs/feature_importance/
  feature_list.json
  feature_list.csv
  feature_importances.csv
  feature_importance_plot.png
  feature_importance_model.png
```

---

## ⚠️ Known Limitations

- The system requires **pre-extracted** network flow features (CICFlowMeter format)
- It does **not** capture live network packets automatically
- No external real-time threat intelligence feeds are connected
- Model accuracy depends on dataset coverage and training sample size
- Real-world deployment requires retraining on current network patterns

---

## 🔮 Future Enhancements

- LSTM/Transformer models for temporal traffic patterns
- Real-time packet capture integration with CICFlowMeter
- SHAP values for per-prediction explainability
- SIEM/SOAR platform integration
- Automated retraining pipeline
- Docker containerization

---

## 📚 References

- Sharafaldin, I., Habibi Lashkari, A., Ghorbani, A.A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* ICISSP 2018.
- CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- Kaggle: https://www.kaggle.com/datasets/cicdataset/cicids2017

---

*SecureNet v1.0.0 | Academic Project | B.Tech First Year*
