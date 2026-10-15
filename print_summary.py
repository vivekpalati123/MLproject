import json, pandas as pd, os

SEP = "=" * 62

def j(path):
    with open(path) as f:
        return json.load(f)

r  = j("outputs/data_understanding/data_understanding_report.json")
bm = j("outputs/evaluation/binary_metrics.json")
mm = j("outputs/evaluation/multiclass_metrics.json")
cr = j("artifacts/reports/cleaning_report.json")
fl = j("outputs/feature_importance/feature_list.json")
fi = pd.read_csv("outputs/feature_importance/top20_features.csv")

print(SEP)
print("     SECURENET - FULL PROJECT OUTPUT SUMMARY")
print(SEP)

print("\n[01] DATA UNDERSTANDING")
print(f"  Total Rows        : {r['n_rows']:,}")
print(f"  Total Columns     : {r['n_cols']}")
print(f"  Numeric Features  : {r['n_numeric']}")
print(f"  Categorical       : {r['n_categorical']}")
print(f"  Benign Rows       : {r['benign_count']:,}")
print(f"  Attack Rows       : {r['attack_count']:,}")
print(f"  Missing Values    : {r['total_missing']}")
print(f"  Duplicate Rows    : {r['duplicate_rows']}")
print(f"  Infinite Values   : {r['infinite_values']}")

print("\n[02] CLASS DISTRIBUTION")
for k,v in sorted(r['class_distribution'].items(), key=lambda x:-x[1]):
    pct = v / r['n_rows'] * 100
    bar = "#" * int(pct/2)
    print(f"  {k:<30} {v:>6,}  ({pct:5.1f}%)  {bar}")

print("\n[03] DATA CLEANING")
print(f"  Initial Shape     : {cr['initial_shape'][0]:,} rows x {cr['initial_shape'][1]} cols")
print(f"  Final Shape       : {cr['final_shape'][0]:,} rows x {cr['final_shape'][1]} cols")
print(f"  Rows Removed      : {cr['rows_removed']:,}")
print(f"  Cols Removed      : {cr['cols_removed']}")
print(f"  Remaining Missing : {cr['remaining_missing']}")

print("\n[04] FEATURE ENGINEERING & SELECTION")
print(f"  Features After Engineering : {fl['count']}")
steps = pd.read_csv("artifacts/reports/cleaning_steps.csv")
print(f"  Cleaning Steps Applied     : {len(steps)}")

print("\n[05] TRAIN/TEST SPLIT")
si = j("artifacts/models/binary_metadata.json")["split_info"]
print(f"  Training Rows     : {si['train_rows']:,}  (80%)")
print(f"  Test Rows         : {si['test_rows']:,}  (20%)")
print(f"  Train Benign      : {si['train_benign']:,}")
print(f"  Train Attack      : {si['train_attack']:,}")
print(f"  Test Benign       : {si['test_benign']:,}")
print(f"  Test Attack       : {si['test_attack']:,}")

print("\n[06] BINARY MODEL - RandomForestClassifier")
print(f"  Accuracy          : {bm['accuracy']*100:.4f} %")
print(f"  Precision         : {bm['precision']:.4f}")
print(f"  Recall            : {bm['recall']:.4f}")
print(f"  F1 Score          : {bm['f1']:.4f}")
print(f"  ROC-AUC           : {bm.get('roc_auc', 'N/A')}")
print(f"  PR-AUC            : {bm.get('pr_auc', 'N/A')}")
print(f"  True Positives    : {bm['tp']:,}  (Attacks Caught)")
print(f"  True Negatives    : {bm['tn']:,}  (Benign Correct)")
print(f"  False Positives   : {bm['fp']:,}  (Wrong Alerts)")
print(f"  False Negatives   : {bm['fn']:,}  (Missed Attacks)")

print("\n[07] MULTICLASS MODEL - ExtraTreesClassifier")
print(f"  Accuracy          : {mm['accuracy']*100:.4f} %")
print(f"  Macro Precision   : {mm['macro_precision']:.4f}")
print(f"  Macro Recall      : {mm['macro_recall']:.4f}")
print(f"  Macro F1          : {mm['macro_f1']:.4f}")
print(f"  Weighted F1       : {mm['weighted_f1']:.4f}")
print(f"  Classes           : {', '.join(mm['classes'])}")

print("\n[08] TOP 10 FEATURE IMPORTANCES")
for _, row in fi.head(10).iterrows():
    bar = "#" * int(row['importance'] * 300)
    print(f"  {row['feature']:<40} {row['importance']:.6f}  {bar}")

print("\n[09] CLASSIFICATION REPORT (BINARY)")
print(bm['classification_report'])

print("\n[10] GENERATED OUTPUT FILES")
for root, dirs, files in os.walk("outputs"):
    for f in files:
        path = os.path.join(root, f)
        size = os.path.getsize(path)
        print(f"  {path:<65} {size:>8,} bytes")
for root, dirs, files in os.walk("artifacts"):
    for f in files:
        path = os.path.join(root, f)
        size = os.path.getsize(path)
        print(f"  {path:<65} {size:>8,} bytes")

print()
print(SEP)
print("  DASHBOARD  ->  http://localhost:8501")
print(SEP)
