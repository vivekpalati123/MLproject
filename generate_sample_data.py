"""
generate_sample_data.py
Creates a realistic CIC-IDS2017-style sample CSV so the full pipeline
can run without downloading the real dataset.
"""
import os, numpy as np, pandas as pd
np.random.seed(42)

N = 15000   # total rows

ATTACK_TYPES = [
    ("BENIGN",        7000),
    ("DoS Hulk",      2000),
    ("DDoS",          1800),
    ("PortScan",      1500),
    ("FTP-Patator",    800),
    ("SSH-Patator",    700),
    ("Bot",            500),
    ("Web Attack Brute Force", 300),
    ("Web Attack XSS", 200),
    ("DoS GoldenEye",  200),
]

FEATURES = [
    "Destination Port","Flow Duration","Total Fwd Packets","Total Backward Packets",
    "Total Length of Fwd Packets","Total Length of Bwd Packets",
    "Fwd Packet Length Max","Fwd Packet Length Min","Fwd Packet Length Mean","Fwd Packet Length Std",
    "Bwd Packet Length Max","Bwd Packet Length Min","Bwd Packet Length Mean","Bwd Packet Length Std",
    "Flow Bytes/s","Flow Packets/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min",
    "Fwd IAT Total","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min",
    "Bwd IAT Total","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min",
    "Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags",
    "Fwd Header Length","Bwd Header Length",
    "Fwd Packets/s","Bwd Packets/s",
    "Min Packet Length","Max Packet Length","Packet Length Mean","Packet Length Std","Packet Length Variance",
    "FIN Flag Count","SYN Flag Count","RST Flag Count","PSH Flag Count","ACK Flag Count",
    "URG Flag Count","CWE Flag Count","ECE Flag Count",
    "Down/Up Ratio","Average Packet Size","Avg Fwd Segment Size","Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk","Fwd Avg Packets/Bulk","Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk","Bwd Avg Packets/Bulk","Bwd Avg Bulk Rate",
    "Subflow Fwd Packets","Subflow Fwd Bytes","Subflow Bwd Packets","Subflow Bwd Bytes",
    "Init_Win_bytes_forward","Init_Win_bytes_backward","act_data_pkt_fwd","min_seg_size_forward",
    "Active Mean","Active Std","Active Max","Active Min",
    "Idle Mean","Idle Std","Idle Max","Idle Min",
]

def make_row(label):
    is_attack = label != "BENIGN"
    row = {}

    # Destination port — attacks often use known ports
    if label == "PortScan":
        row["Destination Port"] = np.random.choice([21,22,23,25,80,443,3389,8080])
    elif label in ("FTP-Patator","SSH-Patator"):
        row["Destination Port"] = 21 if "FTP" in label else 22
    else:
        row["Destination Port"] = np.random.choice([80,443,8080,53,22,25])

    # Flow duration — DoS flows are very long
    if "DoS" in label or "DDoS" in label:
        row["Flow Duration"] = np.random.randint(5_000_000, 120_000_000)
    else:
        row["Flow Duration"] = np.random.randint(100, 5_000_000)

    # Packet counts
    fwd = np.random.randint(50, 5000) if is_attack else np.random.randint(2, 500)
    bwd = np.random.randint(20, 3000) if is_attack else np.random.randint(1, 300)
    row["Total Fwd Packets"] = fwd
    row["Total Backward Packets"] = bwd

    fwd_bytes = fwd * np.random.randint(40, 1500)
    bwd_bytes = bwd * np.random.randint(40, 1500)
    row["Total Length of Fwd Packets"] = fwd_bytes
    row["Total Length of Bwd Packets"] = bwd_bytes

    # Packet lengths
    for prefix, count in [("Fwd", fwd), ("Bwd", bwd)]:
        base = np.random.randint(40, 1500)
        row[f"{prefix} Packet Length Max"] = base + np.random.randint(0,200)
        row[f"{prefix} Packet Length Min"] = max(20, base - np.random.randint(0,100))
        row[f"{prefix} Packet Length Mean"] = base + np.random.randn()*20
        row[f"{prefix} Packet Length Std"] = abs(np.random.randn()*50)

    dur = max(row["Flow Duration"], 1)
    row["Flow Bytes/s"] = (fwd_bytes + bwd_bytes) / (dur / 1e6)
    row["Flow Packets/s"] = (fwd + bwd) / (dur / 1e6)

    # IAT
    for col in ["Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min",
                "Fwd IAT Total","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min",
                "Bwd IAT Total","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min"]:
        row[col] = abs(np.random.exponential(10000))

    # Flags
    for col in ["Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags"]:
        row[col] = np.random.randint(0, 2)
    row["Fwd Header Length"] = np.random.randint(20, 60) * fwd
    row["Bwd Header Length"] = np.random.randint(20, 60) * bwd
    row["Fwd Packets/s"] = fwd / (dur/1e6)
    row["Bwd Packets/s"] = bwd / (dur/1e6)

    all_pkts = np.random.randint(20, 1500, size=fwd+bwd)
    row["Min Packet Length"] = int(all_pkts.min())
    row["Max Packet Length"] = int(all_pkts.max())
    row["Packet Length Mean"] = float(all_pkts.mean())
    row["Packet Length Std"] = float(all_pkts.std())
    row["Packet Length Variance"] = float(all_pkts.var())

    # Flag counts
    row["FIN Flag Count"] = np.random.randint(0,3)
    row["SYN Flag Count"] = np.random.randint(0, 50 if is_attack else 3)
    row["RST Flag Count"] = np.random.randint(0,2)
    row["PSH Flag Count"] = np.random.randint(0, 10 if is_attack else 3)
    row["ACK Flag Count"] = np.random.randint(0, fwd+bwd)
    row["URG Flag Count"] = 0
    row["CWE Flag Count"] = 0
    row["ECE Flag Count"] = 0

    # Misc
    row["Down/Up Ratio"] = bwd / max(fwd,1)
    row["Average Packet Size"] = float(all_pkts.mean())
    row["Avg Fwd Segment Size"] = row["Fwd Packet Length Mean"]
    row["Avg Bwd Segment Size"] = row["Bwd Packet Length Mean"]
    for col in ["Fwd Avg Bytes/Bulk","Fwd Avg Packets/Bulk","Fwd Avg Bulk Rate",
                "Bwd Avg Bytes/Bulk","Bwd Avg Packets/Bulk","Bwd Avg Bulk Rate"]:
        row[col] = 0.0

    row["Subflow Fwd Packets"] = fwd
    row["Subflow Fwd Bytes"] = fwd_bytes
    row["Subflow Bwd Packets"] = bwd
    row["Subflow Bwd Bytes"] = bwd_bytes
    row["Init_Win_bytes_forward"] = np.random.choice([65535, 8192, 16384, 1024])
    row["Init_Win_bytes_backward"] = np.random.choice([65535, 8192, 16384, 1024])
    row["act_data_pkt_fwd"] = np.random.randint(1, fwd+1)
    row["min_seg_size_forward"] = np.random.randint(20, 60)

    for col in ["Active Mean","Active Std","Active Max","Active Min"]:
        row[col] = abs(np.random.exponential(50000))
    for col in ["Idle Mean","Idle Std","Idle Max","Idle Min"]:
        row[col] = abs(np.random.exponential(200000))

    row["Label"] = label
    return row

print("Generating sample CIC-IDS2017-style data...")
rows = []
for label, count in ATTACK_TYPES:
    for _ in range(count):
        rows.append(make_row(label))

df = pd.DataFrame(rows)
# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs("data/raw", exist_ok=True)
out = "data/raw/sample_cicids2017.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df):,} rows × {len(df.columns)} columns → {out}")
print(f"Label distribution:\n{df['Label'].value_counts().to_string()}")
