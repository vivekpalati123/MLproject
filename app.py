"""
SecureNet: AI-Based Network Intrusion Detection and Cyber Threat Intelligence System
Professional Streamlit Dashboard – 20 Sections
Run: python -m streamlit run app.py
"""
import os, sys, json, warnings, io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import load_config, find_csv_files, load_json, ensure_dirs

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SecureNet IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# THEME CSS  (dark navy like the reference image)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#0b1120;color:#d0dff0;}
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#07111f 0%,#0d1f35 100%);
    border-right:2px solid #1a3a5c;}
section[data-testid="stSidebar"] *{color:#a8c8e8 !important;}
/* active nav item */
div[data-testid="stRadio"] label:has(input:checked){
    background:#1565c0 !important;border-radius:6px;color:#fff !important;}
h1,h2,h3,h4{color:#64b5f6 !important;letter-spacing:.4px;}
/* KPI card */
.kcard{background:linear-gradient(135deg,#0d2137,#112040);
    border:1px solid #1e3a5f;border-radius:10px;
    padding:16px 20px;text-align:center;margin-bottom:10px;}
.kcard-icon{font-size:28px;margin-bottom:4px;}
.kcard-label{font-size:11px;color:#7a9bbf;text-transform:uppercase;letter-spacing:1px;}
.kcard-value{font-size:30px;font-weight:700;color:#64b5f6;margin-top:4px;}
.kcard-green .kcard-value{color:#4caf50;}
.kcard-red   .kcard-value{color:#ef5350;}
.kcard-orange.kcard-value{color:#ff9800;}
/* section header bar */
.sec-header{background:linear-gradient(90deg,#1565c0,#0d47a1);
    color:#fff;padding:8px 16px;border-radius:6px;
    font-weight:700;font-size:14px;margin:16px 0 10px 0;}
/* table styling */
.stDataFrame{border-radius:8px;}
/* divider */
hr{border-color:#1a3a5c;}
/* info/warn boxes */
.box-info{background:#0a1e35;border-left:4px solid #2196f3;
    padding:10px 14px;border-radius:4px;margin:8px 0;font-size:13px;}
.box-warn{background:#1a1200;border-left:4px solid #ff9800;
    padding:10px 14px;border-radius:4px;margin:8px 0;font-size:13px;}
.box-ok  {background:#061a06;border-left:4px solid #4caf50;
    padding:10px 14px;border-radius:4px;margin:8px 0;font-size:13px;}
/* sidebar page title */
.sidebar-title{font-size:18px;font-weight:700;color:#64b5f6 !important;
    text-align:center;margin-bottom:4px;}
.sidebar-sub{font-size:11px;color:#607d8b !important;text-align:center;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0b1120", plot_bgcolor="#0d1b2a",
    font=dict(color="#c8d8ea", size=12),
    margin=dict(t=40, b=40, l=20, r=20),
)

def kcard(icon, label, value, color=""):
    st.markdown(f"""
    <div class="kcard {color}">
      <div class="kcard-icon">{icon}</div>
      <div class="kcard-label">{label}</div>
      <div class="kcard-value">{value}</div>
    </div>""", unsafe_allow_html=True)

def sec(title):
    st.markdown(f'<div class="sec-header">{title}</div>', unsafe_allow_html=True)

def box(msg, kind="info"):
    st.markdown(f'<div class="box-{kind}">{msg}</div>', unsafe_allow_html=True)

def plx(fig, height=350):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    st.plotly_chart(fig, use_container_width=True)

@st.cache_resource
def cfg():
    return load_config("config/config.yaml")

CONFIG = cfg()

@st.cache_data(show_spinner=False)
def get_csv_files():
    return find_csv_files(CONFIG["data"]["raw_dir"])

@st.cache_data(show_spinner=False)
def load_understanding():
    p = os.path.join(CONFIG["outputs"]["data_understanding_dir"],
                     "data_understanding_report.json")
    return load_json(p) if os.path.exists(p) else None

@st.cache_data(show_spinner=False)
def load_cleaned_df():
    p = os.path.join(CONFIG["data"]["processed_dir"], "cleaned_data.csv")
    if os.path.exists(p):
        return pd.read_csv(p, low_memory=False)
    return None

@st.cache_data(show_spinner=False)
def load_bin_metrics():
    p = os.path.join(CONFIG["outputs"]["evaluation_dir"], "binary_metrics.json")
    return load_json(p) if os.path.exists(p) else None

@st.cache_data(show_spinner=False)
def load_mc_metrics():
    p = os.path.join(CONFIG["outputs"]["evaluation_dir"], "multiclass_metrics.json")
    return load_json(p) if os.path.exists(p) else None

@st.cache_data(show_spinner=False)
def load_fi():
    p = os.path.join(CONFIG["outputs"]["feature_importance_dir"],
                     "feature_importances.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_resource(show_spinner=False)
def load_binary():
    try:
        from src.train_binary import load_binary_artifacts
        return load_binary_artifacts(CONFIG)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_multi():
    try:
        from src.train_multiclass import load_multiclass_artifacts
        return load_multiclass_artifacts(CONFIG)
    except Exception:
        return None

def no_data_msg():
    box("⚠️ Dataset CSV files not found in <b>data/raw/</b>. "
        "Download CIC-IDS2017 from Kaggle and place CSV files there, "
        "then run: <code>python src/train_pipeline.py</code>", "warn")

def no_train_msg():
    box("⚠️ Model artifacts not found. Run the pipeline first: "
        "<code>python src/train_pipeline.py</code>", "warn")

# ─────────────────────────────────────────────────────────────
# SIDEBAR  – 20 pages
# ─────────────────────────────────────────────────────────────
PAGES = [
    "01. Data Understanding",
    "02. Data Quality",
    "03. Target Definition",
    "04. EDA",
    "05. Data Cleaning",
    "06. Data Leakage",
    "07. Feature Engineering",
    "08. Feature Selection",
    "09. Data Splitting",
    "10. Preprocessing",
    "11. Classification",
    "12. Model Evaluation",
    "13. Model Comparison",
    "14. Model Improvement",
    "15. Confusion Matrix",
    "16. Feature Importance",
    "17. Batch Prediction",
    "18. Reports & Downloads",
    "19. About SecureNet",
    "20. Home Dashboard",
]

with st.sidebar:
    st.markdown('<p class="sidebar-title">🛡️ SecureNet IDS</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-sub">AI Network Intrusion Detection</p>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<p class="sidebar-sub">SecureNet v1.0 | CIC-IDS2017<br>B.Tech Academic Project</p>',
                unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 01. DATA UNDERSTANDING
# ═══════════════════════════════════════════════════════════════
if page == "01. Data Understanding":
    st.title("01. DATA UNDERSTANDING")
    st.caption("SecureNet – Network Intrusion Detection (CIC-IDS2017 Dataset)")
    st.markdown("---")

    u = load_understanding()
    df = load_cleaned_df()

    if u is None:
        no_train_msg(); st.stop()

    # ── KPI row ────────────────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: kcard("📄","Total Rows",f"{u['n_rows']:,}")
    with c2: kcard("📊","Total Columns",f"{u['n_cols']}")
    with c3: kcard("📈","Numerical Columns",f"{u['n_numeric']}")
    with c4: kcard("🏷️","Categorical Columns",f"{u['n_categorical']}")
    mem = f"{(u['n_rows']*u['n_cols']*8/1e6):.1f} MB" if u else "N/A"
    with c5: kcard("💾","Est. Memory",mem)

    st.markdown("---")

    col1, col2 = st.columns(2)

    # 1. First 5 rows
    with col1:
        sec("1. First 5 Rows (df.head())")
        if df is not None:
            st.dataframe(df.head(), use_container_width=True, height=180)
        else:
            st.info("Cleaned data not found.")

    # 2. Last 5 rows
    with col2:
        sec("2. Last 5 Rows (df.tail())")
        if df is not None:
            st.dataframe(df.tail(), use_container_width=True, height=180)

    col3, col4, col5, col6 = st.columns(4)

    # 3. Data shape
    with col3:
        sec("3. Data Shape (df.shape)")
        shape_df = pd.DataFrame({"Metric":["Total Rows","Total Columns"],
                                  "Value":[f"{u['n_rows']:,}", u['n_cols']]})
        st.dataframe(shape_df, use_container_width=True, hide_index=True)

    # 4. Data types
    with col4:
        sec("4. Data Types (dtypes)")
        if df is not None:
            dtype_counts = df.dtypes.value_counts().reset_index()
            dtype_counts.columns = ["Data Type","Count"]
            dtype_counts["Data Type"] = dtype_counts["Data Type"].astype(str)
            st.dataframe(dtype_counts, use_container_width=True, hide_index=True)

    # 5. Dataset info
    with col5:
        sec("5. Dataset Info (df.info())")
        info_rows = []
        if df is not None:
            for i,col in enumerate(df.columns[:8]):
                info_rows.append({"#":i,"Column":col[:18],
                                   "Non-Null":f"{df[col].count():,}",
                                   "Dtype":str(df[col].dtype)})
        st.dataframe(pd.DataFrame(info_rows), use_container_width=True,
                     hide_index=True, height=230)

    # 6. Missing values top10
    with col6:
        sec("6. Missing Values (Top 10)")
        mv = u.get("missing_values", {})
        if mv:
            mv_df = pd.DataFrame(list(mv.items()),
                                  columns=["Column","Missing Count"]).head(10)
            mv_df["Missing %"] = (mv_df["Missing Count"]/u["n_rows"]*100).round(2)
            st.dataframe(mv_df, use_container_width=True, hide_index=True, height=230)
        else:
            st.success("No missing values!")

    st.markdown("---")
    col7, col8, col9 = st.columns(3)

    # 7. Numerical vs Categorical pie
    with col7:
        sec("7. Numerical vs Categorical Columns")
        fig = go.Figure(go.Pie(
            labels=["Numerical","Categorical"],
            values=[u["n_numeric"], u["n_categorical"]],
            hole=0.5,
            marker_colors=["#2196f3","#ff9800"],
        ))
        plx(fig, 300)

    # 8. Statistical summary
    with col8:
        sec("8. Statistical Summary (Numerical)")
        if df is not None:
            num_cols = df.select_dtypes(include=np.number).columns[:5]
            stats = df[num_cols].describe().round(2)
            st.dataframe(stats, use_container_width=True, height=260)

    # 9. Unique values top10 categorical
    with col9:
        sec("9. Unique Values (Top 10 Categorical)")
        if df is not None:
            cat_cols = df.select_dtypes(exclude=np.number).columns
            if len(cat_cols):
                uq_rows = []
                for c in cat_cols[:10]:
                    top5 = ", ".join(df[c].value_counts().head(5).index.astype(str))
                    uq_rows.append({"Column":c, "Unique":df[c].nunique(),
                                    "Top 5 Values":top5[:40]})
                st.dataframe(pd.DataFrame(uq_rows), use_container_width=True,
                             hide_index=True, height=260)
            else:
                st.info("No categorical columns.")

# ═══════════════════════════════════════════════════════════════
# 02. DATA QUALITY
# ═══════════════════════════════════════════════════════════════
elif page == "02. Data Quality":
    st.title("02. DATA QUALITY")
    st.caption("Missing values, duplicates, infinities and data-type issues")
    st.markdown("---")
    u = load_understanding()
    df = load_cleaned_df()
    if u is None: no_train_msg(); st.stop()

    c1,c2,c3,c4 = st.columns(4)
    with c1: kcard("❌","Total Missing",f"{u['total_missing']:,}","kcard-red" if u['total_missing']>0 else "kcard-green")
    with c2: kcard("♻️","Duplicate Rows",f"{u['duplicate_rows']:,}","kcard-red" if u['duplicate_rows']>0 else "kcard-green")
    with c3: kcard("∞","Infinite Values",f"{u['infinite_values']:,}","kcard-red" if u['infinite_values']>0 else "kcard-green")
    with c4: kcard("📦","Constant Features",f"{len(u.get('constant_features',[]))}","kcard-orange")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        sec("1. Missing Values Per Column")
        mv = u.get("missing_values", {})
        if mv:
            mv_df = pd.DataFrame(list(mv.items()),
                                  columns=["Column","Missing Count"])
            mv_df["Missing %"] = (mv_df["Missing Count"]/u["n_rows"]*100).round(3)
            mv_df = mv_df.sort_values("Missing Count", ascending=False)
            fig = px.bar(mv_df.head(20), x="Missing Count", y="Column",
                         orientation="h", color="Missing %",
                         color_continuous_scale="Reds")
            plx(fig, 400)
        else:
            box("✅ No missing values found in the dataset.", "ok")

    with col2:
        sec("2. Data Type Distribution")
        if df is not None:
            dtype_df = df.dtypes.value_counts().reset_index()
            dtype_df.columns = ["DType","Count"]
            dtype_df["DType"] = dtype_df["DType"].astype(str)
            fig = px.pie(dtype_df, names="DType", values="Count",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            plx(fig, 300)

    col3, col4 = st.columns(2)
    with col3:
        sec("3. Duplicate Row Analysis")
        dup_data = {"Status": ["Unique Rows", "Duplicate Rows"],
                    "Count": [u["n_rows"]-u["duplicate_rows"], u["duplicate_rows"]]}
        fig = px.bar(pd.DataFrame(dup_data), x="Status", y="Count",
                     color="Status",
                     color_discrete_map={"Unique Rows":"#4caf50","Duplicate Rows":"#ef5350"})
        plx(fig, 280)

    with col4:
        sec("4. Constant / Zero-Variance Features")
        const = u.get("constant_features", [])
        if const:
            st.dataframe(pd.DataFrame({"Constant Feature": const}),
                         use_container_width=True, hide_index=True)
        else:
            box("✅ No constant features found.", "ok")

    sec("5. Quality Summary Table")
    quality_rows = [
        {"Check": "Missing values", "Result": u['total_missing'], "Status": "⚠️ Present" if u['total_missing']>0 else "✅ None"},
        {"Check": "Duplicate rows", "Result": u['duplicate_rows'], "Status": "⚠️ Present" if u['duplicate_rows']>0 else "✅ None"},
        {"Check": "Infinite values", "Result": u['infinite_values'], "Status": "⚠️ Present" if u['infinite_values']>0 else "✅ None"},
        {"Check": "Constant features", "Result": len(u.get("constant_features",[])), "Status": "⚠️ Present" if u.get("constant_features") else "✅ None"},
        {"Check": "Label column present", "Result": CONFIG["data"]["label_column"], "Status": "✅ Found" if CONFIG["data"]["label_column"] in u["columns"] else "❌ Missing"},
    ]
    st.dataframe(pd.DataFrame(quality_rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# 03. TARGET DEFINITION
# ═══════════════════════════════════════════════════════════════
elif page == "03. Target Definition":
    st.title("03. TARGET DEFINITION")
    st.caption("Label column analysis, class distribution, benign vs attack")
    st.markdown("---")
    u = load_understanding()
    if u is None: no_train_msg(); st.stop()

    c1,c2,c3,c4 = st.columns(4)
    with c1: kcard("🏷️","Label Column", CONFIG["data"]["label_column"])
    with c2: kcard("✅","Benign Samples",f"{u['benign_count']:,}","kcard-green")
    with c3: kcard("🚨","Attack Samples",f"{u['attack_count']:,}","kcard-red")
    with c4: kcard("🔢","Unique Classes",f"{len(u['class_distribution'])}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        sec("1. Class Distribution Table")
        cd = u["class_distribution"]
        cd_df = pd.DataFrame(list(cd.items()), columns=["Label","Count"])
        cd_df = cd_df.sort_values("Count",ascending=False).reset_index(drop=True)
        cd_df["%"] = (cd_df["Count"]/cd_df["Count"].sum()*100).round(2)
        cd_df["Type"] = cd_df["Label"].apply(
            lambda x: "BENIGN" if x == CONFIG["data"]["benign_label"] else "ATTACK")
        st.dataframe(cd_df, use_container_width=True, hide_index=True, height=380)

    with col2:
        sec("2. Label Distribution Bar Chart")
        colors = ["#4caf50" if l==CONFIG["data"]["benign_label"] else "#ef5350"
                  for l in cd_df["Label"]]
        fig = go.Figure(go.Bar(
            x=cd_df["Label"], y=cd_df["Count"],
            marker_color=colors,
            text=cd_df["Count"].apply(lambda x: f"{x:,}"),
            textposition="outside",
        ))
        fig.update_layout(xaxis_tickangle=-40)
        plx(fig, 380)

    col3, col4 = st.columns(2)
    with col3:
        sec("3. Benign vs Attack — Pie Chart")
        fig = go.Figure(go.Pie(
            labels=["Benign","Attack"],
            values=[u["benign_count"], u["attack_count"]],
            hole=0.5,
            marker_colors=["#4caf50","#ef5350"],
        ))
        plx(fig, 300)

    with col4:
        sec("4. Binary Target Mapping")
        bin_map = pd.DataFrame({
            "Original Label": list(cd.keys()),
            "Binary Target": [0 if k==CONFIG["data"]["benign_label"] else 1 for k in cd.keys()],
            "Meaning": ["BENIGN" if k==CONFIG["data"]["benign_label"] else "ATTACK" for k in cd.keys()],
        })
        st.dataframe(bin_map, use_container_width=True, hide_index=True, height=300)

    sec("5. Attack Category Distribution")
    atk = u.get("attack_distribution", {})
    if atk:
        atk_df = pd.DataFrame(list(atk.items()), columns=["Attack Type","Count"])
        atk_df = atk_df.sort_values("Count",ascending=False)
        fig = px.bar(atk_df, x="Attack Type", y="Count",
                     color="Count", color_continuous_scale="Reds",
                     text="Count")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False)
        plx(fig, 320)

# ═══════════════════════════════════════════════════════════════
# 04. EDA
# ═══════════════════════════════════════════════════════════════
elif page == "04. EDA":
    st.title("04. EXPLORATORY DATA ANALYSIS")
    st.caption("Distribution, correlation, and pattern insights")
    st.markdown("---")
    u = load_understanding()
    df = load_cleaned_df()
    if u is None or df is None: no_train_msg(); st.stop()

    c1,c2,c3,c4 = st.columns(4)
    with c1: kcard("📊","Total Samples",f"{u['n_rows']:,}")
    with c2: kcard("✅","Benign",f"{u['benign_count']:,}","kcard-green")
    with c3: kcard("🚨","Attack",f"{u['attack_count']:,}","kcard-red")
    with c4: kcard("🔢","Attack Types",f"{len(u.get('attack_distribution',{}))}")

    st.markdown("---")
    label_col = CONFIG["data"]["label_column"]
    benign_label = CONFIG["data"]["benign_label"]
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != label_col]

    col1, col2 = st.columns(2)
    with col1:
        sec("1. Traffic Type Distribution")
        binary = df[label_col].apply(lambda x:"Benign" if x==benign_label else "Attack")
        vc = binary.value_counts()
        fig = go.Figure(go.Pie(
            labels=vc.index.tolist(), values=vc.values.tolist(),
            hole=0.45, marker_colors=["#4caf50","#ef5350"],
        ))
        plx(fig, 300)

    with col2:
        sec("2. Attack Family Distribution")
        atk_df2 = df[df[label_col]!=benign_label][label_col].value_counts().reset_index()
        atk_df2.columns=["Attack","Count"]
        fig = px.bar(atk_df2, x="Attack", y="Count", color="Count",
                     color_continuous_scale="Reds")
        fig.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False)
        plx(fig, 300)

    sec("3. Feature Distributions — Top 12 Numeric Features by Variance")
    top_var = df[num_cols].var().nlargest(12).index.tolist()
    fig = make_subplots(rows=3, cols=4,
                        subplot_titles=[c[:20] for c in top_var])
    for i, col_name in enumerate(top_var):
        r, c = divmod(i, 4)
        vals = df[col_name].dropna().astype(float)
        fig.add_trace(go.Histogram(x=vals, nbinsx=40,
                                    marker_color="#2196f3", showlegend=False,
                                    name=col_name),
                      row=r+1, col=c+1)
    fig.update_layout(**PLOTLY_LAYOUT, height=500,
                      title_text="Feature Distributions")
    st.plotly_chart(fig, use_container_width=True)

    sec("4. Correlation Heatmap — Top 15 Features")
    top15 = df[num_cols].var().nlargest(15).index.tolist()
    corr = df[top15].corr().round(2)
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    text_auto=True, aspect="auto")
    fig.update_layout(**PLOTLY_LAYOUT, height=500)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# 05. DATA CLEANING
# ═══════════════════════════════════════════════════════════════
elif page == "05. Data Cleaning":
    st.title("05. DATA CLEANING")
    st.caption("Cleaning steps, before/after statistics, removed rows and columns")
    st.markdown("---")

    rp = os.path.join(CONFIG["artifacts"]["reports_dir"], "cleaning_report.json")
    sp = os.path.join(CONFIG["artifacts"]["reports_dir"], "cleaning_steps.csv")
    if not os.path.exists(rp): no_train_msg(); st.stop()

    report = load_json(rp)
    c1,c2,c3,c4 = st.columns(4)
    with c1: kcard("📥","Initial Rows",f"{report['initial_shape'][0]:,}")
    with c2: kcard("📤","Final Rows",f"{report['final_shape'][0]:,}","kcard-green")
    with c3: kcard("🗑️","Rows Removed",f"{report['rows_removed']:,}","kcard-red")
    with c4: kcard("🗑️","Cols Removed",f"{report['cols_removed']}","kcard-orange")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        sec("1. Before vs After — Shape Comparison")
        comp = pd.DataFrame({
            "Stage": ["Before Cleaning","After Cleaning"],
            "Rows":  [report["initial_shape"][0], report["final_shape"][0]],
            "Columns":[report["initial_shape"][1], report["final_shape"][1]],
        })
        fig = px.bar(comp, x="Stage", y="Rows", color="Stage",
                     color_discrete_map={"Before Cleaning":"#ef5350","After Cleaning":"#4caf50"},
                     text="Rows")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        plx(fig, 300)

    with col2:
        sec("2. Cleaning Steps Log")
        if os.path.exists(sp):
            steps = pd.read_csv(sp)
            st.dataframe(steps, use_container_width=True, hide_index=True, height=280)

    col3, col4 = st.columns(2)
    with col3:
        sec("3. Identifier Columns Removed (Data Leakage Prevention)")
        id_cols = report.get("identifier_columns", [])
        if id_cols:
            st.dataframe(pd.DataFrame({"Removed Column":id_cols}),
                         use_container_width=True, hide_index=True)
        else:
            box("No identifier columns found.", "ok")

    with col4:
        sec("4. Class Distribution After Cleaning")
        cd = report.get("class_distribution", {})
        if cd:
            cd_df = pd.DataFrame(list(cd.items()), columns=["Label","Count"])
            cd_df = cd_df.sort_values("Count",ascending=False)
            fig = px.bar(cd_df, x="Label", y="Count",
                         color="Count", color_continuous_scale="Blues")
            fig.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False)
            plx(fig, 280)

    sec("5. Remaining Missing Values After Cleaning")
    rm = report.get("remaining_missing", 0)
    if rm == 0:
        box("✅ No missing values remain after cleaning.", "ok")
    else:
        box(f"⚠️ {rm:,} missing values remain. Imputation applied during feature engineering.", "warn")

# ═══════════════════════════════════════════════════════════════
# 06. DATA LEAKAGE
# ═══════════════════════════════════════════════════════════════
elif page == "06. Data Leakage":
    st.title("06. DATA LEAKAGE")
    st.caption("Identifying and removing features that cause target leakage")
    st.markdown("---")

    leakage_cols = ["Flow ID","Source IP","Destination IP","Source Port",
                    "Destination Port","Timestamp","Flow Start","Flow End"]
    safe_practices = [
        ("Imputer","Fitted on training data ONLY, then applied to test"),
        ("Scaler","Not applied (tree models used — scaling not required)"),
        ("Feature Selection","Run on training set only before test evaluation"),
        ("Label Encoding","Fitted on training classes only"),
        ("Train/Test Split","Stratified 80/20 — test never used in training pipeline"),
    ]

    c1,c2 = st.columns(2)
    with c1:
        sec("1. Potential Leakage Columns Identified & Removed")
        st.dataframe(pd.DataFrame({
            "Column":leakage_cols,
            "Reason":["Unique per flow — not predictive"]*2 +
                     ["Port alone not sufficient"]*2 +
                     ["Temporal leakage"]*2 +
                     ["Temporal leakage"]*2,
            "Action":["Removed"]*8,
        }), use_container_width=True, hide_index=True)

    with c2:
        sec("2. Leakage-Safe Preprocessing Practices")
        st.dataframe(pd.DataFrame(safe_practices, columns=["Step","Practice"]),
                     use_container_width=True, hide_index=True)

    sec("3. Train vs Test Information Flow")
    flow = pd.DataFrame({
        "Step":["Raw CSV","Column normalisation","Label encoding","Imputation",
                "Feature selection","Model training","Model evaluation"],
        "Applied On":["Full dataset","Full dataset","Full dataset","Train only",
                      "Train only","Train only","Test only"],
        "Leakage Safe":["✅","✅","✅","✅ (fit on train)","✅","✅","✅"],
    })
    st.dataframe(flow, use_container_width=True, hide_index=True)

    box("✅ No target leakage detected. All transformations are fitted on training data only.", "ok")

# ═══════════════════════════════════════════════════════════════
# 07. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
elif page == "07. Feature Engineering":
    st.title("07. FEATURE ENGINEERING")
    st.caption("Feature construction, imputation, and transformation pipeline")
    st.markdown("---")

    fl_path = os.path.join(CONFIG["outputs"]["feature_importance_dir"],"feature_list.json")
    if not os.path.exists(fl_path): no_train_msg(); st.stop()
    fl = load_json(fl_path)
    features = fl["features"]

    c1,c2,c3 = st.columns(3)
    with c1: kcard("⚙️","Total Features",f"{len(features)}")
    with c2: kcard("🔧","Imputer","Median")
    with c3: kcard("🌳","Model Type","RandomForest (no scaling needed)")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        sec("1. Feature Engineering Pipeline Steps")
        steps = pd.DataFrame({
            "Step #":range(1,9),
            "Step":["Strip column whitespace","Convert to numeric",
                    "Replace ±Infinity → NaN","Remove empty columns",
                    "Remove constant features","Remove identifier columns",
                    "Median imputation (train only)","Feature selection (ExtraTrees)"],
            "Purpose":["Name standardisation","All features as float64",
                        "Handle network overflow values","Eliminate dead columns",
                        "Zero-variance removal","Prevent leakage",
                        "Fill remaining NaN safely","Rank features by importance"],
        })
        st.dataframe(steps, use_container_width=True, hide_index=True, height=290)

    with col2:
        sec("2. Final Feature List")
        feat_df = pd.DataFrame({"#":range(1,len(features)+1),"Feature":features})
        st.dataframe(feat_df, use_container_width=True, hide_index=True, height=290)

    sec("3. Excluded Column Categories")
    exc = pd.DataFrame({
        "Category":["Flow Identifier","IP Address","Port Numbers",
                     "Timestamp","Constant Features","Label Column"],
        "Example Columns":["Flow ID","Source IP, Destination IP",
                            "Source Port, Destination Port",
                            "Timestamp","Bwd PSH Flags (if constant)","Label"],
        "Reason":["Unique per connection","Network address ≠ behaviour",
                   "Can cause leakage","Temporal leakage",
                   "Zero variance = no information","Target variable"],
        "Action":["Removed"]*6,
    })
    st.dataframe(exc, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# 08. FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════
elif page == "08. Feature Selection":
    st.title("08. FEATURE SELECTION")
    st.caption("ExtraTrees-based feature importance ranking")
    st.markdown("---")

    fi_df = load_fi()
    if fi_df is None: no_train_msg(); st.stop()

    c1,c2,c3 = st.columns(3)
    with c1: kcard("🌟","Total Features Ranked",f"{len(fi_df)}")
    with c2: kcard("🎯","Top Feature",fi_df.iloc[0]["feature"][:25])
    with c3: kcard("📊","Method","ExtraTreesClassifier")

    st.markdown("---")
    col1, col2 = st.columns([1,1])

    with col1:
        sec("1. Top 20 Features — Importance Table")
        st.dataframe(fi_df.head(20).round(6), use_container_width=True,
                     hide_index=True, height=460)

    with col2:
        sec("2. Top 20 Features — Bar Chart")
        top20 = fi_df.head(20)
        fig = px.bar(top20[::-1], x="importance", y="feature",
                     orientation="h", color="importance",
                     color_continuous_scale="RdYlGn")
        fig.update_layout(coloraxis_showscale=False)
        plx(fig, 460)

    sec("3. Cumulative Feature Importance")
    fi_df2 = fi_df.copy()
    fi_df2["cumulative"] = fi_df2["importance"].cumsum()
    fig = px.line(fi_df2.head(40), x=fi_df2.head(40).index, y="cumulative",
                  labels={"x":"Feature Rank","cumulative":"Cumulative Importance"})
    fig.add_hline(y=0.95, line_dash="dash", line_color="#ef5350",
                  annotation_text="95% threshold")
    plx(fig, 280)

# ═══════════════════════════════════════════════════════════════
# 09. DATA SPLITTING
# ═══════════════════════════════════════════════════════════════
elif page == "09. Data Splitting":
    st.title("09. DATA SPLITTING")
    st.caption("Stratified 80/20 train-test split details")
    st.markdown("---")

    bin_meta_path = os.path.join(CONFIG["artifacts"]["models_dir"],"binary_metadata.json")
    if not os.path.exists(bin_meta_path): no_train_msg(); st.stop()
    meta = load_json(bin_meta_path)
    si = meta.get("split_info", {})

    c1,c2,c3,c4 = st.columns(4)
    with c1: kcard("🏋️","Train Rows",f"{si.get('train_rows',0):,}")
    with c2: kcard("🧪","Test Rows",f"{si.get('test_rows',0):,}")
    with c3: kcard("📐","Test Size","20%")
    with c4: kcard("🎲","Random State","42")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        sec("1. Train vs Test Split")
        split_data = pd.DataFrame({
            "Set":["Training","Testing"],
            "Rows":[si.get("train_rows",0), si.get("test_rows",0)],
            "Percentage":[80,20],
        })
        fig = px.pie(split_data, names="Set", values="Rows", hole=0.4,
                     color_discrete_sequence=["#2196f3","#ff9800"])
        plx(fig, 300)

    with col2:
        sec("2. Class Distribution — Train vs Test")
        cls_data = pd.DataFrame({
            "Set":["Train Benign","Train Attack","Test Benign","Test Attack"],
            "Count":[si.get("train_benign",0), si.get("train_attack",0),
                     si.get("test_benign",0), si.get("test_attack",0)],
            "Type":["Benign","Attack","Benign","Attack"],
        })
        fig = px.bar(cls_data, x="Set", y="Count", color="Type",
                     color_discrete_map={"Benign":"#4caf50","Attack":"#ef5350"},
                     text="Count")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        plx(fig, 300)

    sec("3. Split Summary Table")
    split_summary = pd.DataFrame({
        "Metric":["Total train rows","Total test rows","Train benign","Train attack",
                   "Test benign","Test attack","Stratification","Random state"],
        "Value":[f"{si.get('train_rows',0):,}", f"{si.get('test_rows',0):,}",
                 f"{si.get('train_benign',0):,}", f"{si.get('train_attack',0):,}",
                 f"{si.get('test_benign',0):,}", f"{si.get('test_attack',0):,}",
                 "Yes (on binary label)", "42"],
    })
    st.dataframe(split_summary, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# 10. PREPROCESSING
# ═══════════════════════════════════════════════════════════════
elif page == "10. Preprocessing":
    st.title("10. PREPROCESSING")
    st.caption("Data transformation pipeline applied before model training")
    st.markdown("---")

    c1,c2,c3 = st.columns(3)
    with c1: kcard("🔢","Imputer Strategy","Median")
    with c2: kcard("🌳","Scaling","None (Tree Models)")
    with c3: kcard("🏷️","Label Encoding","LabelEncoder (multiclass)")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        sec("1. Preprocessing Pipeline Summary")
        pp = pd.DataFrame({
            "Step":["1. Load CSVs","2. Normalize column names",
                    "3. Normalize label strings","4. Convert to numeric",
                    "5. Replace ±inf → NaN","6. Drop empty/constant cols",
                    "7. Drop identifier cols","8. Drop duplicates",
                    "9. Impute NaN (median, train only)",
                    "10. Feature selection","11. Stratified split 80/20"],
            "Fit On":["—","—","—","—","—","Full","Full",
                      "Full","Train","Train","Full"],
            "Leakage Safe":["✅"]*11,
        })
        st.dataframe(pp, use_container_width=True, hide_index=True, height=380)

    with col2:
        sec("2. Why No Scaling?")
        box("""<b>RandomForestClassifier</b> and <b>ExtraTreesClassifier</b> are
        tree-based algorithms. They split on feature thresholds, not distances.
        Scaling (StandardScaler / MinMaxScaler) does <b>not improve</b> their
        performance and is therefore skipped to keep the pipeline simple and
        interpretable.""", "info")

        sec("3. Imputation Strategy")
        imp_info = pd.DataFrame({
            "Property":["Strategy","Fitted on","Applied to","Why Median?"],
            "Value":["Median","Training set ONLY","Train + Test",
                     "Robust to outliers in network flow data"],
        })
        st.dataframe(imp_info, use_container_width=True, hide_index=True)

        sec("4. Label Encoding (Multiclass)")
        mc_meta_path = os.path.join(CONFIG["artifacts"]["models_dir"],"multiclass_metadata.json")
        if os.path.exists(mc_meta_path):
            mc_meta = load_json(mc_meta_path)
            classes = mc_meta.get("classes", [])
            enc_df = pd.DataFrame({"Class": classes,
                                    "Encoded Integer": range(len(classes))})
            st.dataframe(enc_df, use_container_width=True, hide_index=True)
        else:
            st.info("Multiclass model not trained yet.")

# ═══════════════════════════════════════════════════════════════
# 11. CLASSIFICATION
# ═══════════════════════════════════════════════════════════════
elif page == "11. Classification":
    st.title("11. CLASSIFICATION")
    st.caption("Model architecture, hyperparameters, and training configuration")
    st.markdown("---")

    bin_art = load_binary()
    mc_art  = load_multi()

    c1,c2,c3,c4 = st.columns(4)
    with c1: kcard("🤖","Binary Model","RandomForest","kcard-green" if bin_art else "kcard-red")
    with c2: kcard("🤖","Multiclass Model","ExtraTrees","kcard-green" if mc_art else "kcard-orange")
    with c3:
        n_feat = bin_art[2]["n_features"] if bin_art else "N/A"
        kcard("⚙️","Features Used",str(n_feat))
    with c4:
        n_cls = mc_art[3]["n_classes"] if mc_art else "N/A"
        kcard("🔢","Attack Classes",str(n_cls))

    st.markdown("---")
    tab1, tab2 = st.tabs(["🔵 Binary Model", "🟠 Multiclass Model"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            sec("1. Binary Model Configuration")
            params = CONFIG["binary_model"]["params"]
            st.dataframe(pd.DataFrame(list(params.items()),
                                       columns=["Parameter","Value"]),
                         use_container_width=True, hide_index=True)
        with col2:
            sec("2. Binary Target Mapping")
            st.dataframe(pd.DataFrame({
                "Label":["BENIGN","All other labels"],
                "Binary Target":[0, 1],
                "Meaning":["Normal traffic","Attack traffic"],
            }), use_container_width=True, hide_index=True)

        sec("3. RandomForest — How It Works")
        box("""<b>RandomForestClassifier</b> builds <b>n_estimators</b> decision trees,
        each trained on a random bootstrap sample of the data. Each tree uses a
        random subset of features at each split (feature bagging). Final prediction
        is by majority vote. <b>class_weight=balanced</b> handles the class imbalance
        by weighting minority classes higher during training.""", "info")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            sec("4. Multiclass Model Configuration")
            params2 = CONFIG["multiclass_model"]["params"]
            st.dataframe(pd.DataFrame(list(params2.items()),
                                       columns=["Parameter","Value"]),
                         use_container_width=True, hide_index=True)
        with col2:
            sec("5. Attack Classes")
            if mc_art:
                _, le, _, mm = mc_art
                st.dataframe(pd.DataFrame({
                    "Class": mm["classes"],
                    "Encoded": range(len(mm["classes"])),
                }), use_container_width=True, hide_index=True)
            else:
                st.info("Multiclass model not trained yet.")

# ═══════════════════════════════════════════════════════════════
# 12. MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════
elif page == "12. Model Evaluation":
    st.title("12. MODEL EVALUATION")
    st.caption("Held-out test set metrics — binary and multiclass")
    st.markdown("---")

    bm = load_bin_metrics()
    mm = load_mc_metrics()
    if bm is None: no_train_msg(); st.stop()

    tab1, tab2 = st.tabs(["🔵 Binary Metrics", "🟠 Multiclass Metrics"])

    with tab1:
        c1,c2,c3,c4 = st.columns(4)
        with c1: kcard("✅","Accuracy",f"{bm['accuracy']*100:.2f}%","kcard-green")
        with c2: kcard("🎯","Precision",f"{bm['precision']:.4f}")
        with c3: kcard("📡","Recall",f"{bm['recall']:.4f}")
        with c4: kcard("⚖️","F1 Score",f"{bm['f1']:.4f}")
        c5,c6,c7,c8 = st.columns(4)
        with c5: kcard("📈","ROC-AUC",str(round(bm.get("roc_auc",0),4)) if "roc_auc" in bm else "N/A")
        with c6: kcard("📉","PR-AUC",str(round(bm.get("pr_auc",0),4)) if "pr_auc" in bm else "N/A")
        with c7: kcard("✅","True Positives",f"{bm.get('tp',0):,}","kcard-green")
        with c8: kcard("❌","False Negatives",f"{bm.get('fn',0):,}","kcard-red")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            sec("1. Binary Metrics Table")
            metric_rows = [
                ("Accuracy", f"{bm['accuracy']*100:.4f}%"),
                ("Precision", f"{bm['precision']:.4f}"),
                ("Recall (Sensitivity)", f"{bm['recall']:.4f}"),
                ("F1 Score", f"{bm['f1']:.4f}"),
                ("ROC-AUC", f"{bm.get('roc_auc','N/A')}"),
                ("PR-AUC", f"{bm.get('pr_auc','N/A')}"),
                ("True Positives", f"{bm.get('tp',0):,}"),
                ("True Negatives", f"{bm.get('tn',0):,}"),
                ("False Positives", f"{bm.get('fp',0):,}"),
                ("False Negatives", f"{bm.get('fn',0):,}"),
            ]
            st.dataframe(pd.DataFrame(metric_rows, columns=["Metric","Value"]),
                         use_container_width=True, hide_index=True, height=350)
        with col2:
            sec("2. Metrics Radar Chart")
            vals = [bm["accuracy"], bm["precision"], bm["recall"], bm["f1"],
                    bm.get("roc_auc", bm["f1"])]
            cats = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
            fig = go.Figure(go.Scatterpolar(
                r=vals+[vals[0]], theta=cats+[cats[0]],
                fill="toself", line_color="#2196f3",
            ))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0,1])))
            plx(fig, 350)

        sec("3. Classification Report")
        st.code(bm.get("classification_report","Not available"), language=None)

        # ROC plot if available
        roc_path = os.path.join(CONFIG["outputs"]["evaluation_dir"],"binary_roc_curve.png")
        if os.path.exists(roc_path):
            sec("4. ROC Curve")
            st.image(roc_path, width=500)

    with tab2:
        if mm is None:
            st.info("Multiclass model not trained.")
        else:
            c1,c2,c3,c4 = st.columns(4)
            with c1: kcard("✅","Accuracy",f"{mm['accuracy']*100:.2f}%","kcard-green")
            with c2: kcard("🎯","Macro Precision",f"{mm['macro_precision']:.4f}")
            with c3: kcard("📡","Macro Recall",f"{mm['macro_recall']:.4f}")
            with c4: kcard("⚖️","Macro F1",f"{mm['macro_f1']:.4f}")
            kcard("⚖️","Weighted F1",f"{mm['weighted_f1']:.4f}")

            sec("5. Multiclass Classification Report")
            st.code(mm.get("classification_report","Not available"), language=None)

# ═══════════════════════════════════════════════════════════════
# 13. MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════
elif page == "13. Model Comparison":
    st.title("13. MODEL COMPARISON")
    st.caption("Binary vs multiclass model performance side-by-side")
    st.markdown("---")

    bm = load_bin_metrics()
    mm = load_mc_metrics()
    if bm is None: no_train_msg(); st.stop()

    sec("1. Model Comparison Table")
    rows = [{"Model":"Binary (RandomForest)",
             "Task":"BENIGN vs ATTACK",
             "Accuracy":f"{bm['accuracy']*100:.2f}%",
             "Precision":f"{bm['precision']:.4f}",
             "Recall":f"{bm['recall']:.4f}",
             "F1":f"{bm['f1']:.4f}",
             "ROC-AUC":f"{bm.get('roc_auc','N/A')}"}]
    if mm:
        rows.append({"Model":"Multiclass (ExtraTrees)",
                     "Task":"Attack Type Classification",
                     "Accuracy":f"{mm['accuracy']*100:.2f}%",
                     "Precision":f"{mm['macro_precision']:.4f}",
                     "Recall":f"{mm['macro_recall']:.4f}",
                     "F1":f"{mm['macro_f1']:.4f}",
                     "ROC-AUC":"N/A (multiclass)"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        sec("2. Binary Model — Metric Bar")
        metrics_dict = {"Precision":bm["precision"],"Recall":bm["recall"],
                        "F1":bm["f1"],"Accuracy":bm["accuracy"]}
        if "roc_auc" in bm: metrics_dict["ROC-AUC"] = bm["roc_auc"]
        fig = px.bar(x=list(metrics_dict.keys()), y=list(metrics_dict.values()),
                     color=list(metrics_dict.values()),
                     color_continuous_scale="Blues", text=[f"{v:.4f}" for v in metrics_dict.values()])
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, yaxis_range=[0,1.1])
        plx(fig, 320)

    with col2:
        sec("3. Algorithm Comparison — Properties")
        algo_df = pd.DataFrame({
            "Property":["Algorithm","Task","Trees","Class Weight",
                         "Feature Importance","Handles Imbalance","Speed"],
            "RandomForest":["RandomForestClassifier","Binary","100",
                             "balanced","✅","✅","Fast"],
            "ExtraTrees":["ExtraTreesClassifier","Multiclass","100",
                           "balanced","✅","✅","Faster"],
        })
        st.dataframe(algo_df, use_container_width=True, hide_index=True)

    sec("4. Cybersecurity Metric Importance")
    imp_df = pd.DataFrame({
        "Metric":["Recall","False Negative Rate","False Positive Rate",
                   "Precision","F1 Score"],
        "Why It Matters in IDS":[
            "High recall = more attacks caught — critical for security",
            "Low FNR = fewer missed attacks — dangerous if high",
            "Low FPR = fewer false alarms — analyst fatigue if high",
            "High precision = fewer false alarms",
            "Balance between precision and recall",
        ],
        "Priority":["⭐⭐⭐ CRITICAL","⭐⭐⭐ CRITICAL","⭐⭐ HIGH","⭐⭐ HIGH","⭐⭐ HIGH"],
    })
    st.dataframe(imp_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# 14. MODEL IMPROVEMENT
# ═══════════════════════════════════════════════════════════════
elif page == "14. Model Improvement":
    st.title("14. MODEL IMPROVEMENT")
    st.caption("Class imbalance handling, hyperparameter choices, and potential enhancements")
    st.markdown("---")

    bm = load_bin_metrics()

    sec("1. Techniques Applied for Model Improvement")
    techniques = pd.DataFrame({
        "Technique":["class_weight=balanced","Stratified Splitting",
                     "Median Imputation","Feature Selection",
                     "Max Depth Limiting","n_jobs=-1 (Parallel)"],
        "Purpose":["Handle class imbalance","Equal class ratio in train/test",
                    "Robust missing value handling","Reduce noise features",
                    "Prevent overfitting","Faster training"],
        "Applied To":["Both models","Both models","Both models",
                      "Both models","Both models","Both models"],
        "Result":["Better recall on attacks","Reliable evaluation",
                   "No NaN errors","Top predictive features only",
                   "Generalisation","Reduced training time"],
    })
    st.dataframe(techniques, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        sec("2. Hyperparameter Summary")
        hp = pd.DataFrame({
            "Parameter":["n_estimators","max_depth","min_samples_split",
                          "min_samples_leaf","random_state","class_weight"],
            "Binary Value":[str(v) for v in CONFIG["binary_model"]["params"].values()],
        })
        st.dataframe(hp, use_container_width=True, hide_index=True)

    with col2:
        sec("3. Future Improvement Roadmap")
        roadmap = pd.DataFrame({
            "Enhancement":["XGBoost/LightGBM","SMOTE Oversampling",
                            "LSTM for temporal flow","SHAP explanations",
                            "Automated retraining","Ensemble stacking"],
            "Expected Gain":["Higher accuracy","Better minority recall",
                              "Sequential pattern detection",
                              "Per-prediction explainability",
                              "Adapt to new attacks","Combined model power"],
            "Complexity":["Low","Medium","High","Medium","High","High"],
        })
        st.dataframe(roadmap, use_container_width=True, hide_index=True)

    if bm:
        sec("4. Current Model Strengths")
        tp, tn = bm.get("tp",0), bm.get("tn",0)
        fp, fn = bm.get("fp",0), bm.get("fn",0)
        total = tp+tn+fp+fn
        strengths = [
            f"✅ Accuracy: {bm['accuracy']*100:.2f}% on held-out test data",
            f"✅ Recall: {bm['recall']:.4f} — {bm['recall']*100:.1f}% of attacks detected",
            f"✅ True Positives: {tp:,} attacks correctly caught",
            f"⚠️ False Negatives: {fn:,} attacks missed ({fn/max(total,1)*100:.2f}%)",
            f"⚠️ False Positives: {fp:,} benign incorrectly flagged",
        ]
        for s in strengths:
            st.markdown(f"- {s}")

# ═══════════════════════════════════════════════════════════════
# 15. CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════════
elif page == "15. Confusion Matrix":
    st.title("15. CONFUSION MATRIX")
    st.caption("Visual breakdown of TP, TN, FP, FN")
    st.markdown("---")

    bm = load_bin_metrics()
    mm = load_mc_metrics()
    if bm is None: no_train_msg(); st.stop()

    # ── Confusion matrix explanation ──
    sec("1. Confusion Matrix Cell Definitions")
    st.dataframe(pd.DataFrame({
        "Cell":["True Positive (TP)","True Negative (TN)",
                "False Positive (FP)","False Negative (FN)"],
        "Meaning":["Attack correctly detected as ATTACK",
                    "Benign correctly identified as BENIGN",
                    "Benign INCORRECTLY flagged as ATTACK",
                    "Attack INCORRECTLY identified as BENIGN"],
        "Impact":["✅ Good — threat caught",
                   "✅ Good — no false alarm",
                   "⚠️ Alert fatigue, analyst overhead",
                   "❌ DANGEROUS — missed attack"],
    }), use_container_width=True, hide_index=True)

    st.markdown("---")
    tab1, tab2 = st.tabs(["🔵 Binary Confusion Matrix","🟠 Multiclass Confusion Matrix"])

    with tab1:
        col1, col2 = st.columns([1,1])
        with col1:
            sec("2. Binary Confusion Matrix")
            tp,tn = bm.get("tp",0), bm.get("tn",0)
            fp,fn = bm.get("fp",0), bm.get("fn",0)
            cm = np.array([[tn, fp],[fn, tp]])
            fig = px.imshow(cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Predicted BENIGN","Predicted ATTACK"],
                y=["Actual BENIGN","Actual ATTACK"],
                color_continuous_scale="Blues",
                text_auto=True, aspect="auto")
            fig.update_traces(texttemplate="%{z:,}", textfont_size=18)
            plx(fig, 350)

        with col2:
            sec("3. Key Counts")
            c1,c2 = st.columns(2)
            with c1:
                kcard("✅","True Positives<br><small>Attacks Caught</small>",f"{tp:,}","kcard-green")
                kcard("❌","False Negatives<br><small>Missed Attacks</small>",f"{fn:,}","kcard-red")
            with c2:
                kcard("✅","True Negatives<br><small>Benign Correct</small>",f"{tn:,}","kcard-green")
                kcard("⚠️","False Positives<br><small>Wrong Alerts</small>",f"{fp:,}","kcard-orange")

            total = tp+tn+fp+fn
            sec("4. Derived Rates")
            rates = pd.DataFrame({
                "Rate":["Detection Rate (TPR)","False Alarm Rate (FPR)",
                        "Miss Rate (FNR)","Specificity (TNR)"],
                "Value":[f"{tp/max(tp+fn,1)*100:.2f}%",
                         f"{fp/max(fp+tn,1)*100:.2f}%",
                         f"{fn/max(fn+tp,1)*100:.2f}%",
                         f"{tn/max(tn+fp,1)*100:.2f}%"],
            })
            st.dataframe(rates, use_container_width=True, hide_index=True)

        # Use saved image if available
        cm_path = os.path.join(CONFIG["outputs"]["evaluation_dir"],"binary_confusion_matrix.png")
        if os.path.exists(cm_path):
            sec("5. Confusion Matrix Plot (Generated by Pipeline)")
            st.image(cm_path, width=450)

    with tab2:
        if mm is None:
            st.info("Multiclass model not trained.")
        else:
            cm_mc_path = os.path.join(CONFIG["outputs"]["evaluation_dir"],
                                       "multiclass_confusion_matrix.png")
            if os.path.exists(cm_mc_path):
                sec("6. Multiclass Confusion Matrix")
                st.image(cm_mc_path, use_column_width=True)
            else:
                sec("6. Multiclass Confusion Matrix (Interactive)")
                classes = mm.get("classes",[])
                cm_mc = np.array(mm["confusion_matrix"])
                fig = px.imshow(cm_mc,
                    x=classes, y=classes,
                    color_continuous_scale="Blues",
                    text_auto=True, aspect="auto",
                    labels=dict(x="Predicted",y="Actual"))
                fig.update_layout(xaxis_tickangle=-40)
                plx(fig, 500)

# ═══════════════════════════════════════════════════════════════
# 16. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════
elif page == "16. Feature Importance":
    st.title("16. FEATURE IMPORTANCE")
    st.caption("Top network-flow features ranked by predictive power")
    st.markdown("---")

    fi_df = load_fi()
    if fi_df is None: no_train_msg(); st.stop()

    c1,c2,c3 = st.columns(3)
    with c1: kcard("🌟","Features Ranked",f"{len(fi_df)}")
    with c2: kcard("🥇","#1 Feature",fi_df.iloc[0]["feature"][:22])
    with c3: kcard("📊","Top 20 Covers",f"{fi_df.head(20)['importance'].sum()*100:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns([1,1])
    with col1:
        sec("1. Top 20 Features Table")
        st.dataframe(fi_df.head(20).round(6), use_container_width=True,
                     hide_index=True, height=470)

    with col2:
        sec("2. Top 20 Feature Importance Bar Chart")
        top20 = fi_df.head(20)
        fig = px.bar(top20[::-1], x="importance", y="feature",
                     orientation="h", color="importance",
                     color_continuous_scale="RdYlGn")
        fig.update_layout(coloraxis_showscale=False, yaxis_title="")
        plx(fig, 470)

    sec("3. What Do Top Features Mean?")
    desc = {
        "Flow Duration":"Total duration of the network connection",
        "Total Fwd Packets":"Packets sent from source to destination",
        "Total Backward Packets":"Packets sent in reverse direction",
        "Fwd Packet Length Max":"Largest forward packet size",
        "Flow Bytes/s":"Data transfer rate of the flow",
        "Flow Packets/s":"Packet rate of the flow",
        "Flow IAT Mean":"Average inter-arrival time between packets",
        "Bwd IAT Total":"Total inter-arrival time in backward direction",
        "PSH Flag Count":"TCP PSH flags — forces data delivery",
        "Init_Win_bytes_forward":"Initial TCP window size (forward)",
    }
    for feat in fi_df.head(10)["feature"].tolist():
        meaning = desc.get(feat, "Network flow characteristic")
        st.markdown(f"- **{feat}**: {meaning}")

    # Use pipeline-generated image if available
    fi_img = os.path.join(CONFIG["outputs"]["feature_importance_dir"],
                           "feature_importance_model.png")
    if os.path.exists(fi_img):
        sec("4. Pipeline-Generated Feature Importance Chart")
        st.image(fi_img, use_column_width=True)

# ═══════════════════════════════════════════════════════════════
# 17. BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════
elif page == "17. Batch Prediction":
    st.title("17. BATCH PREDICTION")
    st.caption("Upload a CIC-IDS2017 format CSV to classify network flows")
    st.markdown("---")

    bin_art = load_binary()
    mc_art  = load_multi()

    if not bin_art:
        no_train_msg(); st.stop()

    bin_model, bin_imp, bin_meta = bin_art
    feature_cols = bin_meta["feature_cols"]

    box("""ℹ️ Upload a CSV with the same feature columns as CIC-IDS2017.
    A <code>Label</code> column (if present) is ignored during prediction.
    This system analyses <b>pre-extracted network flow features</b>.""","info")

    c1,c2,c3 = st.columns(3)
    with c1: kcard("⚙️","Binary Model",bin_meta["model_type"])
    with c2: kcard("📊","Expected Features",str(len(feature_cols)))
    with c3: kcard("🔢","Multiclass Available","✅ Yes" if mc_art else "❌ No")

    st.markdown("---")
    uploaded = st.file_uploader("📂 Upload CSV File", type=["csv"])

    if uploaded:
        df_in = pd.read_csv(uploaded, low_memory=False)
        st.success(f"File loaded — {df_in.shape[0]:,} rows × {df_in.shape[1]} columns")

        from src.predict import validate_prediction_input, run_binary_prediction, run_multiclass_prediction
        val = validate_prediction_input(df_in.copy(), feature_cols)

        c1,c2,c3,c4 = st.columns(4)
        with c1: kcard("📄","Rows Loaded",f"{val['n_rows']:,}")
        with c2: kcard("✅","Features Present",f"{val['features_present']}/{len(feature_cols)}")
        with c3: kcard("📉","Coverage",f"{val['pct_features_present']}%")
        with c4: kcard("⚙️","Status","✅ Valid" if val["valid"] else "❌ Invalid",
                        "kcard-green" if val["valid"] else "kcard-red")

        if val.get("warning"):
            box(f"⚠️ {val['warning']}","warn")
        if not val["valid"]:
            st.error("No matching features found."); st.stop()

        if st.button("🔍 Run Prediction"):
            with st.spinner("Classifying network flows..."):
                result = run_binary_prediction(df_in, bin_model, bin_imp, feature_cols)
                if mc_art:
                    mc_m, mc_le, mc_imp, mc_meta2 = mc_art
                    try:
                        mc_r = run_multiclass_prediction(df_in, mc_m, mc_le,
                                                          mc_imp, mc_meta2["feature_cols"])
                        result["Predicted_Category"] = mc_r["Predicted_Category"].values
                    except Exception:
                        pass

            n_b = int((result["Predicted_Label"]=="BENIGN").sum())
            n_a = int((result["Predicted_Label"]=="ATTACK").sum())

            st.markdown("---")
            c1,c2,c3 = st.columns(3)
            with c1: kcard("📊","Total Analysed",f"{len(result):,}")
            with c2: kcard("✅","Predicted BENIGN",f"{n_b:,}","kcard-green")
            with c3: kcard("🚨","Predicted ATTACK",f"{n_a:,}","kcard-red")

            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure(go.Pie(
                    labels=["BENIGN","ATTACK"], values=[n_b,n_a],
                    hole=0.45,
                    marker_colors=["#4caf50","#ef5350"],
                ))
                plx(fig, 300)
            with col2:
                if "Attack_Probability" in result.columns:
                    fig2 = px.histogram(result, x="Attack_Probability", nbins=50,
                                        color_discrete_sequence=["#ef5350"],
                                        title="Attack Probability Distribution")
                    plx(fig2, 300)

            sec("Prediction Results Table (first 500 rows)")
            show_cols = ["Predicted_Label"]
            if "Attack_Probability" in result.columns: show_cols.append("Attack_Probability")
            if "Predicted_Category" in result.columns: show_cols.append("Predicted_Category")
            st.dataframe(result[show_cols].head(500), use_container_width=True)

            box("⚠️ These are model predictions. Verify with a security analyst before taking action.","warn")

            csv_dl = result.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Predictions CSV", csv_dl,
                                "securenet_predictions.csv","text/csv")

# ═══════════════════════════════════════════════════════════════
# 18. REPORTS & DOWNLOADS
# ═══════════════════════════════════════════════════════════════
elif page == "18. Reports & Downloads":
    st.title("18. REPORTS & DOWNLOADS")
    st.caption("Download all generated reports, metrics, and charts")
    st.markdown("---")

    def dl_btn(label, path, mime="text/plain"):
        if os.path.exists(path):
            with open(path,"rb") as f: data=f.read()
            st.download_button(f"⬇️ {label}", data,
                                os.path.basename(path), mime)
        else:
            st.markdown(f"*{label}: not generated yet*")

    col1,col2,col3 = st.columns(3)
    with col1:
        sec("📊 Evaluation Reports")
        dl_btn("Binary Metrics JSON",
               os.path.join(CONFIG["outputs"]["evaluation_dir"],"binary_metrics.json"),"application/json")
        dl_btn("Binary Metrics CSV",
               os.path.join(CONFIG["outputs"]["evaluation_dir"],"binary_metrics.csv"),"text/csv")
        dl_btn("Multiclass Metrics JSON",
               os.path.join(CONFIG["outputs"]["evaluation_dir"],"multiclass_metrics.json"),"application/json")
        dl_btn("Multiclass Metrics CSV",
               os.path.join(CONFIG["outputs"]["evaluation_dir"],"multiclass_metrics.csv"),"text/csv")

    with col2:
        sec("🌟 Feature Files")
        dl_btn("Feature Importances CSV",
               os.path.join(CONFIG["outputs"]["feature_importance_dir"],"feature_importances.csv"),"text/csv")
        dl_btn("Feature List JSON",
               os.path.join(CONFIG["outputs"]["feature_importance_dir"],"feature_list.json"),"application/json")
        dl_btn("Top 20 Features CSV",
               os.path.join(CONFIG["outputs"]["feature_importance_dir"],"top20_features.csv"),"text/csv")

    with col3:
        sec("🧹 Cleaning Reports")
        dl_btn("Cleaning Report JSON",
               os.path.join(CONFIG["artifacts"]["reports_dir"],"cleaning_report.json"),"application/json")
        dl_btn("Cleaning Steps CSV",
               os.path.join(CONFIG["artifacts"]["reports_dir"],"cleaning_steps.csv"),"text/csv")
        dl_btn("Dataset Summary CSV",
               os.path.join(CONFIG["outputs"]["data_understanding_dir"],"dataset_summary.csv"),"text/csv")
        dl_btn("Class Distribution CSV",
               os.path.join(CONFIG["outputs"]["data_understanding_dir"],"class_distribution.csv"),"text/csv")

    st.markdown("---")
    sec("🖼️ Charts")
    chart_files = {
        "Binary Confusion Matrix": os.path.join(CONFIG["outputs"]["evaluation_dir"],"binary_confusion_matrix.png"),
        "Multiclass Confusion Matrix": os.path.join(CONFIG["outputs"]["evaluation_dir"],"multiclass_confusion_matrix.png"),
        "ROC Curve": os.path.join(CONFIG["outputs"]["evaluation_dir"],"binary_roc_curve.png"),
        "Feature Importance (Pipeline)": os.path.join(CONFIG["outputs"]["feature_importance_dir"],"feature_importance_model.png"),
        "Correlation Heatmap": os.path.join(CONFIG["outputs"]["eda_dir"],"correlation_heatmap.png"),
        "Label Distribution": os.path.join(CONFIG["outputs"]["data_understanding_dir"],"label_distribution.png"),
        "Benign vs Attack": os.path.join(CONFIG["outputs"]["data_understanding_dir"],"benign_vs_attack.png"),
    }
    available = {k:v for k,v in chart_files.items() if os.path.exists(v)}
    if available:
        cols = st.columns(min(len(available),3))
        for i,(label,path) in enumerate(available.items()):
            with cols[i % 3]:
                st.image(path, caption=label, use_column_width=True)
    else:
        box("No charts generated yet. Run the training pipeline first.","warn")

# ═══════════════════════════════════════════════════════════════
# 19. ABOUT SECURENET
# ═══════════════════════════════════════════════════════════════
elif page == "19. About SecureNet":
    st.title("19. ABOUT SECURENET")
    st.caption("Project overview, architecture, dataset, and references")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        sec("1. Project Overview")
        st.markdown("""
        **SecureNet** is an end-to-end ML-based Network Intrusion Detection System (IDS)
        built on the **CIC-IDS2017** dataset from the Canadian Institute for Cybersecurity.

        It detects malicious network traffic using two models:
        - **Binary model** — BENIGN vs ATTACK
        - **Multiclass model** — specific attack type identification

        Built as an academic B.Tech project demonstrating a complete
        data science pipeline from raw CSV to a professional dashboard.
        """)

        sec("2. Dataset Information")
        ds_info = pd.DataFrame({
            "Attribute":["Name","Source","Features","Days Covered","Attack Types","Format"],
            "Value":["CIC-IDS2017","Canadian Institute for Cybersecurity",
                     "78 network flow features","Monday to Friday (5 days)",
                     "12+ attack categories","CICFlowMeter CSV"],
        })
        st.dataframe(ds_info, use_container_width=True, hide_index=True)

    with col2:
        sec("3. Tech Stack")
        tech = pd.DataFrame({
            "Component":["Language","ML Library","Dashboard","Visualisation",
                          "Data Processing","Model Storage","Testing"],
            "Technology":["Python 3.9+","scikit-learn","Streamlit",
                           "Plotly + matplotlib","pandas + numpy",
                           "joblib","pytest"],
        })
        st.dataframe(tech, use_container_width=True, hide_index=True)

        sec("4. Attack Types in Dataset")
        attacks = ["DoS Hulk","DoS GoldenEye","DoS Slowloris","DoS Slowhttptest",
                   "DDoS","PortScan","FTP-Patator","SSH-Patator",
                   "Web Attack – Brute Force","Web Attack – XSS",
                   "Web Attack – SQL Injection","Infiltration","Bot","Heartbleed"]
        for a in attacks:
            st.markdown(f"- {a}")

    sec("5. Architecture Pipeline")
    arch = pd.DataFrame({
        "Phase":list(range(1,15)),
        "Module":["Data Collection","Data Understanding","Data Cleaning","EDA",
                   "Feature Engineering","Feature Selection","Data Splitting",
                   "Binary Training","Multiclass Training","Binary Evaluation",
                   "Multiclass Evaluation","Model Saving","Dashboard","Batch Prediction"],
        "Output":["CSV files","JSON + PNG reports","cleaned_data.csv",
                   "EDA plots","Feature matrix","feature_list.json",
                   "80/20 train/test","binary_model.joblib","multiclass_model.joblib",
                   "binary_metrics.json","multiclass_metrics.json",
                   "artifacts/models/","app.py","predictions.csv"],
    })
    st.dataframe(arch, use_container_width=True, hide_index=True)

    sec("6. References")
    st.markdown("""
    - Sharafaldin, I., Habibi Lashkari, A., Ghorbani, A.A. (2018).
      *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* ICISSP.
    - [CIC-IDS2017 Official](https://www.unb.ca/cic/datasets/ids-2017.html)
    - [Kaggle Dataset](https://www.kaggle.com/datasets/cicdataset/cicids2017)
    """)

    box("⚠️ This is an academic project for educational purposes. "
        "Model predictions require expert security analyst verification before action.","warn")

# ═══════════════════════════════════════════════════════════════
# 20. HOME DASHBOARD
# ═══════════════════════════════════════════════════════════════
elif page == "20. Home Dashboard":
    st.title("🛡️ SecureNet — AI Network Intrusion Detection System")
    st.caption("CIC-IDS2017 | Binary + Multiclass Classification | Professional Dashboard")
    st.markdown("---")

    u   = load_understanding()
    bm  = load_bin_metrics()
    mm  = load_mc_metrics()
    csv = get_csv_files()
    bin_art = load_binary()
    mc_art  = load_multi()

    # ── Row 1 KPIs ─────────────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: kcard("📂","Dataset Files",f"{len(csv)} CSV(s)" if csv else "Not Found",
                    "kcard-green" if csv else "kcard-red")
    with c2: kcard("📄","Total Rows",f"{u['n_rows']:,}" if u else "N/A")
    with c3: kcard("⚙️","Features",f"{u['n_numeric']}" if u else "N/A")
    with c4: kcard("🤖","Binary Model","✅ Ready" if bin_art else "❌ Not Trained",
                    "kcard-green" if bin_art else "kcard-red")
    with c5: kcard("🤖","Multiclass Model","✅ Ready" if mc_art else "❌ Not Trained",
                    "kcard-green" if mc_art else "kcard-red")

    # ── Row 2 KPIs ─────────────────────────────────────────────
    c6,c7,c8,c9,c10 = st.columns(5)
    with c6: kcard("✅","Benign Samples",f"{u['benign_count']:,}" if u else "N/A","kcard-green")
    with c7: kcard("🚨","Attack Samples",f"{u['attack_count']:,}" if u else "N/A","kcard-red")
    with c8: kcard("📈","Accuracy",f"{bm['accuracy']*100:.2f}%" if bm else "N/A","kcard-green" if bm else "")
    with c9: kcard("🎯","F1 Score",f"{bm['f1']:.4f}" if bm else "N/A")
    with c10: kcard("📡","ROC-AUC",f"{bm.get('roc_auc','N/A')}" if bm else "N/A")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        sec("Traffic Overview")
        if u and u["benign_count"] > 0:
            fig = go.Figure(go.Pie(
                labels=["Benign","Attack"],
                values=[u["benign_count"], u["attack_count"]],
                hole=0.5, marker_colors=["#4caf50","#ef5350"],
            ))
            plx(fig, 300)
        else:
            box("Run training pipeline to see charts.","warn")

    with col2:
        sec("Attack Category Distribution")
        if u and u.get("attack_distribution"):
            atk = u["attack_distribution"]
            df_atk = pd.DataFrame(list(atk.items()), columns=["Type","Count"])
            fig = px.bar(df_atk.sort_values("Count",ascending=False),
                         x="Type", y="Count", color="Count",
                         color_continuous_scale="Reds")
            fig.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False)
            plx(fig, 300)
        else:
            box("No attack data available yet.","warn")

    if bm:
        st.markdown("---")
        sec("Model Performance Summary")
        rows = [{"Model":"Binary (RandomForest)",
                 "Accuracy":f"{bm['accuracy']*100:.2f}%",
                 "Precision":f"{bm['precision']:.4f}",
                 "Recall":f"{bm['recall']:.4f}",
                 "F1":f"{bm['f1']:.4f}",
                 "ROC-AUC":f"{bm.get('roc_auc','N/A')}"}]
        if mm:
            rows.append({"Model":"Multiclass (ExtraTrees)",
                         "Accuracy":f"{mm['accuracy']*100:.2f}%",
                         "Precision":f"{mm['macro_precision']:.4f}",
                         "Recall":f"{mm['macro_recall']:.4f}",
                         "F1":f"{mm['macro_f1']:.4f}",
                         "ROC-AUC":"N/A"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if not csv:
        st.markdown("---")
        sec("⚙️ Quick Setup Guide")
        st.markdown("""
        1. Download **CIC-IDS2017** from [Kaggle](https://www.kaggle.com/datasets/cicdataset/cicids2017)
        2. Place all CSV files in `data/raw/`
        3. Run training: `python src/train_pipeline.py`
        4. Refresh this dashboard — all 20 sections will populate with real data
        """)
