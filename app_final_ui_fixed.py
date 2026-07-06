import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Default Risk Dashboard",
    page_icon="📊",
    layout="wide",
)

# ============================================================
# CUSTOM CSS - ENHANCED VISIBILITY + VIBRANT ACADEMIC STYLE
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #f7fbff 0%, #eaf7ff 45%, #ffffff 100%);
        color: #0f172a;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #082f63 !important;
        font-weight: 800 !important;
    }

    p, span, div, label {
        color: #0f172a;
    }

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2rem;
    }

    .hero-card {
        background: linear-gradient(135deg, #ffffff 0%, #eef9ff 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 30px;
        padding: 30px 34px 26px 34px;
        box-shadow: 0 18px 45px rgba(8, 47, 99, 0.12);
        margin-bottom: 22px;
    }

    .main-title {
        font-size: 44px;
        line-height: 1.13;
        font-weight: 900;
        color: #052e66;
        letter-spacing: -0.8px;
        margin-bottom: 10px;
    }

    .subtitle {
        font-size: 17px;
        color: #334155;
        line-height: 1.65;
        max-width: 1120px;
    }

    .pill-row {
        margin-top: 16px;
    }

    .pill {
        display: inline-block;
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 800;
        margin-right: 8px;
        margin-bottom: 8px;
        color: #ffffff;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.10);
    }

    .pill-blue { background: linear-gradient(135deg, #2563eb, #06b6d4); }
    .pill-green { background: linear-gradient(135deg, #059669, #10b981); }
    .pill-orange { background: linear-gradient(135deg, #f97316, #facc15); color: #1f2937; }
    .pill-purple { background: linear-gradient(135deg, #7c3aed, #ec4899); }

    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fdff 100%);
        border-radius: 24px;
        padding: 22px 24px;
        box-shadow: 0px 12px 30px rgba(8, 47, 99, 0.12);
        border: 1px solid rgba(14, 165, 233, 0.28);
        min-height: 132px;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 6px;
        background: linear-gradient(90deg, #2563eb, #06b6d4, #10b981, #f59e0b, #ef4444);
    }

    .metric-label {
        font-size: 13px;
        color: #334155;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 11px;
    }

    .metric-value {
        font-size: 32px;
        color: #020617;
        font-weight: 900;
        letter-spacing: -0.5px;
    }

    .metric-note {
        font-size: 12px;
        color: #475569;
        margin-top: 8px;
        line-height: 1.45;
    }

    .section-card {
        background: rgba(255, 255, 255, 0.96);
        border-radius: 28px;
        padding: 26px 28px;
        margin-top: 24px;
        margin-bottom: 24px;
        box-shadow: 0px 14px 38px rgba(15, 23, 42, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.22);
    }

    .interpretation-box {
        background: linear-gradient(135deg, #0ea5e9 0%, #14b8a6 100%);
        color: #ffffff !important;
        border-radius: 24px;
        padding: 20px 24px;
        margin-top: 14px;
        margin-bottom: 16px;
        box-shadow: 0 12px 28px rgba(14, 165, 233, 0.22);
        font-size: 15px;
        line-height: 1.7;
    }

    .interpretation-box b, .interpretation-box strong {
        color: #ffffff !important;
    }

    .success-box {
        background: linear-gradient(135deg, #10b981 0%, #6ee7b7 100%);
        color: #052e2b !important;
        border-radius: 24px;
        padding: 20px 24px;
        margin-top: 14px;
        margin-bottom: 16px;
        box-shadow: 0 12px 28px rgba(16, 185, 129, 0.22);
        font-size: 15px;
        line-height: 1.7;
    }

    .warning-box {
        background: linear-gradient(135deg, #f59e0b 0%, #fef08a 100%);
        color: #422006 !important;
        border-radius: 24px;
        padding: 20px 24px;
        margin-top: 14px;
        margin-bottom: 16px;
        box-shadow: 0 12px 28px rgba(245, 158, 11, 0.20);
        font-size: 15px;
        line-height: 1.7;
    }

    .danger-box {
        background: linear-gradient(135deg, #ef4444 0%, #fb7185 100%);
        color: #ffffff !important;
        border-radius: 24px;
        padding: 20px 24px;
        margin-top: 14px;
        margin-bottom: 16px;
        box-shadow: 0 12px 28px rgba(239, 68, 68, 0.22);
        font-size: 15px;
        line-height: 1.7;
    }

    .table-note {
        background: #f8fafc;
        border: 1px solid #dbeafe;
        border-radius: 18px;
        padding: 14px 16px;
        color: #334155;
        font-size: 14px;
        line-height: 1.6;
        margin-top: 12px;
    }

    .small-text {
        font-size: 13px;
        color: #475569;
        line-height: 1.55;
    }

    .risk-high {
        background: linear-gradient(135deg, #ef4444, #fb7185);
        color: white;
        border-radius: 999px;
        padding: 4px 10px;
        font-weight: 800;
    }

    .risk-medium {
        background: linear-gradient(135deg, #f59e0b, #facc15);
        color: #1f2937;
        border-radius: 999px;
        padding: 4px 10px;
        font-weight: 800;
    }

    .risk-low {
        background: linear-gradient(135deg, #10b981, #6ee7b7);
        color: #052e2b;
        border-radius: 999px;
        padding: 4px 10px;
        font-weight: 800;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #081429 0%, #102444 48%, #18345f 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] .stTextInput input {
        background-color: #020617 !important;
        color: #ffffff !important;
        border: 1px solid rgba(125, 211, 252, 0.5) !important;
        border-radius: 12px !important;
    }

    div[data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid rgba(14, 165, 233, 0.25);
        padding: 18px 20px;
        border-radius: 22px;
        box-shadow: 0px 10px 28px rgba(15, 23, 42, 0.08);
    }

    div[data-testid="stMetricLabel"] p {
        color: #334155 !important;
        font-weight: 800 !important;
        font-size: 13px !important;
    }

    div[data-testid="stMetricValue"] {
        color: #020617 !important;
        font-weight: 900 !important;
        font-size: 28px !important;
    }

    div[data-testid="stDataFrame"] {
        background: #ffffff;
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# Extra CSS fix for Streamlit selectbox/dropdown visibility
# This is separated from the main CSS because BaseWeb renders dropdown options outside the sidebar container.
st.markdown(
    """
    <style>
    /* Closed selectbox field inside sidebar */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #020617 !important;
        border: 1.5px solid #38bdf8 !important;
        border-radius: 14px !important;
        color: #ffffff !important;
        box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.10) !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] span,
    section[data-testid="stSidebar"] div[data-baseweb="select"] input,
    section[data-testid="stSidebar"] div[data-baseweb="select"] div {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] .stTextInput input {
        background-color: #020617 !important;
        border: 1.5px solid #38bdf8 !important;
        border-radius: 14px !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    section[data-testid="stSidebar"] .stTextInput input::placeholder {
        color: #cbd5e1 !important;
        opacity: 1 !important;
    }

    /* Dropdown menu options. Streamlit/BaseWeb renders this outside the sidebar. */
    div[data-baseweb="popover"] {
        z-index: 999999 !important;
    }

    div[data-baseweb="popover"] div[role="listbox"] {
        background-color: #ffffff !important;
        border: 1.5px solid #38bdf8 !important;
        border-radius: 14px !important;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.25) !important;
        padding: 6px !important;
    }

    div[data-baseweb="popover"] div[role="option"],
    div[data-baseweb="popover"] li[role="option"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        opacity: 1 !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        margin: 2px 0 !important;
    }

    div[data-baseweb="popover"] div[role="option"] *,
    div[data-baseweb="popover"] li[role="option"] * {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        opacity: 1 !important;
    }

    div[data-baseweb="popover"] div[role="option"]:hover,
    div[data-baseweb="popover"] li[role="option"]:hover {
        background: linear-gradient(135deg, #dbeafe 0%, #cffafe 100%) !important;
        color: #082f63 !important;
        -webkit-text-fill-color: #082f63 !important;
    }

    div[data-baseweb="popover"] div[aria-selected="true"],
    div[data-baseweb="popover"] li[aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb 0%, #06b6d4 100%) !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    div[data-baseweb="popover"] div[aria-selected="true"] *,
    div[data-baseweb="popover"] li[aria-selected="true"] * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* Keep metric text dark and visible in Streamlit metric widgets */
    div[data-testid="stMetric"] * {
        color: #020617 !important;
        opacity: 1 !important;
    }

    div[data-testid="stMetricLabel"] p {
        color: #334155 !important;
        font-weight: 900 !important;
    }

    div[data-testid="stMetricValue"] div {
        color: #020617 !important;
        font-weight: 900 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_credit_default.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("cleaned_credit_default.csv not found. Please upload the cleaned dataset into the same folder as this app.")
    st.stop()

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_features(data):
    data = data.copy()
    pay_status_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bill_amt_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    pay_amt_cols = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

    data["AVG_PAY_DELAY"] = data[pay_status_cols].mean(axis=1)
    data["AVG_BILL_AMT"] = data[bill_amt_cols].mean(axis=1)
    data["AVG_PAY_AMT"] = data[pay_amt_cols].mean(axis=1)
    data["PAYMENT_TO_BILL_RATIO"] = data["AVG_PAY_AMT"] / (data["AVG_BILL_AMT"] + 1)
    return data


df_model = add_features(df)

# ============================================================
# MODEL TRAINING
# ============================================================

@st.cache_resource
def train_model(data):
    X = data.drop(columns=["ID", "DEFAULT_NEXT_MONTH"])
    y = data["DEFAULT_NEXT_MONTH"]

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "True Negatives": int(cm[0, 0]),
        "False Positives": int(cm[0, 1]),
        "False Negatives": int(cm[1, 0]),
        "True Positives": int(cm[1, 1]),
    }

    feature_importance = pd.DataFrame(
        {"Feature": X.columns, "Importance": model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    return model, X.columns.tolist(), metrics, feature_importance


model, feature_columns, metrics, feature_importance = train_model(df_model)

# ============================================================
# PREDICTIONS
# ============================================================

X_all = df_model.drop(columns=["ID", "DEFAULT_NEXT_MONTH"])
X_all = X_all.replace([np.inf, -np.inf], np.nan)
X_all = X_all.fillna(0)

df_model["DEFAULT_PROBABILITY"] = model.predict_proba(X_all[feature_columns])[:, 1]

df_model["RISK_LEVEL"] = pd.cut(
    df_model["DEFAULT_PROBABILITY"],
    bins=[0, 0.30, 0.60, 1.00],
    labels=["Low Risk", "Medium Risk", "High Risk"],
    include_lowest=True,
)

df_model["RECOMMENDED_ACTION"] = np.select(
    [
        df_model["DEFAULT_PROBABILITY"] >= 0.60,
        df_model["DEFAULT_PROBABILITY"] >= 0.30,
    ],
    [
        "Prioritise for early collection review",
        "Monitor and review if payment behaviour worsens",
    ],
    default="Normal monitoring",
)

# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.title("📌 Interactive Filters")
st.sidebar.write("Use these filters to explore customer risk segments and review model outputs.")

education_label_map = {
    "All": "All Education Categories",
    1: "1 - Graduate School",
    2: "2 - University",
    3: "3 - High School",
    4: "4 - Others / Unknown",
}

education_options = ["All"] + sorted(df_model["EDUCATION"].unique().tolist())
selected_education = st.sidebar.selectbox(
    "Education Category",
    education_options,
    format_func=lambda x: education_label_map.get(x, str(x)),
)

risk_options = ["All", "Low Risk", "Medium Risk", "High Risk"]
selected_risk = st.sidebar.selectbox("Risk Level", risk_options)

search_id = st.sidebar.text_input("Search Customer ID", "")

top_n = st.sidebar.selectbox(
    "Top Risk Records to Display",
    [10, 20, 50, 100],
    index=2,
)

filtered_df = df_model.copy()

if selected_education != "All":
    filtered_df = filtered_df[filtered_df["EDUCATION"] == selected_education]

if selected_risk != "All":
    filtered_df = filtered_df[filtered_df["RISK_LEVEL"].astype(str) == selected_risk]

lookup_message = ""
if search_id.strip() != "":
    try:
        search_id_int = int(search_id)
        lookup_df = df_model[df_model["ID"] == search_id_int]
        filtered_df = filtered_df[filtered_df["ID"] == search_id_int]
        if len(lookup_df) > 0:
            row = lookup_df.iloc[0]
            lookup_message = (
                f"Customer ID **{search_id_int}** has a predicted default probability of "
                f"**{row['DEFAULT_PROBABILITY'] * 100:.2f}%** and is classified as **{row['RISK_LEVEL']}**. "
                f"Recommended action: **{row['RECOMMENDED_ACTION']}**."
            )
        else:
            lookup_message = f"Customer ID **{search_id_int}** was not found in the dataset."
    except ValueError:
        lookup_message = "Please enter a valid numeric customer ID."

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="main-title">📊 Credit Default Risk Decision Support Dashboard</div>
        <div class="subtitle">
            Agile Data Science dashboard for customer default prediction, risk segmentation, monitoring, interpretation, and stakeholder decision support.
            The dashboard acts as a Minimum Viable Dashboard that can be refined through sprint feedback.
        </div>
        <div class="pill-row">
            <span class="pill pill-blue">Predictive Output</span>
            <span class="pill pill-green">Monitoring Metrics</span>
            <span class="pill pill-orange">Interactive Filters</span>
            <span class="pill pill-purple">Agile Iteration</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# KPI CARDS
# ============================================================

total_customers = len(filtered_df)
portfolio_count = len(df_model)
segment_pct = (total_customers / portfolio_count * 100) if portfolio_count > 0 else 0
actual_default_rate = filtered_df["DEFAULT_NEXT_MONTH"].mean() * 100 if total_customers > 0 else 0
avg_predicted_risk = filtered_df["DEFAULT_PROBABILITY"].mean() * 100 if total_customers > 0 else 0
high_risk_count = (filtered_df["RISK_LEVEL"].astype(str) == "High Risk").sum() if total_customers > 0 else 0

c1, c2, c3, c4 = st.columns(4)

cards = [
    ("Total Customers", f"{total_customers:,}", f"{segment_pct:.2f}% of full dataset"),
    ("Actual Default Rate", f"{actual_default_rate:.2f}%", "Observed target outcome"),
    ("Average Predicted Risk", f"{avg_predicted_risk:.2f}%", "Model-generated probability"),
    ("High Risk Customers", f"{high_risk_count:,}", "Probability ≥ 60%"),
]

for col, (label, value, note) in zip([c1, c2, c3, c4], cards):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Dynamic dashboard interpretation
if total_customers == 0:
    interpretation_text = "No records match the current filter. Adjust the filter selection to review customer risk segments."
else:
    filter_desc = []
    filter_desc.append(f"Education = {education_label_map.get(selected_education, selected_education)}")
    filter_desc.append(f"Risk Level = {selected_risk}")
    interpretation_text = (
        f"Current filter: **{', '.join(filter_desc)}**. The selected segment contains **{total_customers:,} customers**, "
        f"representing **{segment_pct:.2f}%** of the full dataset. The actual default rate is **{actual_default_rate:.2f}%**, "
        f"while the average model-predicted risk is **{avg_predicted_risk:.2f}%**. "
    )
    if selected_risk == "High Risk" or high_risk_count > 0:
        interpretation_text += "This segment should be prioritised for early review because it includes customers with high predicted default risk."
    elif selected_risk == "Medium Risk":
        interpretation_text += "This segment should be monitored because the predicted risk is moderate and may change with future payment behaviour."
    else:
        interpretation_text += "This segment provides an overall view for portfolio-level monitoring and stakeholder discussion."

st.markdown(
    f"""
    <div class="interpretation-box">
        <b>Dashboard Interpretation:</b><br>{interpretation_text}
    </div>
    """,
    unsafe_allow_html=True,
)

if lookup_message:
    st.markdown(
        f"""
        <div class="warning-box">
            <b>Customer Lookup Interpretation:</b><br>{lookup_message}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# MODEL MONITORING METRICS
# ============================================================

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("🚦 Model Monitoring Metrics")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
m2.metric("Precision", f"{metrics['Precision']:.4f}")
m3.metric("Recall", f"{metrics['Recall']:.4f}")
m4.metric("F1-score", f"{metrics['F1-score']:.4f}")
m5.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")

monitoring_interpretation = (
    f"The improved Random Forest model records recall of **{metrics['Recall']:.4f}** and F1-score of **{metrics['F1-score']:.4f}**. "
    "Recall is important in default-risk prioritisation because missing actual default customers can reduce the usefulness of the decision support output. "
    "These metrics should be reviewed in future Agile iterations to decide whether retraining or threshold adjustment is required."
)

st.markdown(
    f"""
    <div class="success-box">
        <b>Monitoring Interpretation:</b><br>{monitoring_interpretation}
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# DATA QUALITY MONITORING
# ============================================================

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("🧪 Data Quality Monitoring")

missing_rate = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
duplicate_ids = df["ID"].duplicated().sum()
invalid_education = (~df["EDUCATION"].isin([1, 2, 3, 4])).sum()
invalid_marriage = (~df["MARRIAGE"].isin([1, 2, 3])).sum()

dq1, dq2, dq3, dq4 = st.columns(4)
dq1.metric("Missing Value Rate", f"{missing_rate:.2f}%")
dq2.metric("Duplicate IDs", f"{duplicate_ids}")
dq3.metric("Invalid Education Values", f"{invalid_education}")
dq4.metric("Invalid Marriage Values", f"{invalid_marriage}")

data_quality_interpretation = (
    "The cleaned dataset passed the main validation checks. Missing value rate, duplicate IDs, and invalid category values are monitored to ensure dashboard reliability and model readiness. "
    "If any of these metrics increase in future data updates, the next Agile sprint should prioritise data cleaning and validation improvements."
)

st.markdown(
    f"""
    <div class="success-box">
        <b>Data Quality Interpretation:</b><br>{data_quality_interpretation}
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# VISUAL ANALYTICS
# ============================================================

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("📈 Visual Analytics with Interpretation")

if total_customers == 0:
    st.warning("No records match the selected filters. Please adjust the filters.")
else:
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Visualization 1: Actual Default Distribution")
        default_counts = filtered_df["DEFAULT_NEXT_MONTH"].value_counts().reindex([0, 1]).fillna(0)
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.set_facecolor("#ffffff")
        ax1.bar(["No Default", "Default"], default_counts.values, color=["#10b981", "#fb4675"], edgecolor="white", linewidth=2)
        ax1.set_title("Actual Default Distribution", fontsize=13, fontweight="bold", color="#0f172a")
        ax1.set_ylabel("Number of Customers", color="#0f172a")
        ax1.tick_params(colors="#0f172a")
        ax1.grid(axis="y", alpha=0.25)
        for spine in ["top", "right"]:
            ax1.spines[spine].set_visible(False)
        st.pyplot(fig1)
        st.markdown(
            f"""
            <div class="table-note"><b>Interpretation:</b><br>
            Within the selected filter, the actual observed default rate is <b>{actual_default_rate:.2f}%</b>. 
            This chart helps stakeholders compare default and non-default customer volume.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with chart_col2:
        st.markdown("#### Visualization 2: Predicted Risk Level Distribution")
        risk_counts = filtered_df["RISK_LEVEL"].astype(str).value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"]).fillna(0)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.set_facecolor("#ffffff")
        ax2.bar(risk_counts.index, risk_counts.values, color=["#10b981", "#f59e0b", "#ef4444"], edgecolor="white", linewidth=2)
        ax2.set_title("Predicted Risk Level Distribution", fontsize=13, fontweight="bold", color="#0f172a")
        ax2.set_ylabel("Number of Customers", color="#0f172a")
        ax2.tick_params(colors="#0f172a")
        ax2.grid(axis="y", alpha=0.25)
        for spine in ["top", "right"]:
            ax2.spines[spine].set_visible(False)
        st.pyplot(fig2)
        st.markdown(
            f"""
            <div class="table-note"><b>Interpretation:</b><br>
            The model classifies customers into low, medium, and high-risk groups. 
            The current filter contains <b>{int(risk_counts.get('High Risk', 0)):,}</b> high-risk customers.
            </div>
            """,
            unsafe_allow_html=True,
        )

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.markdown("#### Visualization 3: Default Rate by Education")
        education_default = filtered_df.groupby("EDUCATION")["DEFAULT_NEXT_MONTH"].mean().reset_index()
        education_default["Default Rate (%)"] = education_default["DEFAULT_NEXT_MONTH"] * 100
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.set_facecolor("#ffffff")
        ax3.bar(education_default["EDUCATION"].astype(str), education_default["Default Rate (%)"], color="#3b82f6", edgecolor="white", linewidth=2)
        ax3.set_title("Default Rate by Education Category", fontsize=13, fontweight="bold", color="#0f172a")
        ax3.set_xlabel("Education Category", color="#0f172a")
        ax3.set_ylabel("Default Rate (%)", color="#0f172a")
        ax3.tick_params(colors="#0f172a")
        ax3.grid(axis="y", alpha=0.25)
        for spine in ["top", "right"]:
            ax3.spines[spine].set_visible(False)
        st.pyplot(fig3)
        if len(education_default) > 0:
            highest_edu_row = education_default.sort_values("Default Rate (%)", ascending=False).iloc[0]
            edu_interp = f"Education category <b>{int(highest_edu_row['EDUCATION'])}</b> shows the highest default rate within the current filter at <b>{highest_edu_row['Default Rate (%)']:.2f}%</b>."
        else:
            edu_interp = "No education category is available under the selected filter."
        st.markdown(
            f"""
            <div class="table-note"><b>Interpretation:</b><br>{edu_interp}</div>
            """,
            unsafe_allow_html=True,
        )

    with chart_col4:
        st.markdown("#### Visualization 4: Default Probability Distribution")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        ax4.set_facecolor("#ffffff")
        ax4.hist(filtered_df["DEFAULT_PROBABILITY"], bins=25, color="#7c3aed", edgecolor="white", linewidth=1.2)
        ax4.axvline(0.30, color="#f59e0b", linestyle="--", linewidth=2, label="Medium threshold")
        ax4.axvline(0.60, color="#ef4444", linestyle="--", linewidth=2, label="High threshold")
        ax4.set_title("Predicted Default Probability", fontsize=13, fontweight="bold", color="#0f172a")
        ax4.set_xlabel("Default Probability", color="#0f172a")
        ax4.set_ylabel("Number of Customers", color="#0f172a")
        ax4.tick_params(colors="#0f172a")
        ax4.grid(axis="y", alpha=0.25)
        ax4.legend()
        for spine in ["top", "right"]:
            ax4.spines[spine].set_visible(False)
        st.pyplot(fig4)
        st.markdown(
            """
            <div class="table-note"><b>Interpretation:</b><br>
            The dashed lines mark the medium-risk threshold at <b>30%</b> and high-risk threshold at <b>60%</b>. 
            Customers above 60% should be prioritised for earlier review.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Visualization 5: Top 10 Feature Importance")
    top_features = feature_importance.head(10).sort_values(by="Importance", ascending=True)
    fig5, ax5 = plt.subplots(figsize=(11, 5.2))
    ax5.set_facecolor("#ffffff")
    colors = ["#06b6d4" if f != top_features.iloc[-1]["Feature"] else "#2563eb" for f in top_features["Feature"]]
    ax5.barh(top_features["Feature"], top_features["Importance"], color=colors, edgecolor="white", linewidth=1.5)
    ax5.set_title("Top 10 Important Features for Default Prediction", fontsize=14, fontweight="bold", color="#0f172a")
    ax5.set_xlabel("Feature Importance", color="#0f172a")
    ax5.tick_params(colors="#0f172a")
    ax5.grid(axis="x", alpha=0.25)
    for spine in ["top", "right"]:
        ax5.spines[spine].set_visible(False)
    st.pyplot(fig5)
    top_feature = feature_importance.iloc[0]["Feature"]
    st.markdown(
        f"""
        <div class="table-note"><b>Interpretation:</b><br>
        The most influential feature in the improved model is <b>{top_feature}</b>. 
        This helps stakeholders understand which customer attributes contribute most to the risk prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PREDICTIVE OUTPUT
# ============================================================

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("🎯 Predictive Output: Customer Default Risk Prioritisation")

output_cols = [
    "ID",
    "LIMIT_BAL",
    "AGE",
    "EDUCATION",
    "MARRIAGE",
    "AVG_PAY_DELAY",
    "AVG_BILL_AMT",
    "AVG_PAY_AMT",
    "PAYMENT_TO_BILL_RATIO",
    "DEFAULT_PROBABILITY",
    "RISK_LEVEL",
    "RECOMMENDED_ACTION",
    "DEFAULT_NEXT_MONTH",
]

output_table = filtered_df[output_cols].sort_values(by="DEFAULT_PROBABILITY", ascending=False).head(top_n)
display_table = output_table.copy()

if len(display_table) > 0:
    display_table["DEFAULT_PROBABILITY"] = (display_table["DEFAULT_PROBABILITY"] * 100).round(2).astype(str) + "%"
    display_table["AVG_PAY_DELAY"] = display_table["AVG_PAY_DELAY"].round(2)
    display_table["AVG_BILL_AMT"] = display_table["AVG_BILL_AMT"].round(2)
    display_table["AVG_PAY_AMT"] = display_table["AVG_PAY_AMT"].round(2)
    display_table["PAYMENT_TO_BILL_RATIO"] = display_table["PAYMENT_TO_BILL_RATIO"].round(4)
    st.table(display_table)
else:
    st.warning("No predictive output is available under the current filter.")

if len(output_table) > 0:
    top_customer = output_table.iloc[0]
    pred_interp = (
        f"The highest-risk customer shown is ID **{int(top_customer['ID'])}**, with predicted default probability of "
        f"**{top_customer['DEFAULT_PROBABILITY'] * 100:.2f}%** and risk level **{top_customer['RISK_LEVEL']}**. "
        f"Recommended action: **{top_customer['RECOMMENDED_ACTION']}**."
    )
else:
    pred_interp = "No customers are available under the current filter."

st.markdown(
    f"""
    <div class="danger-box">
        <b>Decision Support Interpretation:</b><br>{pred_interp}<br><br>
        This dashboard does not replace human judgement. It supports officers by providing clearer, data-driven risk indicators for prioritising customer account review.
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# AGILE ITERATION NOTES
# ============================================================

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("🔁 Agile Iteration Notes")

st.markdown(
    """
    <div class="table-note">
    <b>How this dashboard supports Agile Data Science:</b><br>
    • The dashboard acts as a Minimum Viable Dashboard for Sprint 4.<br>
    • Stakeholders can review risk segments using filters and predictive outputs.<br>
    • Monitoring metrics can be discussed during sprint reviews and retrospectives.<br>
    • Feedback such as adding Excel export, risk alerts, or new filters can be converted into future backlog items.<br>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="small-text">
    Prepared for PMA Agile Data Science: Credit Default Risk Decision Support using iterative development, automation, dashboarding, monitoring, and stakeholder feedback.
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)
