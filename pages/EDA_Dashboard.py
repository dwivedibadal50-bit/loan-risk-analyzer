# =========================================
# EDA DASHBOARD
# File: pages/EDA_Dashboard.py
# =========================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="EDA Dashboard",
    layout="wide"
)

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("loan_data.csv")

# =====================================
# ADVANCED FULL BLUE UI THEME
# =====================================

st.markdown("""
<style>

/* ENTIRE PAGE BACKGROUND */
html, body, [class*="css"] {
    background-color: #dceeff !important;
}

/* MAIN APP */
.stApp {
    background: linear-gradient(
        180deg,
        #dceeff 0%,
        #cfe7ff 40%,
        #b9dbff 100%
    );
}

/* MAIN CONTENT AREA */
.main .block-container {

    background: rgba(255,255,255,0.15);

    padding: 2rem;

    border-radius: 25px;

    border: 2px solid #90c2ff;

    box-shadow:
        0 8px 30px rgba(0,0,0,0.08);

    margin-top: 15px;
}

/* TITLE */
h1 {
    color: #08306b !important;
    font-weight: 900 !important;
    text-align: center;
}

/* SUBHEADINGS */
h2, h3 {
    color: #0b5394 !important;
    font-weight: 800 !important;
}

/* KPI BOXES */
[data-testid="metric-container"] {

    background: linear-gradient(
        135deg,
        #ffffff,
        #d6ebff
    );

    border: 2px solid #5aa9ff;

    padding: 22px;

    border-radius: 20px;

    box-shadow:
        0 6px 15px rgba(0,0,0,0.10);

    transition: all 0.3s ease-in-out;
}

/* KPI HOVER */
[data-testid="metric-container"]:hover {

    transform: translateY(-6px);

    border: 2px solid #1976d2;

    box-shadow:
        0 12px 22px rgba(0,0,0,0.18);
}

/* KPI LABEL */
[data-testid="metric-container"] label {

    color: #0b3d91 !important;

    font-size: 16px !important;

    font-weight: 800 !important;
}

/* KPI VALUE */
[data-testid="metric-container"] div {

    color: #002b5b !important;

    font-size: 30px !important;

    font-weight: 900 !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {

    background: linear-gradient(
        180deg,
        #dbeafe,
        #93c5fd
    );

    border-right: 3px solid #60a5fa;
}

/* SIDEBAR TEXT */
section[data-testid="stSidebar"] * {
    color: #08306b !important;
    font-weight: 600;
}

/* DATAFRAME */
.stDataFrame {

    background: white;

    border-radius: 16px;

    border: 2px solid #90caf9;

    padding: 10px;
}

/* CHART AREA */
.element-container:has(canvas) {

    background: white;

    padding: 20px;

    border-radius: 20px;

    border: 2px solid #90caf9;

    margin-bottom: 20px;
}

/* ALERT BOXES */
.stAlert {

    border-radius: 16px !important;
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {
    background: #64b5f6;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# TITLE
# =====================================

st.title("📊 AI Loan Risk EDA Dashboard")

st.markdown("""
### Banking Risk Analytics & Customer Intelligence
Analyze loan approval trends, customer risk, delinquency behaviour, and financial insights.
""")

# =====================================
# KPI SECTION
# =====================================

st.subheader("📌 Portfolio KPIs")

total_customers = df.shape[0]

high_risk = (
    (df["Approved_Flag"] == "P3") |
    (df["Approved_Flag"] == "P4")
).sum()

low_risk = (
    df["Approved_Flag"] == "P1"
).sum()

moderate_risk = (
    df["Approved_Flag"] == "P2"
).sum()

avg_income = int(
    df["NETMONTHLYINCOME"].mean()
)

risk_rate = round(
    (high_risk / total_customers) * 100,
    2
)

if "Credit_Score" in df.columns:
    avg_credit = int(df["Credit_Score"].mean())
else:
    avg_credit = 0

# =====================================
# KPI ROW 1
# =====================================

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("👥 Total Customers", f"{total_customers:,}")

with k2:
    st.metric("🔴 High Risk", f"{high_risk:,}")

with k3:
    st.metric("🟢 Low Risk", f"{low_risk:,}")

with k4:
    st.metric("⚠ Risk Rate", f"{risk_rate}%")

# =====================================
# KPI ROW 2
# =====================================

k5, k6, k7, k8 = st.columns(4)

with k5:
    st.metric("💰 Avg Income", f"₹ {avg_income:,}")

with k6:
    st.metric("📈 Avg Credit Score", avg_credit)

with k7:
    st.metric(
        "❌ Avg Delinquency",
        round(df["num_times_delinquent"].mean(), 2)
    )

with k8:
    st.metric(
        "📋 Avg Enquiries",
        round(df["tot_enq"].mean(), 2)
    )

# =====================================
# DATA PREVIEW
# =====================================

st.subheader("📄 Dataset Preview")

st.dataframe(df.head())

# =====================================
# TARGET DISTRIBUTION
# =====================================

st.subheader("📊 Loan Approval Categories")

target_counts = df["Approved_Flag"].value_counts()

fig1, ax1 = plt.subplots(figsize=(8,5))

ax1.bar(
    target_counts.index,
    target_counts.values,
    color="#1976d2"
)

ax1.set_title("Approved Flag Distribution")

st.pyplot(fig1)

# =====================================
# INCOME DISTRIBUTION
# =====================================

st.subheader("💰 Income Distribution")

fig2, ax2 = plt.subplots(figsize=(8,5))

ax2.hist(
    df["NETMONTHLYINCOME"],
    bins=30,
    color="#42a5f5",
    edgecolor="black"
)

ax2.set_title("Monthly Income Distribution")

st.pyplot(fig2)

# =====================================
# DELINQUENCY
# =====================================

st.subheader("❌ Delinquency Analysis")

fig3, ax3 = plt.subplots(figsize=(8,5))

ax3.hist(
    df["num_times_delinquent"],
    bins=20,
    color="#1565c0",
    edgecolor="black"
)

ax3.set_title("Delinquency Count")

st.pyplot(fig3)

# =====================================
# ENQUIRIES
# =====================================

st.subheader("🔍 Credit Enquiry Analysis")

fig4, ax4 = plt.subplots(figsize=(8,5))

ax4.hist(
    df["tot_enq"],
    bins=20,
    color="#0d47a1",
    edgecolor="black"
)

ax4.set_title("Credit Enquiries")

st.pyplot(fig4)

# =====================================
# MISSING VALUES
# =====================================

st.subheader("🧹 Missing Values")

missing = df.isnull().sum()

missing = missing[missing > 0]

st.dataframe(
    missing.sort_values(ascending=False)
)

# =====================================
# BUSINESS INSIGHTS
# =====================================

st.subheader("🧠 AI Risk Insights")

insights = []

if risk_rate > 20:
    insights.append(
        "⚠ Large number of risky customers detected."
    )

if avg_credit < 700 and avg_credit != 0:
    insights.append(
        "📉 Average credit score is below optimal level."
    )

if df["num_times_delinquent"].mean() > 1:
    insights.append(
        "❌ Delinquency trend increasing financial risk."
    )

if df["tot_enq"].mean() > 3:
    insights.append(
        "🔍 High enquiry behaviour observed."
    )

if moderate_risk > low_risk:
    insights.append(
        "⚠ Moderate-risk customers dominate portfolio."
    )

if len(insights) == 0:
    insights.append(
        "✅ Portfolio appears financially stable."
    )

for i in insights:
    st.info(i)
