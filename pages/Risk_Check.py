# =========================================
# FILE: pages/Risk_Check.py
# =========================================

import streamlit as st
import pandas as pd
import joblib
import os

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Loan Risk Checker",
    page_icon="🏦",
    layout="wide"
)

# =========================================
# PROFESSIONAL BLUE THEME
# =========================================

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #dceeff !important;
}

.stApp {
    background: linear-gradient(
        180deg,
        #dceeff 0%,
        #cfe7ff 40%,
        #b9dbff 100%
    );
}

.main .block-container {

    background: rgba(255,255,255,0.20);

    padding: 2rem;

    border-radius: 28px;

    border: 2px solid #90c2ff;

    box-shadow:
        0 8px 30px rgba(0,0,0,0.08);

    margin-top: 10px;
}

h1 {
    color: #08306b !important;
    font-weight: 900 !important;
    text-align: center;
}

h2, h3, h4 {
    color: #0b5394 !important;
    font-weight: 800 !important;
}

section[data-testid="stSidebar"] {

    background: linear-gradient(
        180deg,
        #dbeafe,
        #93c5fd
    );

    border-right: 3px solid #60a5fa;
}

section[data-testid="stSidebar"] * {
    color: #08306b !important;
    font-weight: 700;
}

.stButton>button {

    background: linear-gradient(
        135deg,
        #1976d2,
        #64b5f6
    );

    color: white;

    border-radius: 16px;

    border: none;

    padding: 14px 28px;

    font-weight: bold;

    font-size: 17px;

    box-shadow:
        0 5px 15px rgba(0,0,0,0.15);

    transition: 0.3s;
}

.stButton>button:hover {

    transform: translateY(-3px);

    box-shadow:
        0 10px 20px rgba(0,0,0,0.18);
}

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
}

.stNumberInput {

    background: white;

    border-radius: 16px;

    padding: 8px;

    border: 2px solid #93c5fd;
}

.streamlit-expanderHeader {

    background: #dbeafe !important;

    border-radius: 12px;

    border: 1px solid #93c5fd;

    font-weight: 700;
}

.stSuccess {
    border-radius: 16px;
}

.stWarning {
    border-radius: 16px;
}

.stError {
    border-radius: 16px;
}

.stInfo {
    border-radius: 16px;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# HEADER
# =========================================

st.markdown("""
<h1>🏦 AI Loan Risk Prediction System</h1>

<p style='
text-align:center;
font-size:20px;
font-weight:600;
color:#334155;
'>
Smart Banking Risk Analytics Dashboard
</p>

<hr>
""", unsafe_allow_html=True)

# =========================================
# ROOT DIRECTORY
# =========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model.joblib"
)

# =========================================
# LOAD MODEL
# =========================================

try:

    if not os.path.exists(MODEL_PATH):

        st.error(
            "❌ model.joblib not found.\n\n"
            "Run train_model.py locally first."
        )

        st.stop()

    pipeline = joblib.load(MODEL_PATH)

    model = pipeline["model"]

    scaler = pipeline["scaler"]

    imputer = pipeline["imputer"]

    selected_columns = pipeline["selected_columns"]

    target_encoder = pipeline["target_encoder"]

    st.success("✅ Model Loaded Successfully")

except Exception as e:

    st.error(f"❌ Error loading model: {e}")

    st.stop()

# =========================================
# KPI SECTION
# =========================================

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Model Accuracy", "67%")

with k2:
    st.metric("Precision", "72%")

with k3:
    st.metric("Recall", "67%")

with k4:
    st.metric("F1 Score", "69%")

st.markdown("---")

# =========================================
# INPUT SECTION
# =========================================

st.subheader("📌 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    AGE = st.number_input(
        "Age",
        18,
        100,
        30
    )

    NETMONTHLYINCOME = st.number_input(
        "Monthly Income",
        1000,
        1000000,
        40000
    )

    Time_With_Curr_Empr = st.number_input(
        "Months With Current Employer",
        0,
        500,
        24
    )

with col2:

    Total_TL = st.number_input(
        "Total Trade Lines",
        0,
        100,
        5
    )

    Tot_Active_TL = st.number_input(
        "Active Trade Lines",
        0,
        100,
        3
    )

    tot_enq = st.number_input(
        "Total Credit Enquiries",
        0,
        100,
        2
    )

with col3:

    Tot_Missed_Pmnt = st.number_input(
        "Missed Payments",
        0,
        100,
        0
    )

    num_times_delinquent = st.number_input(
        "Number of Delinquencies",
        0,
        100,
        0
    )

    num_times_30p_dpd = st.number_input(
        "30+ DPD Count",
        0,
        100,
        0
    )

with st.expander("⚙ Additional Optional Information"):

    num_times_60p_dpd = st.number_input(
        "60+ DPD Count",
        0,
        100,
        0
    )

st.markdown("<br>", unsafe_allow_html=True)

# =========================================
# BUTTON
# =========================================

predict_btn = st.button(
    "🔍 Analyze Customer Risk",
    use_container_width=True
)

# =========================================
# PREDICTION
# =========================================

if predict_btn:

    input_df = pd.DataFrame(
        columns=selected_columns
    )

    input_df.loc[0] = 0

    values = {

        "AGE": AGE,

        "NETMONTHLYINCOME": NETMONTHLYINCOME,

        "Total_TL": Total_TL,

        "Tot_Active_TL": Tot_Active_TL,

        "Time_With_Curr_Empr": Time_With_Curr_Empr,

        "Tot_Missed_Pmnt": Tot_Missed_Pmnt,

        "num_times_delinquent": num_times_delinquent,

        "tot_enq": tot_enq,

        "num_times_30p_dpd": num_times_30p_dpd,

        "num_times_60p_dpd": num_times_60p_dpd,

        "payment_score": (
            Tot_Missed_Pmnt +
            num_times_30p_dpd +
            num_times_60p_dpd
        ),

        "delinq_score": (
            num_times_delinquent
        ),

        "enquiry_score": (
            tot_enq
        )
    }

    for col, val in values.items():

        if col in input_df.columns:

            input_df[col] = val

    input_df = pd.DataFrame(
        imputer.transform(input_df),
        columns=input_df.columns
    )

    input_scaled = scaler.transform(
        input_df
    )

    prediction_encoded = model.predict(
        input_scaled
    )[0]

    prediction = target_encoder.inverse_transform(
        [prediction_encoded]
    )[0]

    probabilities = model.predict_proba(
        input_scaled
    )[0]

    confidence = max(probabilities) * 100

    # =========================================
    # RISK SCORE
    # =========================================

    risk_score = 100

    risk_score -= Tot_Missed_Pmnt * 8
    risk_score -= num_times_delinquent * 10
    risk_score -= num_times_30p_dpd * 6
    risk_score -= num_times_60p_dpd * 10
    risk_score -= tot_enq * 2

    risk_score = max(0, min(100, risk_score))

    # =========================================
    # RESULTS
    # =========================================

    st.markdown("---")

    st.subheader("📊 Risk Analysis Result")

    if prediction == "P1":

        st.success(
            "✅ P1 - Low Risk Customer"
        )

    elif prediction == "P2":

        st.info(
            "🟢 P2 - Moderate Risk Customer"
        )

    elif prediction == "P3":

        st.warning(
            "🟠 P3 - High Risk Customer"
        )

    else:

        st.error(
            "🔴 P4 - Very High Risk Customer"
        )

    m1, m2 = st.columns(2)

    with m1:

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

    with m2:

        st.metric(
            "Risk Score",
            f"{risk_score}/100"
        )

    # =========================================
    # AI INSIGHTS
    # =========================================

    st.subheader("🧠 AI Insights")

    insights = []

    if Tot_Missed_Pmnt > 0:
        insights.append(
            "❌ Customer has missed payment history"
        )

    if num_times_delinquent > 0:
        insights.append(
            "⚠ Delinquency behavior detected"
        )

    if tot_enq > 5:
        insights.append(
            "⚠ High number of recent enquiries"
        )

    if NETMONTHLYINCOME > 80000:
        insights.append(
            "✅ Strong income profile"
        )

    if Tot_Active_TL >= 5:
        insights.append(
            "✅ Healthy trade line activity"
        )

    if len(insights) == 0:
        insights.append(
            "✅ Financial profile looks stable"
        )

    for item in insights:

        st.info(item)

    # =========================================
    # BANK RECOMMENDATION
    # =========================================

    st.markdown("---")

    st.subheader("🏦 Bank Recommendation")

    if prediction == "P1":

        st.success(
            "Loan can be approved with low risk exposure."
        )

    elif prediction == "P2":

        st.info(
            "Loan can be approved with moderate monitoring."
        )

    elif prediction == "P3":

        st.warning(
            "Additional verification recommended before approval."
        )

    else:

        st.error(
            "Loan approval is highly risky."
        )

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.markdown("""
<div style='text-align:center;
padding:15px;
font-size:16px;
font-weight:600;
color:#08306b;'>

🏦 AI Loan Risk Analyzer | Banking Analytics Project 2026

</div>
""", unsafe_allow_html=True)
