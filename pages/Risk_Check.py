# =========================================
# PAGE: Risk_Check.py
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
# ADVANCED FULL BLUE THEME
# =========================================

st.markdown(
    """
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
        color: #0f172a;
    }

    .main .block-container {
        background: rgba(255,255,255,0.18);
        padding: 2rem;
        border-radius: 28px;
        border: 2px solid #90c2ff;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        margin-top: 12px;
    }

    h1 {
        color: #08306b !important;
        font-weight: 900 !important;
        text-align: center;
        letter-spacing: 1px;
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
        font-weight: 600;
    }

    .stNumberInput {
        background: white;
        border-radius: 16px;
        padding: 8px;
        border: 2px solid #93c5fd;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }

    label {
        color: #0b3d91 !important;
        font-weight: 700 !important;
    }

    .streamlit-expanderHeader {
        background: #dbeafe !important;
        border-radius: 12px;
        border: 1px solid #93c5fd;
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
        transition: 0.3s;
        box-shadow: 0 5px 15px rgba(0,0,0,0.15);
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.18);
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
        box-shadow: 0 6px 15px rgba(0,0,0,0.10);
        transition: all 0.3s ease-in-out;
    }

    [data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border: 2px solid #1976d2;
        box-shadow: 0 12px 22px rgba(0,0,0,0.16);
    }

    [data-testid="metric-container"] label {
        color: #0b3d91 !important;
        font-size: 16px !important;
        font-weight: 800 !important;
    }

    [data-testid="metric-container"] div {
        color: #002b5b !important;
        font-size: 30px !important;
        font-weight: 900 !important;
    }

    .stAlert {
        border-radius: 18px !important;
        border: 2px solid #90caf9 !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    }

    .stSuccess {
        background: #d4edda !important;
    }

    .stInfo {
        background: #dbeafe !important;
    }

    .stWarning {
        background: #fff3cd !important;
    }

    .stError {
        background: #ffe5e5 !important;
    }

    hr {
        border: 1px solid #64b5f6;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================
# HEADER
# =========================================

st.markdown(
    """
    <h1>🏦 AI Loan Risk Prediction System</h1>

    <p style='text-align:center;
              color:#334155;
              font-size:18px;
              font-weight:600;'>

    Smart Banking Risk Analytics Dashboard

    </p>

    <hr>
    """,
    unsafe_allow_html=True
)

# =========================================
# MODEL PATH
# =========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model.pkl"
)

# =========================================
# LOAD / TRAIN MODEL
# =========================================

try:

    if not os.path.exists(MODEL_PATH):

        st.warning("⚠ Model not found. Training model...")

        import train_model

        st.success("✅ model.pkl created successfully")

    pipeline = joblib.load(MODEL_PATH)

    model = pipeline["model"]
    scaler = pipeline["scaler"]
    imputer = pipeline["imputer"]
    selected_columns = pipeline["selected_columns"]
    target_encoder = pipeline["target_encoder"]

except Exception as e:

    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# =========================================
# INPUT SECTION
# =========================================

st.subheader("📌 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    AGE = st.number_input("Age", 18, 100, 30)

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

predict_btn = st.button(
    "🔍 Analyze Customer Risk",
    use_container_width=True
)
