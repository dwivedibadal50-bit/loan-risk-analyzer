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
# HEADER
# =========================================

st.title("🏦 AI Loan Risk Prediction System")
st.markdown("### Smart Banking Risk Analytics Dashboard")

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

# =========================================
# PREDICT BUTTON
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

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )