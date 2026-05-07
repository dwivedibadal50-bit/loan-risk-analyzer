import streamlit as st

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="AI Loan Risk Analyzer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# ADVANCED FULL BLUE THEME
# =========================================

st.markdown("""
<style>

/* FULL PAGE */
html, body, [class*="css"] {

    background-color: #dceeff !important;
}

/* APP */
.stApp {

    background: linear-gradient(
        180deg,
        #dceeff 0%,
        #cfe7ff 40%,
        #b9dbff 100%
    );

    color: #0f172a;
}

/* MAIN CONTAINER */
.main .block-container {

    background: rgba(255,255,255,0.18);

    padding: 2rem;

    border-radius: 28px;

    border: 2px solid #90c2ff;

    box-shadow:
        0 8px 30px rgba(0,0,0,0.08);

    margin-top: 12px;
}

/* TITLE */
.main-title {

    text-align: center;

    color: #08306b;

    font-size: 50px;

    font-weight: 900;

    letter-spacing: 1px;
}

/* SUBTITLE */
.sub-title {

    text-align: center;

    color: #334155;

    font-size: 20px;

    font-weight: 600;

    margin-bottom: 30px;
}

/* HEADINGS */
h1, h2, h3, h4 {

    color: #0b5394 !important;

    font-weight: 800 !important;
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

section[data-testid="stSidebar"] * {

    color: #08306b !important;

    font-weight: 600;
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

    transform: translateY(-5px);

    border: 2px solid #1976d2;

    box-shadow:
        0 12px 22px rgba(0,0,0,0.16);
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

/* MAIN CARDS */
.card {

    background: linear-gradient(
        135deg,
        #ffffff,
        #eaf4ff
    );

    padding: 25px;

    border-radius: 22px;

    border: 2px solid #93c5fd;

    box-shadow:
        0 6px 18px rgba(0,0,0,0.08);

    margin-bottom: 15px;

    transition: 0.3s;
}

/* CARD HOVER */
.card:hover {

    transform: translateY(-4px);

    box-shadow:
        0 12px 24px rgba(0,0,0,0.12);
}

/* PERFORMANCE CARDS */
.metric-card {

    background: linear-gradient(
        135deg,
        #ffffff,
        #dbeafe
    );

    padding: 20px;

    border-radius: 20px;

    border: 2px solid #93c5fd;

    text-align: center;

    box-shadow:
        0 6px 18px rgba(0,0,0,0.08);

    transition: 0.3s;
}

.metric-card:hover {

    transform: translateY(-4px);

    box-shadow:
        0 12px 22px rgba(0,0,0,0.14);
}

/* ALERTS */
.stSuccess {

    background: #d4edda !important;

    border-radius: 18px !important;

    border: 2px solid #90caf9 !important;
}

.stInfo {

    background: #dbeafe !important;

    border-radius: 18px !important;
}

.stWarning {

    background: #fff3cd !important;

    border-radius: 18px !important;
}

.stError {

    background: #ffe5e5 !important;

    border-radius: 18px !important;
}

/* BUTTONS */
.stButton > button {

    width: 100%;

    background: linear-gradient(
        135deg,
        #1976d2,
        #42a5f5
    );

    color: white;

    border-radius: 14px;

    border: none;

    padding: 12px;

    font-size: 16px;

    font-weight: 700;

    transition: 0.3s;
}

.stButton > button:hover {

    transform: scale(1.02);

    background: linear-gradient(
        135deg,
        #1565c0,
        #1e88e5
    );
}

/* HORIZONTAL LINE */
hr {

    border: 1px solid #64b5f6;
}

/* REMOVE STREAMLIT BRANDING */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
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

# =========================================
# HEADER
# =========================================

st.markdown("""
<div class='main-title'>
🏦 AI Loan Risk Analyzer
</div>

<div class='sub-title'>
Smart Banking Risk Analytics & Loan Approval Prediction System
</div>
""", unsafe_allow_html=True)

st.success("✅ Application Running Successfully")

# =========================================
# MODEL PERFORMANCE
# =========================================

st.subheader("📊 Model Performance KPIs")

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Accuracy", "67%")

with k2:
    st.metric("Precision", "72%")

with k3:
    st.metric("Recall", "67%")

with k4:
    st.metric("F1 Score", "69%")

# =========================================
# CLASSIFICATION REPORT
# =========================================

st.markdown("---")

st.subheader("🤖 Risk Category Performance")

c1, c2, c3, c4 = st.columns(4)

cards = [
    ("✅ P1", "56%", "89%", "68%"),
    ("🟢 P2", "86%", "70%", "77%"),
    ("🟠 P3", "34%", "44%", "38%"),
    ("🔴 P4", "60%", "65%", "62%")
]

cols = [c1, c2, c3, c4]

for col, card in zip(cols, cards):

    with col:

        st.markdown(f"""
        <div class='metric-card'>

        <h3>{card[0]}</h3>

        Precision: {card[1]} <br>
        Recall: {card[2]} <br>
        F1 Score: {card[3]}

        </div>
        """, unsafe_allow_html=True)

# =========================================
# DATASET INFO
# =========================================

st.markdown("---")

st.subheader("📁 Dataset Information")

d1, d2, d3, d4 = st.columns(4)

with d1:
    st.metric("Customer Records", "51,336")

with d2:
    st.metric("Total Features", "87")

with d3:
    st.metric("Features Used", "9")

with d4:
    st.metric("Target Classes", "4")

# =========================================
# PROJECT OVERVIEW
# =========================================

st.markdown("---")

st.subheader("📌 Project Overview")

st.markdown("""
<div class='card'>

This AI-powered banking analytics system helps identify 
high-risk loan applicants before loan approval.

<br><br>

The model analyzes:

<br><br>

• Delinquency history <br>
• Missed payments <br>
• Credit enquiries <br>
• Trade line activity <br>
• Income stability <br>
• Employment duration <br>
• DPD history

<br><br>

Risk Categories:

<br><br>

✅ P1 → Low Risk <br>
🟢 P2 → Moderate Risk <br>
🟠 P3 → High Risk <br>
🔴 P4 → Very High Risk

</div>
""", unsafe_allow_html=True)

# =========================================
# BUSINESS KPIs
# =========================================

st.markdown("---")

st.subheader("🏦 Banking Risk KPIs")

b1, b2 = st.columns(2)

with b1:

    st.markdown("""
    <div class='card'>

    <h4>📈 Key Business Metrics</h4>

    • Loan Approval Risk Prediction <br>
    • Customer Credit Behavior Analysis <br>
    • Financial Risk Segmentation <br>
    • Delinquency Pattern Detection <br>
    • Smart Lending Decision Support

    </div>
    """, unsafe_allow_html=True)

with b2:

    st.markdown("""
    <div class='card'>

    <h4>🤖 Machine Learning Highlights</h4>

    • Random Forest Classification Model <br>
    • Feature Engineering Applied <br>
    • Missing Value Handling <br>
    • Correlation Reduction <br>
    • Risk Rule Engine Integration

    </div>
    """, unsafe_allow_html=True)

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.caption(
    "AI Loan Risk Analyzer | PRP Banking Analytics Project 2026"
)