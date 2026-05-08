# =========================================
# LOAD / TRAIN MODEL
# =========================================

MODEL_PATH = "model.pkl"

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

    st.error(f"❌ Error loading model: {e}")
    st.stop()
