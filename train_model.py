import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================================
# LOAD DATA
# =========================================

df = pd.read_csv(
    r"C:\Loan_Risk_Project\loan_data.csv"
)

print("✅ Dataset Loaded")

# =========================================
# REMOVE DUPLICATES
# =========================================

df = df.drop_duplicates()

# =========================================
# DROP HIGH MISSING COLUMNS
# =========================================

missing_percent = (
    df.isnull().sum() / len(df)
) * 100

drop_cols = missing_percent[
    missing_percent > 40
].index

df.drop(
    columns=drop_cols,
    inplace=True
)

print("✅ High Missing Columns Removed")

# =========================================
# REMOVE UNWANTED COLUMNS
# =========================================

remove_cols = [

    "PROSPECTID",
    "Credit_Score",
    "pct_closed_tl",
    "pct_of_active_TLs_ever",
    "Age_Oldest_TL"

]

existing = [

    col for col in remove_cols

    if col in df.columns
]

df.drop(
    columns=existing,
    inplace=True
)

# =========================================
# TARGET
# =========================================

TARGET = "Approved_Flag"

# =========================================
# ENCODE TARGET
# =========================================

target_encoder = LabelEncoder()

df[TARGET] = target_encoder.fit_transform(
    df[TARGET]
)

# =========================================
# ENCODE CATEGORICAL
# =========================================

cat_cols = df.select_dtypes(
    include="object"
).columns.tolist()

if TARGET in cat_cols:

    cat_cols.remove(TARGET)

for col in cat_cols:

    le = LabelEncoder()

    df[col] = le.fit_transform(
        df[col].astype(str)
    )

print("✅ Encoding Done")

# =========================================
# FEATURE ENGINEERING
# =========================================

df["payment_score"] = (

    df["Tot_Missed_Pmnt"]

    +

    df["num_times_30p_dpd"]

    +

    df["num_times_60p_dpd"]

)

df["delinq_score"] = (

    df["num_times_delinquent"]

    +

    df["num_deliq_6mts"]

    +

    df["num_deliq_12mts"]

)

df["enquiry_score"] = (

    df["tot_enq"]

    +

    df["enq_L3m"]

    +

    df["enq_L6m"]

)

print("✅ Feature Engineering Done")

# =========================================
# IMPORTANT FEATURES
# =========================================

selected_columns = [

    "AGE",

    "NETMONTHLYINCOME",

    "Total_TL",

    "Tot_Active_TL",

    "Time_With_Curr_Empr",

    "Tot_Missed_Pmnt",

    "num_times_delinquent",

    "tot_enq",

    "num_times_30p_dpd",

    "num_times_60p_dpd",

    "payment_score",

    "delinq_score",

    "enquiry_score"

]

selected_columns = [

    col for col in selected_columns

    if col in df.columns
]

X = df[selected_columns]

y = df[TARGET]

# =========================================
# HANDLE MISSING VALUES
# =========================================

imputer = KNNImputer(
    n_neighbors=5
)

X = pd.DataFrame(

    imputer.fit_transform(X),

    columns=X.columns
)

print("✅ Missing Values Handled")

# =========================================
# SCALING
# =========================================

scaler = RobustScaler()

X_scaled = scaler.fit_transform(X)

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)

# =========================================
# MODEL
# =========================================

model = RandomForestClassifier(

    n_estimators=300,

    max_depth=10,

    min_samples_split=5,

    class_weight="balanced",

    random_state=42
)

# =========================================
# TRAIN
# =========================================

model.fit(
    X_train,
    y_train
)

print("✅ Model Trained")

# =========================================
# ACCURACY
# =========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"\n🎯 Accuracy: {accuracy:.2%}")

# =========================================
# SAVE PIPELINE
# =========================================

pipeline = {

    "model": model,

    "scaler": scaler,

    "imputer": imputer,

    "selected_columns": selected_columns,

    "target_encoder": target_encoder

}

joblib.dump(

    pipeline,

    r"C:\Loan_Risk_Project\model.pkl"
)

print("\n✅ model.pkl Saved Successfully")