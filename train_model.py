# =========================================
# FILE: train_model.py
# =========================================

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (

    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix

)

# =========================================
# BASE DIRECTORY
# =========================================

BASE_DIR = os.getcwd()

# =========================================
# FILE PATHS
# =========================================

DATA_PATH = os.path.join(
    BASE_DIR,
    "loan_data.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model.joblib"
)

# =========================================
# CHECK DATASET
# =========================================

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"loan_data.csv not found at: {DATA_PATH}"
    )

# =========================================
# LOAD DATA
# =========================================

df = pd.read_csv(DATA_PATH)

print("\n✅ Dataset Loaded Successfully")

# =========================================
# REMOVE DUPLICATES
# =========================================

df = df.drop_duplicates()

print("✅ Duplicate Records Removed")

# =========================================
# REMOVE HIGH MISSING COLUMNS
# =========================================

missing_percent = (
    df.isnull().sum() / len(df)
) * 100

drop_cols = missing_percent[
    missing_percent > 45
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

print("✅ Unwanted Columns Removed")

# =========================================
# TARGET VARIABLE
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
# ENCODE CATEGORICAL COLUMNS
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

print("✅ Categorical Encoding Completed")

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

print("✅ Feature Engineering Completed")

# =========================================
# IMPORTANT FEATURES
# =========================================

selected_columns = [

    "Credit_Score",

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
# FEATURE SCALING
# =========================================

scaler = RobustScaler()

X_scaled = scaler.fit_transform(X)

print("✅ Feature Scaling Completed")

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

print("✅ Train Test Split Completed")

# =========================================
# RANDOM FOREST MODEL
# =========================================

model = RandomForestClassifier(

    n_estimators=80,

    max_depth=10,

    min_samples_split=10,

    min_samples_leaf=4,

    max_features="sqrt",

    class_weight="balanced",

    random_state=42
)

# =========================================
# TRAIN MODEL
# =========================================

model.fit(
    X_train,
    y_train
)

print("✅ Model Training Completed")

# =========================================
# PREDICTIONS
# =========================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)

# =========================================
# EVALUATION METRICS
# =========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(

    y_test,
    y_pred,

    average="weighted"
)

recall = recall_score(

    y_test,
    y_pred,

    average="weighted"
)

f1 = f1_score(

    y_test,
    y_pred,

    average="weighted"
)

roc_auc = roc_auc_score(

    y_test,
    y_prob,

    multi_class="ovr"
)

# =========================================
# PRINT METRICS
# =========================================

print("\n====================================")
print(" MODEL EVALUATION METRICS ")
print("====================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

# =========================================
# CLASSIFICATION REPORT
# =========================================

print("\n====================================")
print(" CLASSIFICATION REPORT ")
print("====================================\n")

print(

    classification_report(
        y_test,
        y_pred
    )
)

# =========================================
# CONFUSION MATRIX
# =========================================

print("\n====================================")
print(" CONFUSION MATRIX ")
print("====================================\n")

print(

    confusion_matrix(
        y_test,
        y_pred
    )
)

# =========================================
# FEATURE IMPORTANCE
# =========================================

importance_df = pd.DataFrame({

    "Feature": selected_columns,

    "Importance": model.feature_importances_

})

importance_df = importance_df.sort_values(

    by="Importance",
    ascending=False
)

print("\n====================================")
print(" FEATURE IMPORTANCE ")
print("====================================\n")

print(importance_df)

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
    MODEL_PATH
)

print(f"\n✅ model.joblib saved at: {MODEL_PATH}")
