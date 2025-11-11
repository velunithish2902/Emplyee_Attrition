# --------------------------
# Employee Attrition Dashboard
# --------------------------
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from joblib import load
import pickle
from pathlib import Path
import os

# --------------------------
# Streamlit Page Setup
# --------------------------
st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")
st.title("📊 Employee Attrition Analysis and Prediction Dashboard")

BASE_DIR = Path(__file__).parent

# --------------------------
# Load Data and Model
# --------------------------
@st.cache_data
def load_raw_data():
    csv_path = r"C:/Users/katlee/nithi/cleaned_employee_attrition_final.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    st.warning("⚠️ Cleaned dataset not found.")
    return pd.DataFrame()

raw_df = load_raw_data()

model_path = r"C:/Users/katlee/nithi/random_forest_model.joblib"
cols_path = r"C:/Users/katlee/nithi/feature_columns.pkl"
meta_path = r"C:/Users/katlee/nithi/metadata.json"

rf_model = load(model_path) if os.path.exists(model_path) else None
feature_cols = pickle.load(open(cols_path, "rb")) if os.path.exists(cols_path) else []
THRESHOLD = 0.35

if os.path.exists(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)
        THRESHOLD = float(meta.get("threshold", THRESHOLD))

# --------------------------
# Tabs
# --------------------------
tabs = st.tabs(["📈 EDA & Insights", "⭐ Feature Importance", "🤖 Prediction"])

# --------------------------
# EDA TAB
# --------------------------
with tabs[0]:
    st.header("Exploratory Data Analysis (EDA)")

    if raw_df.empty:
        st.info("No dataset found. Please upload or check your file path.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Attrition Count")
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.countplot(x='attrition', data=raw_df, ax=ax)
            ax.set_title("Attrition (Yes/No)")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            if 'department' in raw_df.columns:
                st.subheader("Attrition by Department")
                tmp = raw_df.copy()
                if tmp['attrition'].dtype == object:
                    tmp['attrition'] = tmp['attrition'].map({'Yes': 1, 'No': 0})
                dept_attr = tmp.groupby('department')['attrition'].mean().reset_index()
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.barplot(x='department', y='attrition', data=dept_attr, ax=ax)
                ax.set_ylabel("Attrition Rate")
                ax.tick_params(axis='x', labelrotation=20)
                st.pyplot(fig)
                plt.close(fig)

        # ✅ Gender-wise Attrition Bar Chart
        if 'gender' in raw_df.columns and 'attrition' in raw_df.columns:
            st.subheader("Gender-wise Attrition Rate")
            tmp = raw_df.copy()
            if tmp['attrition'].dtype == object:
                tmp['attrition'] = tmp['attrition'].map({'Yes': 1, 'No': 0})
            gender_attr = tmp.groupby('gender')['attrition'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x='gender', y='attrition', data=gender_attr, palette="coolwarm", ax=ax)
            ax.set_title("Attrition Rate by Gender")
            ax.set_ylabel("Attrition Rate")
            ax.set_xlabel("Gender")
            st.pyplot(fig)
            plt.close(fig)

        col3, col4 = st.columns(2)
        with col3:
            if 'monthlyincome' in raw_df.columns:
                st.subheader("Monthly Income vs Attrition")
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.boxplot(x='attrition', y='monthlyincome', data=raw_df, ax=ax)
                st.pyplot(fig)
                plt.close(fig)

        with col4:
            if 'overtime' in raw_df.columns:
                st.subheader("Attrition vs Overtime")
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.countplot(x='overtime', hue='attrition', data=raw_df, ax=ax)
                st.pyplot(fig)
                plt.close(fig)

        # ✅ Correlation Heatmap
        st.subheader("Correlation Heatmap (Numeric Columns)")
        numeric_df = raw_df.select_dtypes(include=['int64', 'float64'])
        if not numeric_df.empty:
            corr = numeric_df.corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                linewidths=0.5,
                cbar=True,
                square=True
            )
            ax.set_title("Correlation Heatmap", fontsize=14, pad=12)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("No numeric columns available for correlation heatmap.")

# --------------------------
# Feature Importance TAB
# --------------------------
with tabs[1]:
    st.header("Top Features Influencing Attrition")
    if rf_model and len(feature_cols) > 0:
        importances = rf_model.feature_importances_
        features = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=features.head(15), y=features.head(15).index, ax=ax, palette="viridis")
        ax.set_title("Top 15 Important Features")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Feature importance not available — model or columns missing.")

# --------------------------
# Prediction TAB
# --------------------------
def get_category_options(prefix: str):
    pref = prefix.lower()
    return sorted(set(
        [col[len(prefix):] for col in feature_cols if col.lower().startswith(pref)]
    ))

def build_empty_row():
    return pd.DataFrame([[0]*len(feature_cols)], columns=feature_cols)

def set_one_hot(row: pd.DataFrame, prefix: str, chosen: str):
    """Set one-hot variable (case-insensitive match)."""
    colname = prefix + chosen
    for c in row.columns:
        if c.lower() == colname.lower():
            row.at[0, c] = 1
            return
    print(f"⚠️ Missing category column: {colname}")

with tabs[2]:
    st.header("Predict Employee Attrition")

    dept_opts = get_category_options("department_")
    role_opts = get_category_options("jobrole_")
    bt_opts = get_category_options("businesstravel_")
    ms_opts = get_category_options("maritalstatus_")
    ef_opts = get_category_options("educationfield_")
    tc_opts = get_category_options("tenurecategory_")
    gender_opts = get_category_options("gender_")

    # ✅ Always show both Male & Female
    if not gender_opts or len(gender_opts) < 2:
        gender_opts = ["Male", "Female"]

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 65, 30)
        monthly_income = st.slider("Monthly Income", 1000, 20000, 5000, 500)
        distance = st.slider("Distance From Home (km)", 0, 100, 10, 1)
        jobsatisfaction = st.selectbox("Job Satisfaction (1–4)", [1,2,3,4], index=2)
        job_level = st.number_input("Job Level (1–5)", 1, 5, 2)

    with col2:
        overtime = st.selectbox("Overtime", ["Yes", "No"])
        years_at_company = st.number_input("Years at Company", 0, 40, 5)
        department = st.selectbox("Department", dept_opts or ["R&D","Sales","HR"])
        jobrole = st.selectbox("Job Role", role_opts or ["Research Scientist","Sales Executive"])
        btravel = st.selectbox("Business Travel", bt_opts or ["Travel_Rarely","Travel_Frequently","Non-Travel"])
        mstatus = st.selectbox("Marital Status", ms_opts or ["Single","Married","Divorced"])
        edfield = st.selectbox("Education Field", ef_opts or ["Life Sciences","Medical","Marketing"])
        tencat = st.selectbox("Tenure Category", tc_opts or ["0-2 yrs","3-5 yrs","6-9 yrs","10+ yrs"])
        gender = st.selectbox("Gender", gender_opts, index=1 if "Female" in gender_opts else 0)

    overtime_bin = 1 if overtime == "Yes" else 0

    if st.button("🔍 Predict Attrition"):
        if rf_model is None:
            st.error("Model file not found. Please ensure the Random Forest model is in the correct path.")
        else:
            user_row = build_empty_row()

            # Fill numeric fields
            numeric_inputs = {
                "age": age,
                "monthlyincome": monthly_income,
                "distancefromhome": distance,
                "jobsatisfaction": jobsatisfaction,
                "overtime": overtime_bin,
                "joblevel": job_level,
                "yearsatcompany": years_at_company
            }
            for k, v in numeric_inputs.items():
                if k in user_row.columns:
                    user_row.at[0, k] = v

            # One-hot encoding
            set_one_hot(user_row, "department_", department)
            set_one_hot(user_row, "jobrole_", jobrole)
            set_one_hot(user_row, "businesstravel_", btravel)
            set_one_hot(user_row, "maritalstatus_", mstatus)
            set_one_hot(user_row, "educationfield_", edfield)
            set_one_hot(user_row, "tenurecategory_", tencat)
            set_one_hot(user_row, "gender_", gender)

            # Predict
            try:
                proba = rf_model.predict_proba(user_row)[0, 1]
                pred = int(proba >= THRESHOLD)
                if pred == 1:
                    st.error(f"Prediction: **Leave** 🧳 (Probability: {proba:.2f}) • Threshold={THRESHOLD:.2f}")
                else:
                    st.success(f"Prediction: **Stay** 🏢 (Probability: {proba:.2f}) • Threshold={THRESHOLD:.2f}")
            except Exception as e:
                st.error(f"⚠️ Error during prediction: {e}")
