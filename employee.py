import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# --------------------------
# Page Setup
# --------------------------
st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")
st.title("📊 Employee Attrition Analysis and Prediction Dashboard")

# --------------------------
# Load Data and Model
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\user\nith\employee_attrition_cleaned.csv")
    df.columns = df.columns.str.strip().str.lower()  # normalize column names

    # Convert 'True'/'False' strings to 1/0 (fix for ValueError)
    df = df.replace({'True': 1, 'False': 0, 'true': 1, 'false': 0})

    return df

df = load_data()

# Load trained model, feature columns, and label encoders
with open(r"C:\Users\user\nith\attrition_model.pkl", "rb") as f:
    rf_model, feature_cols, label_encoders = pickle.load(f)

if isinstance(feature_cols, pd.Index):
    feature_cols = feature_cols.tolist()

# --------------------------
# Tabs
# --------------------------
tabs = st.tabs(["EDA & Insights", "Feature Importance", "Prediction"])

# --------------------------
# TAB 1: EDA & Insights
# --------------------------
with tabs[0]:
    st.header("Exploratory Data Analysis (EDA)")

    # Convert categorical columns for plotting
    for col in ['attrition', 'overtime'] + [c for c in df.columns if c.startswith('department_')]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Attrition Count")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.countplot(x='attrition', data=df, ax=ax)
        ax.set_title("Attrition (0 = Stay, 1 = Leave)")
        st.pyplot(fig)

        st.subheader("Monthly Income vs Attrition")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.boxplot(x='attrition', y='monthlyincome', data=df, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Attrition by Department")
        dept_cols = [c for c in df.columns if c.startswith('department_')]
        if dept_cols:
            dept_attrition = pd.DataFrame({
                'department': [c.replace('department_', '').replace('_', ' ') for c in dept_cols],
                'attrition_rate': [
                    pd.to_numeric(df[c].replace({'True': 1, 'False': 0}), errors='coerce').mean()
                    for c in dept_cols
                ]
            })
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.barplot(x='department', y='attrition_rate', data=dept_attrition, ax=ax)
            ax.set_ylabel("Attrition Rate")
            st.pyplot(fig)

        st.subheader("Attrition vs Overtime")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.countplot(x='overtime', hue='attrition', data=df, ax=ax)
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    numeric_cols = df.select_dtypes(include='number').columns
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# --------------------------
# TAB 2: Feature Importance
# --------------------------
with tabs[1]:
    st.header("Top Features Influencing Attrition")
    importances = rf_model.feature_importances_
    features = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=features[:15], y=features.index[:15], ax=ax, palette="viridis")
    ax.set_title("Top 15 Important Features")
    st.pyplot(fig)

# --------------------------
# TAB 3: Prediction
# --------------------------
with tabs[2]:
    st.header("Predict Employee Attrition")

    # Input Form
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        monthly_income = st.slider("Monthly Income", min_value=1000, max_value=20000, value=5000, step=500)
        distance = st.slider("Distance From Home (km)", min_value=0, max_value=100, value=10, step=1)
        jobsatisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2)
    with col2:
        overtime = st.selectbox("Overtime", ["Yes", "No"])
        job_level = st.number_input("Job Level", min_value=1, max_value=5, value=2)
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)

        # Dynamically get department options
        department_cols = [c for c in feature_cols if c.startswith('department_')]
        departments = [c.replace('department_', '').replace('_', ' ') for c in department_cols]
        department = st.selectbox("Department", departments)

        # Dynamically get job role options
        jobrole_cols = [c for c in feature_cols if c.startswith('jobrole_')]
        jobroles = [c.replace('jobrole_', '').replace('_', ' ') for c in jobrole_cols]
        jobrole = st.selectbox("Job Role", jobroles)

    overtime_val = 1 if overtime == "Yes" else 0

    if st.button("Predict Attrition"):
        # Create empty DataFrame
        user_data = pd.DataFrame([[0]*len(feature_cols)], columns=feature_cols)

        # Fill numeric fields
        numeric_mapping = {
            'age': age,
            'monthlyincome': monthly_income,
            'distancefromhome': distance,
            'jobsatisfaction': jobsatisfaction,
            'overtime': overtime_val,
            'joblevel': job_level,
            'yearsatcompany': years_at_company
        }
        for col, val in numeric_mapping.items():
            if col in user_data.columns:
                user_data[col] = val

        # Fill one-hot encoded fields
        dept_col_name = f"department_{department.lower().replace(' ', '_')}"
        if dept_col_name in user_data.columns:
            user_data[dept_col_name] = 1

        jobrole_col_name = f"jobrole_{jobrole.lower().replace(' ', '_')}"
        if jobrole_col_name in user_data.columns:
            user_data[jobrole_col_name] = 1

        # Predict
        prediction = rf_model.predict(user_data)[0]
        proba = rf_model.predict_proba(user_data)[0][prediction]

        if prediction == 1:
            st.error(f"Prediction: **Leave** (Probability: {proba:.2f})")
        else:
            st.success(f"Prediction: **Stay** (Probability: {proba:.2f})")
