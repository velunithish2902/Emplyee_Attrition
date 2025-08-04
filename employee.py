import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Employee-Attrition - Employee-Attrition.csv")

df = load_data()
st.title("\U0001F468‍\U0001F4BC Employee Attrition Analysis and Prediction")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["DASHBOARD", "ATTRITION PREDICTION"])

# Label Encoding
def preprocess_data(df):
    df = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    return df

# Page 1: EDA
if page == "DASHBOARD":
    st.header("\U0001F4CA Exploratory Data Analysis")
    st.subheader("Dataset Overview")
    st.dataframe(df.head(10))

    if st.checkbox("Show Shape and Nulls"):
        st.write("Shape:", df.shape)
        st.write("Missing Values:\n", df.isnull().sum())

    st.subheader("Attrition Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Attrition", data=df, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Attrition by Department")
    fig2, ax2 = plt.subplots()
    sns.countplot(x="Department", hue="Attrition", data=df, ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    df_encoded = preprocess_data(df)
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    sns.heatmap(df_encoded.corr(), cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

# Page 2: Prediction
elif page =="ATTRITION PREDICTION":
    st.header("\U0001F52E Predict Employee Attrition")

    # Preprocessing
    df_model = preprocess_data(df)
    X = df_model.drop(columns=["Attrition", "EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"])
    y = df_model["Attrition"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # User Input
    st.subheader("Make a Prediction")
    input_data = {}

    for col in X.columns:
        if col == "MaritalStatus":
            input_data[col] = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"])
        elif col == "BusinessTravel":
            input_data[col] = st.selectbox("Business Travel", options=["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        elif col == "Age":
            input_data[col] = st.selectbox("🎯 Select Age", sorted(df['Age'].unique()))
        elif col == 'DailyRate':
            input_data[col] = st.selectbox("✅ Select DailyRate (greater than 100)", list(range(101, 1001)))
        elif col == 'Department':
            input_data[col] = st.selectbox("Department", options=['Sales', 'Research & Development', 'Human Resources'])
        elif col == 'MonthlyIncome':
            input_data[col] = st.selectbox("MonthlyIncome", sorted(df['MonthlyIncome'].unique()))
        elif col == 'OverTime':
            input_data[col] = 'Yes' if st.checkbox("OverTime (Check if Yes)") else 'No'
        else:
            input_data[col] = st.number_input(f"{col}", value=float(X[col].mean()))

    input_df = pd.DataFrame([input_data])

    # Encode input data
    input_df_encoded = input_df.copy()
    for col in input_df_encoded.select_dtypes(include='object').columns:
        le = LabelEncoder()
        le.fit(df[col])
        input_df_encoded[col] = le.transform(input_df_encoded[col])

    # 🔧 Align with training columns
    input_df_encoded = input_df_encoded.reindex(columns=X_train.columns, fill_value=0)

    # Accuracy
    st.subheader("Model Accuracy on Test Data")
    acc = model.score(X_test, y_test)
    st.write(f"\U0001F539 Accuracy: {acc:.2f}")

    # Predict
    if st.button("\U0001F50D Predict"):
        prediction = model.predict(input_df_encoded)[0]
        prediction_proba = model.predict_proba(input_df_encoded)[0][1]

        if prediction == 1:
            st.markdown(
                f"<h3 style='color:red;'>⚠️ The employee is likely to leave. (Risk: {prediction_proba * 100:.2f}%)</h3>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<h3 style='color:green;'>✅ The employee is likely to stay. (Confidence: {(1 - prediction_proba) * 100:.2f}%)</h3>",
                unsafe_allow_html=True)
