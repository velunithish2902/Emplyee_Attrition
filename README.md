🧠 Employee Attrition Analysis and Prediction
📋 Project Overview

Employee turnover is one of the most critical challenges faced by organizations today. This project focuses on analyzing employee data to identify key factors influencing attrition and building a predictive model to forecast employees who are most likely to leave the organization.

Using machine learning techniques and interactive visualizations (via Streamlit), this project provides actionable insights to HR teams, helping them make data-driven decisions for employee retention.

🎯 Problem Statement

Employee attrition leads to increased costs, reduced productivity, and disruptions in team dynamics.
The goal of this project is to:

Analyze employee data to uncover the drivers of attrition.

Predict which employees are most likely to leave.

Provide actionable insights for retention strategies.

🧩 Business Use Cases

Employee Retention: Identify at-risk employees and take proactive actions.

Cost Optimization: Reduce recruitment, onboarding, and training expenses.

Workforce Planning: Use predictive insights to align HR strategies with company goals.

⚙️ Approach

Data Collection & Preprocessing

Cleaned and prepared data (handled missing values, categorical encoding, outliers).

Feature selection and transformation.

Exploratory Data Analysis (EDA)

Studied relationships between variables (age, salary, job satisfaction, overtime, etc.).

Visualized attrition trends using Matplotlib and Seaborn.

Feature Engineering

Created new variables like tenure categories and performance groups.

Model Development

Built classification models (Logistic Regression, Decision Tree, Random Forest).

Tuned hyperparameters for better accuracy.

Model Evaluation

Evaluated models using Accuracy, Precision, Recall, F1-Score, and AUC-ROC.

Deployment (Streamlit App)

Developed an interactive dashboard using Streamlit for real-time prediction and analysis.

📊 Results

Prediction Accuracy: ~85%+

Key Drivers of Attrition: Low job satisfaction, low salary, long overtime hours, poor work-life balance.

Impact: Helps HR teams prioritize retention strategies and reduce attrition costs.

📈 Evaluation Metrics
Metric	Description
Accuracy	Percentage of correct predictions
Precision	True positives out of predicted positives
Recall	True positives out of actual positives
F1-Score	Harmonic mean of precision and recall
AUC-ROC	Measures model’s ability to distinguish between classes
🧠 Predictive Use Cases
1. Employee Attrition Prediction

Predict whether an employee will leave or stay.
Features: Age, Department, JobSatisfaction, MonthlyIncome, OverTime, YearsAtCompany, etc.

2. Performance Rating Prediction

Predict employee performance using factors like JobLevel, Experience, and Education.

3. Promotion Likelihood Prediction

Predict when an employee is likely to get promoted based on tenure, performance, and job level.

🧰 Skills & Tools Used

Languages: Python
Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
Framework: Streamlit
Techniques:

Data Cleaning & Preprocessing

Exploratory Data Analysis (EDA)

Feature Engineering

Machine Learning Model Building

Model Evaluation Metrics

Dashboard Development

🗂️ Dataset

Name: Employee Attrition Dataset
Target Variable: Attrition (1 = Left, 0 = Stayed)

Sample Features:

Age

Department

DistanceFromHome

JobSatisfaction

MonthlyIncome

Overtime

YearsAtCompany

Dataset link: Employee-Attrition.csv

🖥️ Streamlit Dashboard

An interactive web app to visualize attrition trends and make predictions.

Features:

Upload new employee data for prediction

Department-wise attrition analysis

Key insights visualization

Download model results

🚀 Project Deliverables

✅ Cleaned Dataset (CSV)

✅ Preprocessing & Model Training Code (.py)

✅ Trained Model (.pkl or .joblib)

✅ Streamlit App Script (app.py)

✅ Documentation & Report (README.md)

🧮 Example Output
Employee ID	Department	Attrition Probability	Prediction
101	Sales	0.87	Likely to Leave
102	R&D	0.12	Likely to Stay
📚 References

Streamlit Documentation

Project Orientation: Employee Attrition Analysis and Prediction (Tamil)

Exploratory Data Analysis (EDA) Guide

GitHub Guide: How to Use GitHub

🧑‍🏫 Project Support Sessions

Project Doubt Clarification: Mon–Sat (4:00 PM – 5:00 PM)
👉 Book Slot

Live Evaluation Session: Mon–Sat (5:30 PM – 7:00 PM)
👉 Book Slot

🧾 Approval Workflow
Role	Name
Created By	—
Verified By	Gomathi A
Approved By	Shadiya P P, Nehlath Harmain
🏷️ Technical Tags

Data Analytics • Machine Learning • HR Analytics • EDA • Feature Engineering • Streamlit • Scikit-learn • Model Evaluation • AUC-ROC • F1-Score
