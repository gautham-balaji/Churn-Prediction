import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="🛡️ XGBoost", page_icon="🛡️")
st.title("🛡️ XGBoost Churn Prediction")

model = joblib.load("models/xgboost_tuned.pkl")

final_features=['RowNumber', 'CustomerId', 'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
                        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                        'Satisfaction', 'Points', 'Geo_Germany', 'Geo_Spain',
                        'Card_Gold', 'Card_Platinum', 'Card_Silver']

st.subheader("Enter Customer Details (18 features)")

RowNumber = st.number_input("Row Number", min_value=1)
CustomerId = st.number_input("Customer ID", min_value=1)
CreditScore = st.number_input("Credit Score", min_value=300, max_value=900)
Gender = st.selectbox("Gender", ["Male", "Female"])
Gender = 1 if Gender == "Male" else 0
Age = st.number_input("Age", min_value=18)
Tenure = st.slider("Tenure", 0, 10)
Balance = st.number_input("Balance")
NumOfProducts = st.slider("Number of Products", 1, 4)
HasCrCard = st.selectbox("Has Credit Card?", [0, 1])
IsActiveMember = st.selectbox("Is Active Member?", [0, 1])
EstimatedSalary = st.number_input("Estimated Salary")
Satisfaction = st.slider("Satisfaction Score", 1, 5)
Points = st.number_input("Points Earned")
Geography = st.selectbox("Geography", ["Germany", "Spain", "France"])
Geo_Germany = 1 if Geography == "Germany" else 0
Geo_Spain = 1 if Geography == "Spain" else 0
CardType = st.selectbox("Card Type", ["Gold", "Platinum", "Silver", "Diamond"])
Card_Gold = 1 if CardType == "Gold" else 0
Card_Platinum = 1 if CardType == "Platinum" else 0
Card_Silver = 1 if CardType == "Silver" else 0

input_data = np.array([[RowNumber, CustomerId, CreditScore, Gender, Age, Tenure, Balance,
                        NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary,
                        Satisfaction, Points, Geo_Germany, Geo_Spain,
                        Card_Gold, Card_Platinum, Card_Silver]])

if st.button("Predict"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    st.success("Customer is likely to CHURN" if pred == 1 else "Customer will NOT churn")
    st.info(f"Churn Probability: {prob:.2%}")

    st.subheader("🔍 Why This Prediction? (SHAP Explanation)")

    input_df = pd.DataFrame(input_data, columns=final_features)
    explainer = shap.Explainer(model, input_df)
    shap_values = explainer(input_df)

    plt.figure(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0])
    st.pyplot(plt.gcf())
