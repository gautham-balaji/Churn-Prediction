import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="🔍 Logistic Regression", page_icon="🔍")
st.title("🔍 Logistic Regression Churn Prediction")

model = joblib.load("models/logistic_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl") 

final_features = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'Point Earned',
                  'Geography_Germany', 'Geography_Spain', 'Gender', 'HasCrCard', 'IsActiveMember',
                  'Satisfaction Score', 'Card Type_GOLD', 'Card Type_PLATINUM', 'Card Type_SILVER']

st.subheader("Enter Customer Details (15 features)")

Age = st.number_input("Age", min_value=18, max_value=100)
Tenure = st.slider("Tenure", 0, 10)
Balance = st.number_input("Balance")
NumOfProducts = st.slider("Number of Products", 1, 4)
EstimatedSalary = st.number_input("Estimated Salary")
Points = st.number_input("Points Earned")
Geo = st.selectbox("Geography", ["Germany", "Spain", "France"])
Geo_Germany = 1 if Geo == "Germany" else 0
Geo_Spain = 1 if Geo == "Spain" else 0
Gender = st.selectbox("Gender", ["Male", "Female"]) 
Gender = 1 if Gender == "Male" else 0
HasCrCard = st.selectbox("Has Credit Card?", [0, 1])
IsActiveMember = st.selectbox("Is Active Member?", [0, 1])
Satisfaction = st.slider("Satisfaction Score", 1, 5)
CardType = st.selectbox("Card Type", ["Gold", "Platinum", "Silver", "Diamond"])
Gold = 1 if CardType == "Gold" else 0
Plat = 1 if CardType == "Platinum" else 0
Silver = 1 if CardType == "Silver" else 0

raw_input = np.array([[Age, Tenure, Balance, NumOfProducts, EstimatedSalary, Points,
                       Geo_Germany, Geo_Spain, Gender, HasCrCard, IsActiveMember,
                       Satisfaction, Gold, Plat, Silver]])

scaled_input = scaler.transform(raw_input)

if st.button("Predict"):
    pred = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    result = "Customer is likely to CHURN" if pred == 1 else "Customer will NOT churn"
    st.success(result)
    st.info(f"Churn Probability: {prob:.2%}")

    st.subheader("🔍 Why This Prediction? (SHAP Explanation)")

    input_df = pd.DataFrame(scaled_input, columns=final_features)
    explainer = shap.Explainer(model, input_df)
    shap_values = explainer(input_df)

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], max_display=15, show=False)
    st.pyplot(fig)


