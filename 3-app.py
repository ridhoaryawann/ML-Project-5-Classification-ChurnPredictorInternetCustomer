# gender 1 = female
# churn 1 = yes

import streamlit as st 
import numpy as np
import pandas as pd 
import joblib 

scaler  = joblib.load('scaler.pkl')
model   = joblib.load('model.pkl')

st.title("Churn Prediction App")

st.write("With this app, you can estimate whether the customer will be churn or still with us ")

st.divider()

age     = st.number_input("Enter age:", min_value=10, max_value=90, value=25)
tenure  = st.number_input("Enter tenure", min_value=0, max_value=130, value=10)
monthlycharge = st.number_input("Enter monthly charge", min_value=30, max_value=150)
gender  = st.selectbox("Select gender", options=("Male","Female"))

st.divider()

predictbutton = st.button("Predict")

if predictbutton:
    gender_selected = 1 if gender == "Female" else 0

    x   = [age, gender_selected, tenure, monthlycharge]
    x1  = np.array(x)

    x_array = scaler.transform([x1])

    prediction = model.predict(x_array)[0]

    predicted  = "Churn" if prediction == 1 else "Stay"

    st.write(f"Your customer is likely to {predicted}")

else:
    st.write("Please enter all the values and click predict button")

