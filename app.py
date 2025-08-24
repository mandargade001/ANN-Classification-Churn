import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

## Load the model

model=tf.keras.models.load_model('model.h5')

## Load the encoders, scaler

with open('onehot_encoder_geo.pkl', 'rb') as file:
    geo_encoder = pickle.load(file)

with open('gender_encoder.pkl', 'rb') as file:
    gender_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## Streamlit app

st.title('Customer Churn Prediction')

### take the input from the user by providing way to input data

# CreditScore,
credit_score = st.number_input("Credit Score", min_value=0.0)
# Geography,
geography = st.selectbox("Geography", geo_encoder.categories_[0])
# Gender,
gender = st.selectbox("Gender", gender_encoder.classes_)
# Age,
age = st.slider("Age", min_value=18, max_value=90, value=18)
# Tenure,
tenure = st.slider('Tenure', 0, 10, 5)
# Balance,
balance = st.number_input('Balance', min_value=0.0)

# NumOfProducts,
num_of_products = st.slider('Number of Products', 1, 4, 1)
# HasCrCard,
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
# IsActiveMember,
is_active_member = st.selectbox('Is Active Member', [0, 1])
# EstimatedSalary
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)



## create a dictionary for the input data to pass to model
input_data = {
    'CreditScore': credit_score,
    'Gender': gender_encoder.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_credit_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}
input_df = pd.DataFrame([input_data])


encoded_geography = geo_encoder.transform([[geography]])
encoded_geography_df = pd.DataFrame(encoded_geography.toarray(), columns=geo_encoder.get_feature_names_out(['Geography']))

## Combine one-hot encoded column
full_input_data = pd.concat([input_df.reset_index(drop=True), encoded_geography_df], axis=1)

## Scale the data

scaled_data = scaler.transform(full_input_data)


## Make prediction
prediction = model.predict(scaled_data)
prediction_prob = prediction[0][0]

st.write("Churn probability: {prediction_prob:%0.2f}")

if prediction_prob>=0.5:
    st.write("The customer is not likely to churn.")
elif prediction_prob<0.5:
    st.write("The customer is likely to churn.")

