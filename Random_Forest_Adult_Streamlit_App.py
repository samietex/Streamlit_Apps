# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 13:16:04 2023

@author: Windows
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import load
import streamlit as st

# Load the pre-trained model
model = load('rf_model_scaled_adult.joblib')

def preprocess_data(df):
    # Deep copy the DataFrame to avoid modifying the original data
    processed_df = df.copy()

    # Perform label encoding and one-hot encoding
    categorical_cols = ['relationship', 'marital-status', 'education', 'workclass', 'occupation', 'race', 'gender', 'native-country']
    for col in categorical_cols:
        if col in processed_df.columns:
            lb = LabelEncoder()
            processed_df[col] = lb.fit_transform(processed_df[col])

    processed_df = pd.get_dummies(processed_df, columns=['race', 'gender', 'native-country'])

    # Scale the numerical features
    numerical_cols = [col for col in processed_df.columns if col not in ['income']]
    scaler = MinMaxScaler()
    processed_df[numerical_cols] = scaler.fit_transform(processed_df[numerical_cols])

    return processed_df

# Streamlit app
def main():
    st.title('Income Prediction App')

    # Add file uploader for CSV
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data.head())

        # Preprocess the uploaded data
        preprocessed_data = preprocess_data(data)

        # Make predictions
        predictions = model.predict(preprocessed_data)

        st.write("Predictions:")
        st.write(predictions)

if __name__ == '__main__':
    main()
