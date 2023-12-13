# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:00:10 2023

@author: PC
"""

# Import required libraries
import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import minmax_scale, LabelEncoder
from PIL import Image

model = load('Titanic_Survival_Model.joblib')

# Function to preprocess categorical data
def preprocess_cat(df):
    lb = LabelEncoder()
    
    # Encode the sex column
    df['Sex'] = lb.fit_transform(df['Sex'])
    return df
# Function to preprocess numerical data
def preprocess_num(df):
    num_cols = ['Age']
    df[num_cols] = minmax_scale(df[num_cols])
    return df
    
# Function to preprocess input data
def preprocess_input(input_df):
    input_df = preprocess_cat(input_df)
    input_df = preprocess_num(input_df)
    return input_df

# Main function to create web app interface
def main():
    st.title('Titanic Survivor Prediction App')
    st.write('This App determines if a Titanic ship passenger survived or died in the illfated ship capsize that occured April, 1912, depending on some available features. ')
    img = Image.open('Titanic-Sinking.jpg')
    st.image(img, width=500)
    st.sidebar.text("Number of deaths: 1,496\nDates: 14 Apr 1912 – 15 Apr 1912\nLocation: North Atlantic Ocean\nCause: Collision with iceberg on 14 April")
    input_data = {} # Dictionary to store input data
    col1, col2 = st.columns(2)

    with col1:
        # Collect user data
        input_data['Pclass'] = st.number_input('Passenger Class', step=1)
        input_data['Sex'] = st.number_input("Passenger's Gender, if male; 1, if female; 0", min_value=0, max_value=1)
        input_data['Age'] = st.number_input('Passenger Age', step=1)
        
    with col2:
        input_data['SibSp'] = st.number_input("Passenger's siblings onboard", step=1)
        input_data['Parch'] = st.number_input("Passenger's Parents and Children Onboard", step=1)
        
        
    input_df = pd.DataFrame([input_data]).reset_index(drop=True)
    st.write(input_df) # Display collected data
    if st.button('PREDICT'):
        final_df = preprocess_input(input_df)
        prediction = model.predict(final_df)[0]
        
        # Display prediction result
        if prediction == 1:
            st.write("There's a likelihood that this passenger survived the capsize")
        elif prediction == 0:
            st.write("There's a likelihood that this passenger didn't survive the capsize")
        else:
            st.write('Nothing to predict')
# Run the main function when this scipt is executed directly
if __name__ == '__main__':
    main()


