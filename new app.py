import streamlit as st
from joblib import load
import pandas as pd

# Load your trained Random Forest model
rf_model = load(r'C:\Users\Windows\Downloads\rf_model.joblib')

# Define the Streamlit app
def main():
    st.title('Random Forest Model Deployment')

    # You can add inputs based on the features your model uses
    input_feature_1 = st.number_input('Input Feature 1')
    input_feature_2 = st.number_input('Input Feature 2')
    # Add as many input features as required

    # Button to make predictions
    if st.button('Predict'):
        # Create a DataFrame with the input features
        input_df = pd.DataFrame([[input_feature_1, input_feature_2]],
                                columns=['feature_1', 'feature_2'])
        # Make prediction
        prediction = rf_model.predict(input_df)
        st.write(f'Prediction: {prediction[0]}')

if __name__ == '__main__':
    main()
