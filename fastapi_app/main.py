import streamlit as st
import numpy as np
import pandas as pd
import joblib  


model = joblib.load('model_1.pkl')

st.title('ML Model Deployment with 133 Features')


uploaded_file = st.file_uploader('sample.csv', type="csv")

if uploaded_file is not None:
    try:
        
        input_data = pd.read_csv(uploaded_file)
        
        # according to our features in the training
        if input_data.shape[1] == 132:
            # Convert to numpy array for prediction
            input_data = input_data.values
            
            # Model inference for both class predictions and probabilities
            predictions = model.predict(input_data)  # For class predictions
            probabilities = model.predict_proba(input_data)  # For probabilities
            
            # Display predictions and probabilities for each sample
            for i, (prediction, probability) in enumerate(zip(predictions, probabilities)):
                st.write(f'Sample {i + 1}:')
                st.write(f' - Predicted Class: {prediction}')
                st.write(f' - Prediction Probabilities: {probability}')
        else:
            st.write(f"Error: The uploaded file must contain 133 columns, but got {input_data.shape[1]} columns.")
    
    except Exception as e:
        st.write(f"Error: {e}")
