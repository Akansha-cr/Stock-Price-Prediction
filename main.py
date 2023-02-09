import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the stock data
df = pd.read_csv('ASIANPAINT.csv', sep =",")

# Plot the stock price
st.line_chart(df.set_index('date')['close'])

# Train the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(df[['open', 'high', 'low']], df['close'])
# Predict the stock price
def predict_price(model, data):
    return model.predict(data)

# Get the input data
open_price = st.number_input('Open price')
high_price = st.number_input('High price')
low_price = st.number_input('Low price')

# Make the prediction
if st.button('Predict'):
    prediction = predict_price(model, np.array([[open_price, high_price, low_price]]))
    st.write('The predicted stock price is: ', prediction[0])
