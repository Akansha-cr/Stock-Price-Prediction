import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Stock Price Prediction')
st.write('Upload a CSV file to generate Stock price Prediction:')

uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
if uploaded_file is None:
    st.write("No file uploaded. Please upload a file containing data.")
else:
    df = pd.read_csv(uploaded_file)
    if df.empty:
        st.write("The uploaded file is empty. Please upload a file containing data.")
    else:
        st.line_chart(df.set_index('Date')['Close'])

        # Train the model
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(df[['Open', 'High', 'Low']], df['Close'])

        # Predict the stock price
        def predict_price(model, data):
            return model.predict(data)

        # Get the input data
        open_price = st.number_input('Open price')
        high_price = st.number_input('High price')
        low_price = st.number_input('Low price')
        
        #plot the linear regression
        open_price = np.array(df['Open']).reshape(-1, 1)
        prediction = model.predict(open_price)
        plt.plot(open_price, prediction, color='red')
        plt.xlabel('Open price')
        plt.ylabel('Close price')

        # Show the plot
        st.pyplot()

        # Make the prediction
        if st.button('Predict'):
            prediction = predict_price(model, np.array([[open_price, high_price, low_price]]))
            st.write('The predicted stock price is: ', prediction[0])
