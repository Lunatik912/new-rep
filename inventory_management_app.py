import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import streamlit as st

# Title of the app
st.title('AI-Powered Inventory Management System')

# Step 1: Upload CSV file
uploaded_file = st.file_uploader("Upload your sales data (CSV)", type="csv")

if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)
    
    # Ensure the data has the correct columns
    if 'Date' in data.columns and 'Sales' in data.columns:
        data['ds'] = pd.to_datetime(data['Date'])
        data['y'] = data['Sales']
        data = data[['ds', 'y']]
        
        # Step 2: Train the Prophet model
        model = Prophet()
        model.fit(data)

        # Step 3: Make future predictions
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Step 4: Calculate optimal inventory levels
        forecasted_demand = forecast[['ds', 'yhat']].tail(30)
        safety_stock = 0.10 * forecasted_demand['yhat']
        forecasted_demand['optimal_inventory'] = forecasted_demand['yhat'] + safety_stock

        # Step 5: Display results
        st.subheader('Forecasted Demand and Optimal Inventory Levels')
        st.write(forecasted_demand[['ds', 'yhat', 'optimal_inventory']])

        # Step 6: Visualize the forecast
        fig, ax = plt.subplots()
        model.plot(forecast, ax=ax)
        plt.title('Sales Demand Forecast')
        st.pyplot(fig)

        # Optional: Visualize forecasted demand vs optimal inventory levels
        plt.figure(figsize=(10, 5))
        plt.plot(forecasted_demand['ds'], forecasted_demand['yhat'], label='Forecasted Demand', color='blue')
        plt.plot(forecasted_demand['ds'], forecasted_demand['optimal_inventory'], label='Optimal Inventory Level', color='red', linestyle='--')
        plt.title('Forecasted Demand vs Optimal Inventory Levels')
        plt.xlabel('Date')
        plt.ylabel('Units')
        plt.legend()
        st.pyplot(plt)
    else:
        st.error("The uploaded CSV must contain 'Date' and 'Sales' columns.")