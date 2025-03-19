

import streamlit as st  # Streamlit is used for creating the web interface
import matplotlib.pyplot as plt
import pandas as pd
from app.config import DATA_PATH, MODEL_PATH  # Import paths for data and model
from data.data_utils import load_data, preprocess_input_data  # Import functions to load and preprocess data
from model.model_utils import load_model, predict  # Import functions to load the model and make predictions
import datetime  # Used for handling date inputs
from data.data_loader import get_df_g  # Import df_g loader

def main():
    st.title("Corporación Favorita Guayas Sales Forecasting")

    df_g = get_df_g()  # Load df_g here
    # Load data and model
    df_stores, df_items, df_transactions, df_oil, df_holidays, df_train = load_data(DATA_PATH)
    model = load_model(MODEL_PATH)

    # UI components for selecting store and item
    store_ids = df_g['store_nbr'].unique()  # Get unique store IDs
    store_id = st.selectbox("Select Store", store_ids)

    # Get the available items for the selected store
    available_items = df_g[df_g['store_nbr'] == store_id]['item_nbr'].unique()
    item_id = st.selectbox("Select Item", available_items)

    # After this, the rest of the logic can proceed with the selected store_id and item_id

    default_date = datetime.date(2014, 1, 1)  # Default to March 1, 2014
    min_date = datetime.date(2014, 1, 1)
    max_date = datetime.date(2014, 3, 31)
    date = st.date_input("Forecast Date",value=default_date,min_value=min_date,max_value=max_date)


    # Run prediction when button is clicked
    if st.button("Get Forecast"):
        input_data = preprocess_input_data(df_train, store_id, item_id, date)

        # Check if input_data is empty (no available data for the store-item-date combination)
        if input_data is None or input_data.empty:
            st.warning(f"⚠️ Prediction not available for Store {store_id} and Item {item_id} on {date}.")
        else:
            prediction = predict(model, input_data)
            st.success(f"✅ Predicted Sales for {date}: {prediction[0]:.2f}")


if __name__ == "__main__":
    main()