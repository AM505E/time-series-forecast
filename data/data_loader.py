import pandas as pd
from app.config import DATA_PATH, GOOGLE_DRIVE_LINKS
from data.data_utils import load_data  # Import existing load_data function

def get_df_g():
    """Loads and returns df_g."""
    df_stores, df_items, df_transactions, df_oil, df_holidays, df_g = load_data(DATA_PATH)
    return df_g  # Only return df_g
