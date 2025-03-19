import pandas as pd
import gdown
from app.config import DATA_PATH, GOOGLE_DRIVE_LINKS
from sklearn.preprocessing import LabelEncoder
import os


def download_file(file_path, url):
    """Downloads a file from Google Drive """
    if not os.path.exists(file_path):
        gdown.download(url, file_path, quiet=False)
    else:
        print(f"{file_path} already exists.")


def load_data(data_path=DATA_PATH):
    """Downloads necessary data from Google Drive and loads CSV files into DataFrames."""

    files = {
        "stores": f"{data_path}stores.csv",  # Path for stores data
        "items": f"{data_path}items.csv",  # Path for items data
        "transactions": f"{data_path}transactions.csv",  # Path for transactions data
        "oil": f"{data_path}oil.csv",  # Path for oil prices data
        "holidays_events": f"{data_path}holidays_events.csv",  # Path for holidays and events data
        "train": f"{data_path}train.csv"  # Path for training data
    }

    # Download the files if they don't already exist locally
    for key, file_path in files.items():
        download_file(file_path, GOOGLE_DRIVE_LINKS[key])
        # Load each downloaded CSV file into a pandas DataFrame
    df_stores = pd.read_csv(files["stores"])  # Stores data
    df_items = pd.read_csv(files["items"])  # Items data
    df_transactions = pd.read_csv(files["transactions"])  # Transactions data
    df_oil = pd.read_csv(files["oil"])  # Oil prices data
    df_holidays = pd.read_csv(files["holidays_events"])  # Holidays and events data


    # Filter stores in Guayas
    df_guayas_stores = df_stores[df_stores["state"] == 'Guayas']['store_nbr'].unique()

    # Define the item families to keep
    item_families = ['GROCERY I', 'BEVERAGES', 'CLEANING']

    # Filter df_items to keep only rows where 'family' is in item_families
    df_filtered_items = df_items[df_items['family'].isin(item_families)]

    # Chunk size
    chunk_size = 10 ** 6

    # List to store filtered chunks
    filtered_chunks = []

    # Read train dataset in chunks
    for chunk in pd.read_csv(files["train"], chunksize=chunk_size,parse_dates=["date"],low_memory=False):

        # Filter for stores in Guayas
        chunk_filtered = chunk[chunk["store_nbr"].isin(df_guayas_stores)]

        # Filter for the date range
        chunk_filtered = chunk_filtered[(chunk_filtered["date"] <= '2014-03-31')]

        # Merge with filtered items (important step!)
        chunk_filtered = chunk_filtered.merge(df_filtered_items, on="item_nbr", how="inner")

        # Append the filtered chunk
        filtered_chunks.append(chunk_filtered)

    # Concatenate all filtered chunks
    df_g = pd.concat(filtered_chunks, ignore_index=True)

    # Clean up memory
    del filtered_chunks

    return  df_stores, df_items, df_transactions, df_oil, df_holidays, df_g


def preprocess_input_data(df_g, store_id, item_id, date):
    print(f"🔍 Processing Store: {store_id}, Item: {item_id}, Date: {date}")  # Debugging

    # Convert 'date' column to datetime format
    df_g['date'] = pd.to_datetime(df_g['date'])

    # Get the full date range in the dataset
    min_date = df_g['date'].min()
    max_date = df_g['date'].max()

    # Create a full date range DataFrame
    full_date_range = pd.DataFrame({'date': pd.date_range(start=min_date, end=max_date, freq='D')})

    # Get all (store_nbr, item_nbr) combinations
    store_item_combinations = df_g[['store_nbr', 'item_nbr']].drop_duplicates()

    # Perform a cross join to get all (store, item, date) combinations
    all_combinations = store_item_combinations.merge(full_date_range, how='cross')

    # Merge with original data to fill missing dates
    df_filled = all_combinations.merge(df_g, on=['store_nbr', 'item_nbr', 'date'], how='left')

    # Fill missing sales values with 0
    df_filled['unit_sales'] = df_filled['unit_sales'].fillna(0)

    # Fill missing 'onpromotion' values with False
    df_filled["onpromotion"] = df_filled["onpromotion"].fillna(False)

    # Ensure no negative unit sales
    df_filled["unit_sales"] = df_filled["unit_sales"].apply(lambda x: max(x, 0))

    # Impute missing 'family', 'class', 'perishable' using mode
    for col in ["family", "class", "perishable"]:
        df_filled[col] = df_filled.groupby(["store_nbr", "item_nbr"])[col].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else -1 if col == "class" else 0)
        )

    # Create time-based features
    df_filled["year"] = df_filled["date"].dt.year
    df_filled["month"] = df_filled["date"].dt.month
    df_filled["day"] = df_filled["date"].dt.day
    df_filled["day_of_week"] = df_filled["date"].dt.dayofweek

    # Create lag features (previous sales)
    df_filled["lag_1"] = df_filled.groupby(["store_nbr", "item_nbr"])["unit_sales"].shift(1)
    df_filled["lag_7"] = df_filled.groupby(["store_nbr", "item_nbr"])["unit_sales"].shift(7)
    df_filled["lag_14"] = df_filled.groupby(["store_nbr", "item_nbr"])["unit_sales"].shift(14)

    # Rolling averages & standard deviation
    df_filled["rolling_avg_7"] = df_filled.groupby(["store_nbr", "item_nbr"])["unit_sales"].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    df_filled["rolling_stdv_7"] = df_filled.groupby(["store_nbr", "item_nbr"])["unit_sales"].transform(
        lambda x: x.rolling(window=7, min_periods=1).std()
    )

    # Drop rows with NaN values after creating lag features
    df_filled = df_filled.dropna().reset_index(drop=True)

    # Encode categorical columns
    for col in ["family", "class"]:
        le = LabelEncoder()
        df_filled[col] = le.fit_transform(df_filled[col])

    # Drop unnecessary columns
    df_filled = df_filled.drop(columns=["id"], errors="ignore")

    # ✅ **NOW filter for the requested store, item, and date**
    df_filtered = df_filled[
        (df_filled["store_nbr"] == store_id) &
        (df_filled["item_nbr"] == item_id) &
        (df_filled["date"] == pd.to_datetime(date))
    ]

    if df_filtered.empty:
        print("⚠️ Warning: No matching data found! Check store/item selection.")
        return None  # Return None if no data matches

    # Drop unnecessary columns before passing to the model
    df_filtered = df_filtered.drop(columns=["unit_sales", "date"], errors="ignore")

    print("✅ Filtered Data Sample for Prediction:")
    print(df_filtered.head())  # Debugging

    return df_filtered

