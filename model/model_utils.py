import os
from data.data_utils import download_file
import gdown  # For downloading from Google Drive
import xgboost as xgb
from app.config import MODEL_PATH, GOOGLE_DRIVE_LINKS_MODELS



def download_xgb_model(model_path, drive_link):
    """Downloads the model from Google Drive if it doesn't exist locally."""
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        gdown.download(drive_link, model_path, quiet=False)


def load_model(model_path=MODEL_PATH):
    """Downloads and loads a pre-trained XGBoost model."""
    # Define path to the model file
    model_file = f"{model_path}model.xgb"
    print(model_file)
    # Download if not available
    download_xgb_model(model_file, GOOGLE_DRIVE_LINKS_MODELS["xgboost_model"])

    # Load XGBoost model
    xgboost_model = xgb.XGBRegressor()
    xgboost_model.load_model(model_file)

    print("✅ Model loaded successfully!")
    return xgboost_model


def predict(model, input_data):
    """Runs prediction on input data using the pre-trained model."""
    # Ensure the required columns are removed safely
    input_data = input_data.drop(columns=['date', 'unit_sales'], errors='ignore')

    # Make prediction
    prediction = model.predict(input_data)
    return prediction
