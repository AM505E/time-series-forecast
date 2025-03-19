import os

DATA_PATH = os.path.join(os.getcwd(), "data/")

your_file_id_for_stores_csv = '1vpX19ag2tW2ik5BzoL6y2KlK8KO5v3aL'
your_file_id_for_items_csv = '11Fs0sUqeWXjBo8Ac2J3LouEWcH061hUJ'
your_file_id_for_transactions_csv = '1FdhaE96in25NvD7dX69WBnmPrnSUwlL6'
your_file_id_for_oil_csv = '1IuTclY2paVyd42R5FboERpj6Igi_ykWz'
your_file_id_for_holidays_csv = '1X5q4FlsHk7nR4owi-jrJ3JNrap2rgS4m'
your_file_id_for_train_csv = '14y0ZPZSUfgFaWXKkjZHzqPFsDZmpAt8e'

GOOGLE_DRIVE_LINKS = {
    "stores": f"https://drive.google.com/uc?id={your_file_id_for_stores_csv}",
    "items": f"https://drive.google.com/uc?id={your_file_id_for_items_csv}",
    "transactions": f"https://drive.google.com/uc?id={your_file_id_for_transactions_csv}",
    "oil": f"https://drive.google.com/uc?id={your_file_id_for_oil_csv}",
    "holidays_events": f"https://drive.google.com/uc?id={your_file_id_for_holidays_csv}",
    "train": f"https://drive.google.com/uc?id={your_file_id_for_train_csv}"
}

MODEL_PATH = "model/"

your_file_id_for_xgboost_model_xgb = "1-6X9o-lEjiToGhrMXnmpczdgtViv06Dw"
GOOGLE_DRIVE_LINKS_MODELS = {
    "xgboost_model": f"https://drive.google.com/uc?id={your_file_id_for_xgboost_model_xgb}"
}
