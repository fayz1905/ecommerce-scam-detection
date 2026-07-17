import requests
import os

FILE_URLS = {
    "models/best_model.joblib": "https://github.com/fayz1905/ecommerce-scam-detection/releases/download/v1.0/best_model.joblib",
    "models/scaler.joblib": "https://github.com/fayz1905/ecommerce-scam-detection/releases/download/v1.0/scaler.joblib",
    "models/label_encoders.joblib": "https://github.com/fayz1905/ecommerce-scam-detection/releases/download/v1.0/label_encoders.joblib",
    "models/archetype_kmeans.pkl": "https://github.com/fayz1905/ecommerce-scam-detection/releases/download/v1.0/archetype_kmeans.pkl",
    "models/archetype_scaler.pkl": "https://github.com/fayz1905/ecommerce-scam-detection/releases/download/v1.0/archetype_scaler.pkl",
    "src/cluster_risk_map.json": "https://github.com/fayz1905/ecommerce-scam-detection/releases/download/v1.0/cluster_risk_map.json",
    "data/Fraudulent_E-Commerce_Transaction_Data.csv": "https://github.com/fayz1905/ecommerce-scam-detection/releases/download/v1.0/Fraudulent_E-Commerce_Transaction_Data.csv",
}

def download_if_missing(local_path, url):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        print(f"Downloading {local_path}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
        print(f"Done: {local_path}")
    else:
        print(f"Already exists: {local_path}")

def ensure_all_files():
    for local_path, url in FILE_URLS.items():
        download_if_missing(local_path, url)