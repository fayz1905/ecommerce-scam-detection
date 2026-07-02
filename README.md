# E-Commerce Fraud Detection Tool

## Setup
```bash
cd D:\ecommerce-scam-detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

## Run Streamlit
```bash
streamlit run app.py
```

## Project Structure
- notebooks/ - 4 EDA & training notebooks
- src/ - Python modules for fraud detection
- models/ - trained models (local only)
- app.py - web dashboard