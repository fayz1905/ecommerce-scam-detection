import streamlit as st
import pandas as pd
import sys
import os

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from data_loader import ensure_all_files
ensure_all_files()

from fraud_score import FraudScoreEngine

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("🚨 E-Commerce Fraud Detection Dashboard")

@st.cache_resource
def load_engine():
    df = pd.read_csv("data/Fraudulent_E-Commerce_Transaction_Data.csv")
    engine = FraudScoreEngine(
        model_path="models/best_model.joblib",
        scaler_path="models/scaler.joblib",
        dataframe=df
    )
    return engine, df

with st.spinner("Loading models and data..."):
    engine, df = load_engine()

page = st.sidebar.selectbox("Pages", ["Dashboard Overview", "Check a Transaction", "Analytics", "Info"])

if page == "Dashboard Overview":
    st.header("📊 Dashboard Overview")
    st.success("✅ Models and dataset loaded successfully")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Fraud Rate", f"{df['Is Fraudulent'].mean()*100:.2f}%")
    col3.metric("Fraud Cases", f"{df['Is Fraudulent'].sum():,}")

elif page == "Check a Transaction":
    st.header("🔍 Check a Transaction")
    with st.form("transaction_form"):
        amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
        category = st.selectbox("Product Category", df["Product Category"].unique())
        quantity = st.number_input("Quantity", min_value=1, value=1)
        age = st.number_input("Customer Age", min_value=1, value=30)
        location = st.selectbox("Customer Location", df["Customer Location"].unique())
        account_age = st.number_input("Account Age (Days)", min_value=0, value=100)
        hour = st.slider("Transaction Hour", 0, 23, 12)
        shipping = st.text_input("Shipping Address", "123 Main St")
        billing = st.text_input("Billing Address", "123 Main St")
        payment = st.selectbox("Payment Method", df["Payment Method"].unique())
        device = st.selectbox("Device Used", df["Device Used"].unique())
        submitted = st.form_submit_button("Check Transaction")

    if submitted:
        transaction_data = {
            "Transaction Amount": amount,
            "Product Category": category,
            "Quantity": quantity,
            "Customer Age": age,
            "Customer Location": location,
            "Account Age Days": account_age,
            "Transaction Hour": hour,
            "Shipping Address": shipping,
            "Billing Address": billing,
            "Payment Method": payment,
            "Device Used": device,
            "IP Address": "0.0.0.0",
        }
        result = engine.analyze_transaction(transaction_data)

        st.subheader(f"{result['icon']} {result['label']}")
        st.write(result["description"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ML Score", result["ml_score"])
        col2.metric("Category Risk", result["category_risk"])
        col3.metric("Location Risk", result["location_risk"])
        col4.metric("Archetype Risk", result["archetype_risk"])
        st.metric("Final Composite Score", result["final_score"])

elif page == "Analytics":
    st.header("📈 Analytics")
    st.subheader("Fraud Rate by Product Category")
    cat_fraud = df.groupby("Product Category")["Is Fraudulent"].mean().sort_values(ascending=False) * 100
    st.bar_chart(cat_fraud)

    st.subheader("Fraud Rate by Payment Method")
    pay_fraud = df.groupby("Payment Method")["Is Fraudulent"].mean().sort_values(ascending=False) * 100
    st.bar_chart(pay_fraud)

else:
    st.header("ℹ️ About This Project")
    st.write("""
    This dashboard is part of a CCRI summer internship project detecting e-commerce fraud
    using a two-layer scoring system: an XGBoost classifier (Layer 1) and K-Means transaction
    archetype clustering (Layer 2), fused into a single composite fraud risk score.
    """)