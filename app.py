import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from fraud_scorer import FraudScorer
from product_analyzer import ProductAnalyzer
from seller_analyzer import SellerAnalyzer

import os
print(f"Current working directory: {os.getcwd()}")
print(f"Models folder contents: {os.listdir('models')}")

# Load model & data
@st.cache_resource
def load_model():
    return FraudScorer()

@st.cache_data
def load_dataset():
    return pd.read_csv('data/Fraudulent_E-Commerce_Transaction_Data.csv')

# Config
st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("🚨 E-Commerce Fraud Detection Dashboard")

# Load
model = load_model()
df = load_dataset()

# Sidebar navigation
page = st.sidebar.selectbox("Select Page", [
    "Dashboard Overview",
    "Single Transaction Analysis",
    "Batch Analysis",
    "Risk Analytics"
])

if page == "Dashboard Overview":
    st.header("📊 Dashboard Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(df))
    col2.metric("Fraudulent Cases", (df['Is Fraudulent'] == 1).sum())
    col3.metric("Fraud Rate", f"{(df['Is Fraudulent'] == 1).sum() / len(df) * 100:.2f}%")
    
    st.subheader("Fraud Distribution")
    fig = px.pie(values=df['Is Fraudulent'].value_counts(), names=['Non-Fraud', 'Fraud'], color_discrete_sequence=['green', 'red'])
    st.plotly_chart(fig, use_container_width=True)

elif page == "Single Transaction Analysis":
    st.header("🔍 Single Transaction Analysis")
    
    st.write("Input transaction details to check fraud probability")
    
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount", min_value=0.0)
        quantity = st.number_input("Quantity", min_value=1, step=1)
        age = st.number_input("Customer Age", min_value=18, max_value=100)
        device = st.selectbox("Device Used", df['Device Used'].unique())
    
    with col2:
        payment = st.selectbox("Payment Method", df['Payment Method'].unique())
        category = st.selectbox("Product Category", df['Product Category'].unique())
        location = st.selectbox("Customer Location", df['Customer Location'].unique())
        hour = st.number_input("Transaction Hour", min_value=0, max_value=23)
    
    if st.button("Analyze Transaction"):
        # Create input
        input_data = {col: 0 for col in df.select_dtypes(include=[np.number]).columns if col != 'Is Fraudulent'}
        input_data['Transaction Amount'] = amount
        input_data['Quantity'] = quantity
        input_data['Customer Age'] = age
        input_data['Transaction Hour'] = hour
        
        # Score
        fraud_prob = model.score_transaction(input_data)
        risk_level = "🔴 HIGH RISK" if fraud_prob > 0.7 else "🟡 MEDIUM RISK" if fraud_prob > 0.4 else "🟢 LOW RISK"
        
        st.success(f"Fraud Probability: {fraud_prob:.2%} - {risk_level}")

elif page == "Batch Analysis":
    st.header("📁 Batch Transaction Analysis")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        fraud_probs = model.batch_score(batch_df.select_dtypes(include=[np.number]))
        batch_df['Fraud_Probability'] = fraud_probs
        batch_df['Risk_Level'] = batch_df['Fraud_Probability'].apply(
            lambda x: 'HIGH' if x > 0.7 else 'MEDIUM' if x > 0.4 else 'LOW'
        )
        
        st.subheader("Results")
        st.dataframe(batch_df[['Fraud_Probability', 'Risk_Level']])
        st.download_button("Download Results", batch_df.to_csv(index=False), "results.csv")

elif page == "Risk Analytics":
    st.header("📈 Risk Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fraud Rate by Product Category")
        product_analyzer = ProductAnalyzer(df)
        fraud_by_category = product_analyzer.get_fraud_rate_by_category()
        if fraud_by_category is not None:
            fig = px.bar(fraud_by_category, y='Fraud Rate', color='Fraud Rate', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Fraud Rate by Location")
        seller_analyzer = SellerAnalyzer(df)
        fraud_by_location = seller_analyzer.get_fraud_rate_by_location()
        if fraud_by_location is not None:
            fig = px.bar(fraud_by_location, y='Fraud Rate', color='Fraud Rate', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("Fraud Detection Dashboard v1.0 | CCRI Internship 2026")