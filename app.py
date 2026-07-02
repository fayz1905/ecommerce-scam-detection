import streamlit as st

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("🚨 E-Commerce Fraud Detection Dashboard")

st.warning("Demo Mode: Models stored externally. Full app requires cloud storage setup.")

page = st.sidebar.selectbox("Pages", ["Dashboard Overview", "Info"])

if page == "Dashboard Overview":
    st.header("📊 Dashboard Overview")
    st.info("✅ Project complete - notebooks, src modules, app structure ready")
    st.metric("Status", "Requires external model storage for full deployment")