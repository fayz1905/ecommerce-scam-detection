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
from phishing_checker import PhishingChecker

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("🚨 E-Commerce Fraud Detection Dashboard")


@st.cache_resource
def load_engine():
    df = pd.read_csv(PROJECT_ROOT / "data" / "Fraudulent_E-Commerce_Transaction_Data.csv")
    engine = FraudScoreEngine(
        model_path=str(PROJECT_ROOT / "models" / "best_model.joblib"),
        scaler_path=str(PROJECT_ROOT / "models" / "scaler.joblib"),
        dataframe=df
    )
    return engine, df


@st.cache_resource
def load_phishing_checker():
    return PhishingChecker(model_path=str(PROJECT_ROOT / "models" / "phishing_model.joblib"))


@st.cache_data
def compute_archetype_data(_engine, df):
    import joblib
    archetype_scaler = joblib.load(PROJECT_ROOT / "models" / "archetype_scaler.pkl")
    archetype_model = joblib.load(PROJECT_ROOT / "models" / "archetype_kmeans.pkl")

    feature_cols = ['price_deviation', 'addr_mismatch', 'is_odd_hour',
                     'is_new_account', 'qty_deviation', 'amount_per_age']

    sample = df.sample(n=min(20000, len(df)), random_state=42).reset_index(drop=True)

    category_avg = sample.groupby('Product Category')['Transaction Amount'].transform('mean')
    category_std = sample.groupby('Product Category')['Transaction Amount'].transform('std')
    sample['price_deviation'] = ((sample['Transaction Amount'] - category_avg) / category_std).replace([float('inf'), -float('inf')], 0).fillna(0)

    sample['addr_mismatch'] = (sample['Shipping Address'] != sample['Billing Address']).astype(int)
    sample['is_odd_hour'] = sample['Transaction Hour'].apply(lambda h: 1 if h < 6 or h > 22 else 0)
    sample['is_new_account'] = (sample['Account Age Days'] < 30).astype(int)

    qty_avg = sample.groupby('Product Category')['Quantity'].transform('mean')
    qty_std = sample.groupby('Product Category')['Quantity'].transform('std')
    sample['qty_deviation'] = ((sample['Quantity'] - qty_avg) / qty_std).replace([float('inf'), -float('inf')], 0).fillna(0)

    sample['amount_per_age'] = sample['Transaction Amount'] / (sample['Account Age Days'] + 1)

    X_scaled = archetype_scaler.transform(sample[feature_cols])
    sample['cluster'] = archetype_model.predict(X_scaled)

    return sample, X_scaled, feature_cols


with st.spinner("Loading models and data..."):
    engine, df = load_engine()
    phishing_checker = load_phishing_checker()


page = st.sidebar.selectbox("Pages", ["Dashboard Overview", "Check a Transaction", "Analytics", "Archetype Clusters", "Batch Alerts", "Website Risk Checker", "Info"])

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
    import matplotlib.pyplot as plt

    st.subheader("Fraud Rate by Product Category")
    cat_fraud = df.groupby("Product Category")["Is Fraudulent"].mean().sort_values(ascending=False) * 100

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    cat_fraud.plot(kind='bar', ax=ax1, color='indianred')
    ax1.set_ylabel("Fraud Rate (%)")
    ax1.set_ylim(cat_fraud.min() * 0.8, cat_fraud.max() * 1.2)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig1)

    st.subheader("Fraud Rate by Payment Method")
    pay_fraud = df.groupby("Payment Method")["Is Fraudulent"].mean().sort_values(ascending=False) * 100

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    pay_fraud.plot(kind='bar', ax=ax2, color='steelblue')
    ax2.set_ylabel("Fraud Rate (%)")
    ax2.set_ylim(pay_fraud.min() * 0.8, pay_fraud.max() * 1.2)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig2)

elif page == "Archetype Clusters":
    st.header("🧩 Transaction Archetype Clusters")
    st.write("Unsupervised clustering groups transactions by behavioral pattern, complementing the ML model with a novel risk signal.")

    with st.spinner("Computing archetype clusters..."):
        archetype_df, X_scaled, feature_cols = compute_archetype_data(engine, df)

    st.subheader("Cluster Summary")
    summary = archetype_df.groupby('cluster').agg(
        transaction_count=('Is Fraudulent', 'count'),
        fraud_rate=('Is Fraudulent', 'mean')
    ).reset_index()
    summary['fraud_rate'] = (summary['fraud_rate'] * 100).round(2)
    summary = summary.sort_values('fraud_rate', ascending=False)
    summary.columns = ['Cluster ID', 'Transaction Count', 'Fraud Rate (%)']
    st.dataframe(summary, use_container_width=True)

    st.subheader("Dominant Feature Signature per Cluster")
    feature_summary = archetype_df.groupby('cluster')[feature_cols].mean().round(2)
    st.dataframe(feature_summary, use_container_width=True)

    st.subheader("2D Projection of Transaction Archetypes")
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plot_df = pd.DataFrame({
        'x': X_pca[:, 0],
        'y': X_pca[:, 1],
        'cluster': archetype_df['cluster'].astype(str),
        'is_fraud': archetype_df['Is Fraudulent']
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=plot_df, x='x', y='y', hue='cluster', style='is_fraud', palette='tab10', alpha=0.6, s=40, ax=ax)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('Transaction Archetype Clusters (PCA Projection)')
    st.pyplot(fig)

elif page == "Batch Alerts":
    st.header("🚨 Batch Transaction Alerts")
    st.write("Upload a CSV of transactions to screen them all at once and flag high-risk cases.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty or has no readable data. Please upload a valid CSV with at least one transaction.")
            st.stop()

        if len(batch_df) == 0:
            st.error("The uploaded file is empty. Please upload a CSV with at least one transaction.")
            st.stop()

        if len(batch_df) > 1000:
            st.warning("Large file detected. Processing may take a while — consider uploading a smaller sample for quick testing.")

        required_cols = ["Transaction Amount", "Product Category", "Quantity", "Customer Age",
                          "Customer Location", "Account Age Days", "Transaction Hour",
                          "Shipping Address", "Billing Address", "Payment Method", "Device Used"]
        missing_cols = [c for c in required_cols if c not in batch_df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            blank_rows = batch_df[required_cols].isnull().any(axis=1).sum()
            if blank_rows > 0:
                st.warning(f"{blank_rows} row(s) have missing values in required fields. These rows will still be scored, but results may be less accurate.")

            with st.spinner(f"Scoring {len(batch_df)} transactions..."):
                results = []
                for idx, row in batch_df.iterrows():
                    transaction_data = row.to_dict()
                    if "IP Address" not in transaction_data:
                        transaction_data["IP Address"] = "0.0.0.0"
                    result = engine.analyze_transaction(transaction_data)
                    results.append(result)

                results_df = pd.DataFrame(results)
                combined_df = pd.concat([batch_df.reset_index(drop=True), results_df], axis=1)

            st.success(f"Scored {len(combined_df)} transactions")

            col1, col2, col3 = st.columns(3)
            col1.metric("Safe", (combined_df['label'] == 'Safe').sum())
            col2.metric("Suspicious", (combined_df['label'] == 'Suspicious').sum())
            col3.metric("Fraud", (combined_df['label'] == 'Fraud').sum())

            st.subheader("Flagged Transactions (Suspicious + Fraud)")
            flagged = combined_df[combined_df['label'] != 'Safe'].sort_values('final_score', ascending=False)
            st.dataframe(
                flagged[['Transaction Amount', 'Product Category', 'Customer Location',
                         'final_score', 'label']],
                use_container_width=True
            )

            st.subheader("All Results")
            st.dataframe(
                combined_df[['Transaction Amount', 'Product Category', 'Customer Location',
                             'ml_score', 'category_risk', 'location_risk', 'archetype_risk',
                             'final_score', 'label']],
                use_container_width=True
            )

            csv_export = combined_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Full Results as CSV",
                data=csv_export,
                file_name="scored_transactions.csv",
                mime="text/csv"
            )

elif page == "Website Risk Checker":
    st.header("🔗 Website Risk Checker")
    st.write("Paste a URL below to check whether it shows characteristics of a phishing website.")

    url_input = st.text_input("Website URL", placeholder="https://example.com")
    check_button = st.button("Check Website")

    st.caption("Note: This checker is trained primarily on Western/English-language URL patterns and may be less reliable for regional domains (e.g., .co.id, .co.uk). Always use additional judgment for suspicious sites.")

    if check_button and url_input:
        with st.spinner("Analyzing URL..."):
            result = phishing_checker.check_url(url_input)

        if result['prediction'] == 'Legitimate':
            st.success(f"✅ Legitimate — {result['legitimate_probability']}% confidence")
        else:
            st.error(f"⚠️ Phishing — {result['phishing_probability']}% confidence")

        col1, col2 = st.columns(2)
        col1.metric("Legitimate Probability", f"{result['legitimate_probability']}%")
        col2.metric("Phishing Probability", f"{result['phishing_probability']}%")
    elif check_button and not url_input:
        st.warning("Please enter a URL first.")

else:
    st.header("ℹ️ About This Project")
    st.write("""
    This dashboard is part of a CCRI summer internship project detecting e-commerce fraud
    and online scams. It combines two independent modules: a transaction fraud detection
    system (XGBoost classifier and K-Means archetype clustering, fused into a composite
    risk score), and a phishing website checker (Random Forest classifier trained on
    URL structural features).
    """)