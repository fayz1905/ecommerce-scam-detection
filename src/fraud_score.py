import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from product_analyzer import ProductAnalyzer
from seller_analyzer import SellerAnalyzer
from fraud_scorer import FraudScorer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FraudScoreEngine:
    """
    Combines FraudScorer (ML model), ProductAnalyzer, SellerAnalyzer,
    and Transaction Archetype Clustering into a single integrated scoring function.
    """

    def __init__(self, model_path=None, scaler_path=None, dataframe=None):
        self.scorer = FraudScorer(model_path, scaler_path) if model_path else FraudScorer()
        self.df = dataframe
        self.product_analyzer = ProductAnalyzer(dataframe) if dataframe is not None else None
        self.seller_analyzer = SellerAnalyzer(dataframe) if dataframe is not None else None

        self.archetype_model = None
        self.archetype_scaler = None
        self.cluster_risk_map = {}
        self.cat_price_median = {}
        self.cat_qty_median = {}

        archetype_model_path = PROJECT_ROOT / 'models' / 'archetype_kmeans.pkl'
        archetype_scaler_path = PROJECT_ROOT / 'models' / 'archetype_scaler.pkl'
        cluster_risk_map_path = PROJECT_ROOT / 'src' / 'cluster_risk_map.json'

        if archetype_model_path.exists() and archetype_scaler_path.exists():
            self.archetype_model = joblib.load(archetype_model_path)
            self.archetype_scaler = joblib.load(archetype_scaler_path)

        if cluster_risk_map_path.exists():
            with open(cluster_risk_map_path) as f:
                self.cluster_risk_map = json.load(f)

        if dataframe is not None:
            self.cat_price_median = dataframe.groupby('Product Category')['Transaction Amount'].median().to_dict()
            self.cat_qty_median = dataframe.groupby('Product Category')['Quantity'].median().to_dict()

            # Precompute category/location risk tables once, instead of recalculating on every call
            self.category_risk_table = self.product_analyzer.get_fraud_rate_by_category() if self.product_analyzer else None
            self.location_risk_table = self.seller_analyzer.get_fraud_rate_by_location() if self.seller_analyzer else None
        else:
            self.category_risk_table = None
            self.location_risk_table = None

    def _build_single_archetype_features(self, transaction_data):
        category = transaction_data.get('Product Category')
        amount = transaction_data.get('Transaction Amount', 0)
        quantity = transaction_data.get('Quantity', 1)
        hour = transaction_data.get('Transaction Hour', 12)
        account_age = transaction_data.get('Account Age Days', 999)
        age = transaction_data.get('Customer Age', 1) or 1
        shipping = transaction_data.get('Shipping Address', '')
        billing = transaction_data.get('Billing Address', '')

        price_med = self.cat_price_median.get(category, amount if amount else 1)
        qty_med = self.cat_qty_median.get(category, quantity if quantity else 1)

        price_deviation = (amount - price_med) / price_med if price_med else 0
        addr_mismatch = 1 if shipping != billing else 0
        is_odd_hour = 1 if (hour < 6 or hour > 22) else 0
        is_new_account = 1 if account_age < 30 else 0
        qty_deviation = (quantity - qty_med) / qty_med if qty_med else 0
        amount_per_age = amount / age if age else 0

        return pd.DataFrame([{
            'price_deviation': price_deviation,
            'addr_mismatch': addr_mismatch,
            'is_odd_hour': is_odd_hour,
            'is_new_account': is_new_account,
            'qty_deviation': qty_deviation,
            'amount_per_age': amount_per_age
        }])

    def get_archetype_risk(self, transaction_data):
        if self.archetype_model is None or self.archetype_scaler is None:
            return 0.0

        feats = self._build_single_archetype_features(transaction_data)
        X_scaled = self.archetype_scaler.transform(feats)
        cluster = self.archetype_model.predict(X_scaled)[0]
        fraud_rate = self.cluster_risk_map.get(str(cluster), 0.0)
        return round(float(fraud_rate) * 100, 2)

    def fraud_score(self, transaction_data):
        ml_prob = self.scorer.score_transaction(transaction_data)
        ml_score = ml_prob * 100

        category_risk = 0
        if self.category_risk_table is not None:
            category = transaction_data.get('Product Category')
            if category in self.category_risk_table.index:
                category_risk = self.category_risk_table.loc[category, 'Fraud Rate'] * 100

        location_risk = 0
        if self.location_risk_table is not None:
            location = transaction_data.get('Customer Location')
            if location in self.location_risk_table.index:
                location_risk = self.location_risk_table.loc[location, 'Fraud Rate'] * 100

        archetype_risk = self.get_archetype_risk(transaction_data)

        final_score = (
            (ml_score * 0.60) +
            (category_risk * 0.15) +
            (location_risk * 0.10) +
            (archetype_risk * 0.15)
        )
        final_score = min(final_score, 100)

        return {
            'ml_score': round(float(ml_score), 2),
            'category_risk': round(float(category_risk), 2),
            'location_risk': round(float(location_risk), 2),
            'archetype_risk': round(float(archetype_risk), 2),
            'final_score': round(float(final_score), 2)
        }

    def classify_risk(self, final_score):
        if final_score < 30:
            return {'label': 'Safe', 'color': 'green', 'icon': '🟢', 'description': 'Normal transaction, low risk'}
        elif final_score <= 50:
            return {'label': 'Suspicious', 'color': 'orange', 'icon': '🟡', 'description': 'Requires manual review, medium risk'}
        else:
            return {'label': 'Fraud', 'color': 'red', 'icon': '🔴', 'description': 'High risk, likely fraudulent'}

    def analyze_transaction(self, transaction_data):
        score_result = self.fraud_score(transaction_data)
        risk_result = self.classify_risk(score_result['final_score'])
        return {**score_result, **risk_result}