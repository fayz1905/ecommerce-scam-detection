import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from product_analyzer import ProductAnalyzer
from seller_analyzer import SellerAnalyzer
from fraud_scorer import FraudScorer

class FraudScoreEngine:
    """
    Combines FraudScorer (ML model), ProductAnalyzer, SellerAnalyzer
    into a single integrated scoring function.
    """
    def __init__(self, model_path=None, scaler_path=None, dataframe=None):
        self.scorer = FraudScorer(model_path, scaler_path) if model_path else FraudScorer()
        self.df = dataframe
        self.product_analyzer = ProductAnalyzer(dataframe) if dataframe is not None else None
        self.seller_analyzer = SellerAnalyzer(dataframe) if dataframe is not None else None

    def fraud_score(self, transaction_data):
        """
        transaction_data: dict with keys:
            Transaction Amount, Quantity, Customer Age, Account Age Days,
            Transaction Hour, Payment Method, Product Category,
            Device Used, Customer Location

        Returns: dict {
            'ml_score': float (0-100),
            'category_risk': float (0-100),
            'location_risk': float (0-100),
            'final_score': float (0-100)
        }
        """
        ml_prob = self.scorer.score_transaction(transaction_data)
        ml_score = ml_prob * 100

        category_risk = 0
        if self.product_analyzer:
            fraud_by_cat = self.product_analyzer.get_fraud_rate_by_category()
            category = transaction_data.get('Product Category')
            if fraud_by_cat is not None and category in fraud_by_cat.index:
                category_risk = fraud_by_cat.loc[category, 'Fraud Rate'] * 100

        location_risk = 0
        if self.seller_analyzer:
            fraud_by_loc = self.seller_analyzer.get_fraud_rate_by_location()
            location = transaction_data.get('Customer Location')
            if fraud_by_loc is not None and location in fraud_by_loc.index:
                location_risk = fraud_by_loc.loc[location, 'Fraud Rate'] * 100

        final_score = (ml_score * 0.70) + (category_risk * 0.15) + (location_risk * 0.15)
        final_score = min(final_score, 100)

        return {
            'ml_score': round(float(ml_score), 2),
            'category_risk': round(float(category_risk), 2),
            'location_risk': round(float(location_risk), 2),
            'final_score': round(float(final_score), 2)
        }

    def classify_risk(self, final_score):
        """
        Classify score into 3 categories:
        - Safe: < 30
        - Suspicious: 30-70
        - Fraud: > 70
        """
        if final_score < 30:
            return {
                'label': 'Safe',
                'color': 'green',
                'icon': '🟢',
                'description': 'Normal transaction, low risk'
            }
        elif final_score <= 70:
            return {
                'label': 'Suspicious',
                'color': 'orange',
                'icon': '🟡',
                'description': 'Requires manual review, medium risk'
            }
        else:
            return {
                'label': 'Fraud',
                'color': 'red',
                'icon': '🔴',
                'description': 'High risk, likely fraudulent'
            }

    def analyze_transaction(self, transaction_data):
        """
        Full pipeline: scoring + classification in one call
        """
        score_result = self.fraud_score(transaction_data)
        risk_result = self.classify_risk(score_result['final_score'])
        return {**score_result, **risk_result}