import joblib
import pandas as pd
import numpy as np

class FraudScorer:
    def __init__(self, model_path='../models/final_model.joblib', scaler_path='../models/scaler.joblib'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
    
    def score_transaction(self, transaction_data):
        """
        transaction_data: dict dengan keys sesuai feature names
        return: probability fraud (0-1)
        """
        df = pd.DataFrame([transaction_data])
        df_scaled = self.scaler.transform(df)
        fraud_prob = self.model.predict_proba(df_scaled)[0][1]
        return fraud_prob
    
    def batch_score(self, dataframe):
        """Score multiple transactions"""
        df_scaled = self.scaler.transform(dataframe)
        fraud_probs = self.model.predict_proba(df_scaled)[:, 1]
        return fraud_probs