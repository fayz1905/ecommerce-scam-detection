import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class FraudScorer:
    def __init__(self, model_path=None, scaler_path=None, encoders_path=None):
        if model_path is None:
            model_path = PROJECT_ROOT / 'models' / 'best_model.joblib'
        if scaler_path is None:
            scaler_path = PROJECT_ROOT / 'models' / 'scaler.joblib'
        if encoders_path is None:
            encoders_path = PROJECT_ROOT / 'models' / 'label_encoders.joblib'

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoders = joblib.load(encoders_path)
        self.feature_names = self.scaler.feature_names_in_

    def _encode(self, df):
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str).apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
        return df

    def score_transaction(self, transaction_data):
        df = pd.DataFrame([transaction_data])
        df = self._encode(df)
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]
        df_scaled = self.scaler.transform(df)
        return self.model.predict_proba(df_scaled)[0][1]

    def batch_score(self, dataframe):
        df = dataframe.copy()
        df = self._encode(df)
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]
        df_scaled = self.scaler.transform(df)
        return self.model.predict_proba(df_scaled)[:, 1]