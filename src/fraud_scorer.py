import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path

class FraudScorer:
    def __init__(self, model_path=None, scaler_path=None):
        if model_path is None:
            base_dir = Path(__file__).parent.parent
            model_path = base_dir / 'models' / 'best_model.joblib'
        if scaler_path is None:
            base_dir = Path(__file__).parent.parent
            scaler_path = base_dir / 'models' / 'scaler.joblib'
        
        self.model = joblib.load(str(model_path))
        self.scaler = joblib.load(str(scaler_path))