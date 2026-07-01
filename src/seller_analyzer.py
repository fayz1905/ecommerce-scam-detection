import pandas as pd

class SellerAnalyzer:
    def __init__(self, dataframe):
        self.df = dataframe
    
    def get_fraud_rate_by_location(self):
        """Analyze fraud rate by customer location"""
        if 'Customer Location' not in self.df.columns or 'Is Fraudulent' not in self.df.columns:
            return None
        
        fraud_by_location = self.df.groupby('Customer Location')['Is Fraudulent'].agg(['sum', 'count', 'mean'])
        fraud_by_location.columns = ['Fraud Count', 'Total', 'Fraud Rate']
        return fraud_by_location.sort_values('Fraud Rate', ascending=False)
    
    def get_suspicious_locations(self, threshold=0.1):
        """Get locations with fraud rate above threshold"""
        fraud_rate = self.get_fraud_rate_by_location()
        if fraud_rate is not None:
            return fraud_rate[fraud_rate['Fraud Rate'] > threshold]
        return None