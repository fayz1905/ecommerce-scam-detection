import pandas as pd

class ProductAnalyzer:
    def __init__(self, dataframe):
        self.df = dataframe
    
    def get_fraud_rate_by_category(self):
        """Analyze fraud rate by product category"""
        if 'Product Category' not in self.df.columns or 'Is Fraudulent' not in self.df.columns:
            return None
        
        fraud_by_category = self.df.groupby('Product Category')['Is Fraudulent'].agg(['sum', 'count', 'mean'])
        fraud_by_category.columns = ['Fraud Count', 'Total', 'Fraud Rate']
        return fraud_by_category.sort_values('Fraud Rate', ascending=False)
    
    def get_high_risk_categories(self, threshold=0.1):
        """Get categories with fraud rate above threshold"""
        fraud_rate = self.get_fraud_rate_by_category()
        if fraud_rate is not None:
            return fraud_rate[fraud_rate['Fraud Rate'] > threshold]
        return None