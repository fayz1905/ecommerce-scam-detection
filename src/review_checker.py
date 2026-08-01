import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ReviewChecker:
    def __init__(self, model_path=None, vectorizer_path=None):
        if model_path is None:
            model_path = PROJECT_ROOT / 'models' / 'review_model.joblib'
        if vectorizer_path is None:
            vectorizer_path = PROJECT_ROOT / 'models' / 'review_vectorizer.joblib'

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def check_review(self, text):
        text_tfidf = self.vectorizer.transform([text])
        proba = self.model.predict_proba(text_tfidf)[0]
        prediction = self.model.predict(text_tfidf)[0]
        classes = self.model.classes_

        cg_prob = round(float(proba[list(classes).index('CG')]) * 100, 2)
        or_prob = round(float(proba[list(classes).index('OR')]) * 100, 2)

        return {
            'prediction': 'Likely AI-Generated' if prediction == 'CG' else 'Likely Genuine',
            'cg_probability': cg_prob,
            'or_probability': or_prob
        }