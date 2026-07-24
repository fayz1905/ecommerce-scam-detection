from urllib.parse import urlparse
import re
import pandas as pd
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_COLS = [
    'URLLength', 'DomainLength', 'IsDomainIP', 'TLDLength', 'NoOfSubDomain',
    'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
    'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL',
    'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS',
    'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio', 'CharContinuationRate'
]


def extract_url_features(url):
    COMPOUND_TLDS = ['co.id', 'co.uk', 'com.au', 'co.jp', 'com.br', 'co.in', 'co.nz']

    parsed = urlparse(url)
    domain = parsed.netloc

    url_length = len(url)
    domain_length = len(domain)

    is_domain_ip = 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', domain) else 0

    tld = domain.split('.')[-1] if '.' in domain else ''
    for compound in COMPOUND_TLDS:
        if domain.endswith('.' + compound):
            tld = compound
            break
    tld_length = len(tld)

    domain_for_counting = domain
    compound_stripped = False
    for compound in COMPOUND_TLDS:
        if domain_for_counting.endswith('.' + compound):
            domain_for_counting = domain_for_counting[:-(len(compound) + 1)]
            compound_stripped = True
            break

    subdomain_parts = domain_for_counting.split('.')
    if compound_stripped:
        no_of_subdomain = max(len(subdomain_parts) - 1, 0)
    else:
        no_of_subdomain = max(len(subdomain_parts) - 2, 0)

    no_of_letters = sum(c.isalpha() for c in url)
    letter_ratio = no_of_letters / url_length if url_length > 0 else 0

    no_of_digits = sum(c.isdigit() for c in url)
    digit_ratio = no_of_digits / url_length if url_length > 0 else 0

    no_of_equals = url.count('=')
    no_of_qmark = url.count('?')
    no_of_ampersand = url.count('&')

    special_chars = sum(not c.isalnum() for c in url)
    special_char_ratio = special_chars / url_length if url_length > 0 else 0

    other_special_chars = special_chars - no_of_equals - no_of_qmark - no_of_ampersand

    is_https = 1 if parsed.scheme == 'https' else 0

    obfuscation_chars = url.count('%')
    has_obfuscation = 1 if obfuscation_chars > 0 else 0
    obfuscation_ratio = obfuscation_chars / url_length if url_length > 0 else 0

    max_run = 1
    current_run = 1
    for i in range(1, len(url)):
        if url[i] == url[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    char_continuation_rate = max_run / url_length if url_length > 0 else 0

    return {
        'URLLength': url_length,
        'DomainLength': domain_length,
        'IsDomainIP': is_domain_ip,
        'TLDLength': tld_length,
        'NoOfSubDomain': no_of_subdomain,
        'NoOfLettersInURL': no_of_letters,
        'LetterRatioInURL': letter_ratio,
        'NoOfDegitsInURL': no_of_digits,
        'DegitRatioInURL': digit_ratio,
        'NoOfEqualsInURL': no_of_equals,
        'NoOfQMarkInURL': no_of_qmark,
        'NoOfAmpersandInURL': no_of_ampersand,
        'NoOfOtherSpecialCharsInURL': other_special_chars,
        'SpacialCharRatioInURL': special_char_ratio,
        'IsHTTPS': is_https,
        'HasObfuscation': has_obfuscation,
        'NoOfObfuscatedChar': obfuscation_chars,
        'ObfuscationRatio': obfuscation_ratio,
        'CharContinuationRate': char_continuation_rate
    }


class PhishingChecker:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = PROJECT_ROOT / 'models' / 'phishing_model.joblib'
        self.model = joblib.load(model_path)

    def check_url(self, url):
        features = extract_url_features(url)
        features_df = pd.DataFrame([features])[FEATURE_COLS]
        proba = self.model.predict_proba(features_df)[0]
        prediction = self.model.predict(features_df)[0]
        return {
            'prediction': 'Legitimate' if prediction == 1 else 'Phishing',
            'legitimate_probability': round(float(proba[1]) * 100, 2),
            'phishing_probability': round(float(proba[0]) * 100, 2)
        }