"""
=============================================================================
FEATURE EXTRACTION PIPELINE FOR PHISHING EMAIL DETECTION
=============================================================================
Implements custom Scikit-Learn Transformer classes to extract:
1. URL and Domain-Level Security Heuristics (IP URLs, Suspicious TLDs, Depth).
2. Lexical & Psychological Threat Indicators (Urgency, Coercion, Financial Lures).
3. TF-IDF N-Gram Text Vectorization.
=============================================================================
"""

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler

# Suspicious Top-Level Domains heavily abused in phishing campaigns
SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".tk", ".ru", ".cc", ".vip", ".click", 
    ".club", ".info", ".biz", ".work", ".link", ".live", ".icu"
}

# Major brand targets frequently spoofed in subdomains
TARGET_BRANDS = [
    "paypal", "microsoft", "office365", "apple", "amazon", 
    "netflix", "wellsfargo", "chase", "google", "dhl", "fedex", "binance", "coinbase"
]

# Urgency / Threat keywords
URGENCY_KEYWORDS = [
    "urgent", "immediately", "immediate", "suspend", "suspended", "suspension",
    "expire", "expires", "expiration", "unauthorized", "compromised", "locked",
    "terminate", "termination", "action required", "final notice", "critical",
    "deadline", "security alert", "security department", "account locked"
]

# Financial / Transactional lure keywords
FINANCIAL_KEYWORDS = [
    "invoice", "wire transfer", "wire", "transfer", "bank", "credit card",
    "bitcoin", "crypto", "wallet", "refund", "charged", "surcharge",
    "payout", "reward", "grant", "balance", "beneficiary", "claim"
]

# Credential harvesting keywords
CREDENTIAL_KEYWORDS = [
    "password", "passwords", "credentials", "login", "signin", "sign-in",
    "verify", "verification", "identity", "confirm", "portal", "re-activate"
]

URL_REGEX = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+', re.IGNORECASE)
IP_REGEX = re.compile(r'https?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/[^\s]*)?', re.IGNORECASE)


class EmailStructuralFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts numerical features based on URLs, keywords, and structural formatting.
    Compatible with Scikit-learn Pipeline and FeatureUnion.
    """

    def __init__(self):
        self.feature_names_ = [
            "url_count",
            "has_ip_url",
            "suspicious_tld_count",
            "brand_spoofing_count",
            "at_symbol_in_url",
            "max_url_length",
            "has_http_insecure",
            "urgency_count",
            "financial_count",
            "credential_count",
            "caps_ratio",
            "exclamation_count",
            "question_count",
            "dollar_count"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transforms a collection of email text strings into a 2D NumPy array of numerical features.
        """
        features_list = []
        for text in X:
            features_list.append(self._extract_single(str(text)))
        return np.array(features_list, dtype=np.float64)

    def _extract_single(self, text):
        lower_text = text.lower()
        urls = URL_REGEX.findall(text)
        
        # 1. URL Features
        url_count = len(urls)
        has_ip_url = 1.0 if any(IP_REGEX.search(u) for u in urls) else 0.0
        
        suspicious_tld_count = 0.0
        brand_spoofing_count = 0.0
        at_symbol_in_url = 0.0
        max_url_length = 0.0
        has_http_insecure = 0.0

        for url in urls:
            u_lower = url.lower()
            if len(url) > max_url_length:
                max_url_length = len(url)
            if u_lower.startswith("http://"):
                has_http_insecure = 1.0
            if "@" in url:
                at_symbol_in_url = 1.0
            if any(u_lower.endswith(tld) or (tld + "/") in u_lower for tld in SUSPICIOUS_TLDS):
                suspicious_tld_count += 1.0
            if any(brand in u_lower for brand in TARGET_BRANDS):
                brand_spoofing_count += 1.0

        # 2. Textual & Psychological Urgency Features
        urgency_count = sum(lower_text.count(kw) for kw in URGENCY_KEYWORDS)
        financial_count = sum(lower_text.count(kw) for kw in FINANCIAL_KEYWORDS)
        credential_count = sum(lower_text.count(kw) for kw in CREDENTIAL_KEYWORDS)

        # 3. Structural Punctuation & Capitalization
        total_chars = max(1, len(text))
        caps_count = sum(1 for c in text if c.isupper())
        caps_ratio = caps_count / total_chars
        exclamation_count = text.count("!")
        question_count = text.count("?")
        dollar_count = text.count("$")

        return [
            url_count,
            has_ip_url,
            suspicious_tld_count,
            brand_spoofing_count,
            at_symbol_in_url,
            max_url_length,
            has_http_insecure,
            urgency_count,
            financial_count,
            credential_count,
            caps_ratio,
            exclamation_count,
            question_count,
            dollar_count
        ]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


def build_feature_pipeline():
    """
    Constructs a complete Scikit-Learn FeatureUnion pipeline combining:
    1. TF-IDF N-gram text representation (1-gram and 2-gram).
    2. Scaled numerical URL and structural security features.
    """
    tfidf_branch = Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=3000,
            sublinear_tf=True
        ))
    ])

    structural_branch = Pipeline([
        ('structural_extractor', EmailStructuralFeatureExtractor()),
        ('scaler', MinMaxScaler())
    ])

    combined_features = FeatureUnion([
        ('tfidf_features', tfidf_branch),
        ('structural_features', structural_branch)
    ])

    return combined_features
