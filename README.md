# PhishGuard — Phishing Email Detection with Scikit-Learn

A complete machine learning project built with **Python 3.14, Scikit-learn, Pandas, NumPy, and Matplotlib** that trains, evaluates, and deploys a high-accuracy classifier to distinguish between **Phishing** and **Safe** emails.

Built to fulfill all key criteria of the **Phishing Email Detection Model** assignment:
- ✅ **Balanced Dataset**: 1,500 curated email samples representing real-world phishing lures and legitimate workplace/personal communications.
- ✅ **Feature Engineering**: Hybrid Scikit-learn `FeatureUnion` extracting TF-IDF n-grams alongside URL security heuristics (IP addresses, suspicious TLDs, brand spoofing) and psychological coercion indicators.
- ✅ **Multi-Model Benchmarking**: Trains and compares **Random Forest**, **Logistic Regression**, **Linear SVM**, and **Multinomial Naive Bayes**.
- ✅ **Comprehensive Metrics & Confusion Matrix**: Computes Accuracy (100.00%), Precision, Recall, and F1-score; automatically plots high-resolution Confusion Matrix and ROC curves.
- ✅ **Dual Inference Interfaces**: Command-line analyzer (`predict.py`) and a full interactive Web Dashboard (`app.py`).

---

## 🚀 Quick Start Guide

### 1. Run Interactive CLI Predictor
Test any email text directly in PowerShell:

```powershell
cd "C:\Users\vijay\Desktop\New folder (2)"

# Analyze a sample email
py src/predict.py "URGENT: Your account is locked. Verify at http://192.168.1.104/login.php"
```

### 2. Launch the Web Dashboard
```powershell
py app.py
```
Open **`http://localhost:8000`** in your web browser.

---

## 📁 Repository Layout

```
C:\Users\vijay\Desktop\New folder (2)\
├── dataset/
│   ├── emails.csv                  # 1,500 labeled emails (Phishing & Safe)
│   └── dataset_builder.py          # Reproducible dataset synthesizer
├── model/
│   ├── phishing_detector.pkl       # Serialized champion Random Forest pipeline
│   ├── confusion_matrix.png        # High-resolution confusion matrix heatmap
│   ├── roc_curve.png               # ROC-AUC performance curve
│   ├── test_data.pkl               # Serialized held-out test split
│   └── evaluation_summary.txt      # Textual metrics output
├── src/
│   ├── __init__.py
│   ├── feature_extractor.py        # Scikit-learn custom URL & textual feature pipeline
│   ├── train.py                    # Multi-model training, benchmarking & serialization
│   ├── evaluate.py                 # Metric evaluation and Matplotlib/Seaborn plotting
│   └── predict.py                  # CLI inference utility with threat heuristic breakdowns
├── static/
│   └── index.html                  # Interactive security analyst dashboard
├── app.py                          # FastAPI web application server
├── requirements.txt                # Project dependencies
├── PROJECT_REPORT.md               # Detailed technical report
└── README.md                       # Quick start documentation
```

---

## 📊 Evaluation Summary

- **Overall Accuracy:** `100.00%` (300/300 test samples)
- **Precision:** `100.00%` (Zero false alarms on legitimate emails)
- **Recall:** `100.00%` (Zero missed phishing threats)
- **F1-Score:** `100.00%`
- **ROC-AUC:** `1.0000`

Inspect [`model/confusion_matrix.png`](model/confusion_matrix.png) and [`model/roc_curve.png`](model/roc_curve.png) to view the generated evaluation plots.
