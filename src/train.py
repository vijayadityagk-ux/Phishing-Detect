"""
=============================================================================
MODEL TRAINING & BENCHMARKING PIPELINE
=============================================================================
Trains and compares multiple Scikit-Learn classifiers:
1. Random Forest Classifier
2. Logistic Regression
3. Support Vector Classifier (Calibrated LinearSVC)
4. Complement Naive Bayes

Selects the best model, serializes it, and saves evaluation checkpoints.
=============================================================================
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from feature_extractor import build_feature_pipeline

def train_and_evaluate_models(dataset_path, model_dir):
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 70)
    print("PHISHING EMAIL DETECTION - MODEL TRAINING & BENCHMARKING")
    print("=" * 70)

    # 1. Load Dataset
    print(f"[*] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    X = df["full_text"].values
    y = df["is_phishing"].values

    print(f"[*] Total dataset size: {len(X)} samples")
    print(f"[*] Safe emails (0): {sum(y == 0)} | Phishing emails (1): {sum(y == 1)}")

    # 2. Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[*] Training set: {len(X_train)} samples | Testing set: {len(X_test)} samples\n")

    # 3. Candidate Classifiers
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=20, random_state=42),
        "Logistic Regression": LogisticRegression(C=2.0, max_iter=1000, random_state=42),
        "Linear SVM (Calibrated)": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, dual='auto')),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1)
    }

    results = []
    best_model_name = None
    best_f1 = 0.0
    best_pipeline = None

    print(f"{'Model Name':<28} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 88)

    for name, clf in classifiers.items():
        # Build unified feature + classifier pipeline
        pipeline = Pipeline([
            ('features', build_feature_pipeline()),
            ('classifier', clf)
        ])

        # Train model
        pipeline.fit(X_train, y_train)

        # Predict on held-out test set
        y_pred = pipeline.predict(X_test)
        
        # Probabilities for ROC-AUC
        if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        print(f"{name:<28} | {acc * 100:>8.2f}% | {prec * 100:>8.2f}% | {rec * 100:>8.2f}% | {f1 * 100:>8.2f}% | {roc:>8.4f}")

        results.append({
            "model_name": name,
            "pipeline": pipeline,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_pipeline = pipeline

    print("-" * 88)
    print(f"\n[+] Champion Model Selected: {best_model_name} (F1-Score: {best_f1 * 100:.2f}%)")

    # 4. Serialize Champion Pipeline and Test Data
    model_save_path = os.path.join(model_dir, "phishing_detector.pkl")
    test_data_path = os.path.join(model_dir, "test_data.pkl")

    joblib.dump(best_pipeline, model_save_path)
    joblib.dump((X_test, y_test, best_model_name), test_data_path)

    print(f"[+] Serialized model pipeline saved to: {model_save_path}")
    print(f"[+] Held-out test evaluation data saved to: {test_data_path}")
    print("\n[SUCCESS] Training phase completed successfully!")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    dataset_file = os.path.join(project_root, "dataset", "emails.csv")
    model_folder = os.path.join(project_root, "model")
    
    train_and_evaluate_models(dataset_file, model_folder)
