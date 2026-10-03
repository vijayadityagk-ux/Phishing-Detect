"""
=============================================================================
MODEL EVALUATION & VISUALIZATION PIPELINE
=============================================================================
Evaluates the trained champion model on held-out test data.
Generates:
1. Confusion Matrix Heatmap (model/confusion_matrix.png)
2. Receiver Operating Characteristic (ROC) Curve (model/roc_curve.png)
3. Precision-Recall Curve (model/precision_recall_curve.png)
4. Comprehensive Metrics Summary Table
=============================================================================
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for generating image files
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def evaluate_model(model_dir):
    model_path = os.path.join(model_dir, "phishing_detector.pkl")
    test_path = os.path.join(model_dir, "test_data.pkl")

    if not os.path.exists(model_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Trained model or test data not found. Please run train.py first.")

    print("=" * 70)
    print("PHISHING EMAIL DETECTION - COMPREHENSIVE MODEL EVALUATION")
    print("=" * 70)

    # 1. Load Model and Test Split
    pipeline = joblib.load(model_path)
    X_test, y_test, model_name = joblib.load(test_path)

    print(f"[*] Evaluating Champion Model: {model_name}")
    print(f"[*] Total Test Samples: {len(y_test)}")

    # 2. Compute Predictions & Probabilities
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 50)
    print("           OVERALL PERFORMANCE METRICS")
    print("=" * 50)
    print(f" Accuracy : {acc * 100:.2f}%")
    print(f" Precision: {prec * 100:.2f}%")
    print(f" Recall   : {rec * 100:.2f}%")
    print(f" F1-Score : {f1 * 100:.2f}%")
    print("=" * 50)

    # 3. Detailed Classification Report
    print("\nDETAILED CLASSIFICATION REPORT:")
    print("-" * 55)
    report = classification_report(
        y_test, 
        y_pred, 
        target_names=["Safe (0)", "Phishing (1)"], 
        digits=4
    )
    print(report)

    # 4. Generate & Save Confusion Matrix Plot
    print("[*] Generating Confusion Matrix Visualization...")
    plt.figure(figsize=(7, 6), dpi=300)
    
    # Calculate percentages for annot
    cm_sum = np.sum(cm)
    group_counts = [f"{val:d}" for val in cm.flatten()]
    group_percentages = [f"{val / cm_sum * 100:.1f}%" for val in cm.flatten()]
    labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Predicted Safe", "Predicted Phishing"],
        yticklabels=["Actual Safe", "Actual Phishing"],
        linewidths=1.5,
        linecolor="white",
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix — {model_name}\nAccuracy: {acc * 100:.2f}% | F1: {f1 * 100:.2f}%", fontsize=13, weight="bold", pad=15)
    plt.ylabel("True Class", fontsize=11, weight="bold")
    plt.xlabel("Predicted Class", fontsize=11, weight="bold")
    plt.tight_layout()

    cm_path = os.path.join(model_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"[+] Confusion matrix plot saved to: {cm_path}")

    # 5. Generate & Save ROC-AUC Curve
    print("[*] Generating ROC-AUC Curve...")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5.5), dpi=300)
    plt.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Chance Baseline")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, weight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, weight="bold")
    plt.title(f"Receiver Operating Characteristic (ROC) — {model_name}", fontsize=13, weight="bold", pad=15)
    plt.legend(loc="lower right", frameon=True, facecolor="#f8fafc")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    roc_path = os.path.join(model_dir, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"[+] ROC curve plot saved to: {roc_path}")

    # 6. Save Metrics Summary Text File
    summary_file = os.path.join(model_dir, "evaluation_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n")
        f.write(f"Precision: {prec * 100:.2f}%\n")
        f.write(f"Recall: {rec * 100:.2f}%\n")
        f.write(f"F1-Score: {f1 * 100:.2f}%\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"TN: {cm[0][0]} | FP: {cm[0][1]}\n")
        f.write(f"FN: {cm[1][0]} | TP: {cm[1][1]}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"[+] Textual summary saved to: {summary_file}")
    print("\n[SUCCESS] Evaluation completed successfully!")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    model_folder = os.path.join(project_root, "model")
    
    evaluate_model(model_folder)
