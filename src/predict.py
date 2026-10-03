"""
=============================================================================
INTERACTIVE INFERENCE & PREDICTION UTILITY
=============================================================================
Classifies raw email text or files as "Phishing" or "Safe" using the trained model.
Provides confidence probability and security heuristic breakdowns.
=============================================================================
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import joblib
from feature_extractor import (
    URL_REGEX, IP_REGEX, SUSPICIOUS_TLDS, TARGET_BRANDS,
    URGENCY_KEYWORDS, FINANCIAL_KEYWORDS, CREDENTIAL_KEYWORDS
)

def inspect_heuristics(text):
    lower_text = text.lower()
    urls = URL_REGEX.findall(text)
    
    triggers = []
    
    # URL checks
    if any(IP_REGEX.search(u) for u in urls):
        triggers.append("⚠️ Contains direct IP-based URL (classic phishing indicator)")
    
    for url in urls:
        u_lower = url.lower()
        if any(u_lower.endswith(tld) or (tld + "/") in u_lower for tld in SUSPICIOUS_TLDS):
            triggers.append(f"⚠️ Uses high-risk/abused TLD in link: {url}")
        if any(brand in u_lower for brand in TARGET_BRANDS):
            triggers.append(f"⚠️ Brand name spoofing detected in URL: {url}")
        if "@" in url:
            triggers.append(f"⚠️ Obfuscated '@' syntax found in URL: {url}")

    # Keyword checks
    found_urgency = [kw for kw in URGENCY_KEYWORDS if kw in lower_text]
    if found_urgency:
        triggers.append(f"⚡ Urgency/Coercion keywords detected: {', '.join(found_urgency[:4])}")

    found_credentials = [kw for kw in CREDENTIAL_KEYWORDS if kw in lower_text]
    if found_credentials:
        triggers.append(f"🔑 Credential harvesting keywords detected: {', '.join(found_credentials[:4])}")

    found_financial = [kw for kw in FINANCIAL_KEYWORDS if kw in lower_text]
    if found_financial:
        triggers.append(f"💰 Financial lure/billing keywords detected: {', '.join(found_financial[:4])}")

    return urls, triggers

def predict_email(text, model_dir=None):
    if model_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.abspath(os.path.join(current_dir, "..", "model"))
    
    model_path = os.path.join(model_dir, "phishing_detector.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run train.py first.")

    pipeline = joblib.load(model_path)

    # Prediction
    prediction = pipeline.predict([text])[0]
    
    confidence = 0.5
    if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
        probabilities = pipeline.predict_proba([text])[0]
        confidence = probabilities[prediction]
    
    label = "Phishing" if prediction == 1 else "Safe"
    urls, triggers = inspect_heuristics(text)

    return {
        "label": label,
        "is_phishing": bool(prediction == 1),
        "confidence": float(confidence),
        "urls_detected": urls,
        "triggers": triggers
    }

def print_report(result):
    print("\n" + "=" * 60)
    print("           EMAIL SECURITY ANALYSIS REPORT")
    print("=" * 60)

    if result["is_phishing"]:
        print(f" VERDICT    : 🚨 PHISHING DETECTED")
        print(f" CONFIDENCE : {result['confidence'] * 100:.2f}% Phishing Probability")
        print(f" RISK LEVEL : HIGH THREAT")
    else:
        print(f" VERDICT    : ✅ SAFE / LEGITIMATE")
        print(f" CONFIDENCE : {result['confidence'] * 100:.2f}% Safe Probability")
        print(f" RISK LEVEL : LOW / CLEAN")

    print("-" * 60)
    print(f" URLs Found ({len(result['urls_detected'])}):")
    if result['urls_detected']:
        for u in result['urls_detected']:
            print(f"   • {u}")
    else:
        print("   (None detected)")

    print("\n Security Heuristic Triggers:")
    if result['triggers']:
        for t in result['triggers']:
            print(f"   {t}")
    else:
        print("   ✅ No high-risk coercive or domain anomalies identified.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if os.path.isfile(arg):
            with open(arg, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = " ".join(sys.argv[1:])
        res = predict_email(content)
        print_report(res)
    else:
        # Interactive demo prompt
        demo_phish = """URGENT: Your Microsoft 365 Password expires in 2 hours.
Click here immediately to verify account identity or risk account termination:
http://login-microsoft365-verify.auth-portal.xyz/login.php"""
        
        print("[*] Running inference on sample test lure:")
        res = predict_email(demo_phish)
        print_report(res)
