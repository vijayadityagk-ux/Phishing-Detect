# Machine Learning Project Report: Phishing Email Detection Model

**Domain:** Cybersecurity & Machine Learning (NLP + Tabular Heuristics)  
**Framework:** Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn, FastAPI  
**Project Location:** `C:\Users\vijay\Desktop\New folder (2)`  

---

## 1. Executive Summary & Assignment Alignment

| Task Rubric Requirement | Project Implementation | Verification File |
| :--- | :--- | :--- |
| **Train on a dataset of phishing & legitimate emails** | Curated balanced dataset of 1,500 realistic emails (750 Phishing, 750 Safe) | [`dataset/emails.csv`](dataset/emails.csv) |
| **Extract and analyze email features (URLs, keywords, etc.)** | Custom Scikit-learn Transformers extracting URL heuristics, domain spoofing, urgency keywords, and TF-IDF n-grams | [`src/feature_extractor.py`](src/feature_extractor.py) |
| **Classify emails as "Phishing" or "Safe"** | Dual inference interfaces: Command-Line (`predict.py`) and full Web GUI Dashboard (`app.py`) | [`src/predict.py`](src/predict.py)<br>[`app.py`](app.py) |
| **Display accuracy and confusion matrix** | Automated evaluation generating visual Confusion Matrix heatmap, ROC-AUC curve, and detailed classification report | [`model/confusion_matrix.png`](model/confusion_matrix.png)<br>[`src/evaluate.py`](src/evaluate.py) |

---

## 2. Dataset Architecture & Threat Taxonomy

The dataset [`dataset/emails.csv`](dataset/emails.csv) was curated with 1,500 samples balanced across 14 real-world email threat vectors:

### 2.1 Phishing Attack Vectors (750 samples)
1. **Urgent Account Suspension:** Impersonating Microsoft 365, PayPal, Apple ID, Amazon, and Wells Fargo warning of account termination within 24 hours.
2. **Fake Invoices & Billing Fraud:** Spurious Geek Squad, Norton, and McAfee recurring billing notices demanding immediate cancellation phone calls or link clicks.
3. **IT Helpdesk & Credential Harvesting:** Urgent notifications that enterprise intranet passwords expire within hours.
4. **Parcel Delivery Surcharges:** Fake FedEx, DHL, and USPS package delivery failures prompting payment of small customs redelivery fees.
5. **Financial Windfall / Crypto Grants:** Deceptive Binance, Coinbase, and international foundation reward notices.
6. **Executive / CEO Fraud:** Confidential wire transfer requests sent during emergency meetings.
7. **Government Tax Rebate Scams:** Fake IRS refund notices requesting direct banking details.

### 2.2 Legitimate / Safe Vectors (750 samples)
1. **Engineering Collaboration:** Sprint planning agendas, Jira ticket reviews, and architectural syncs.
2. **Software Development:** GitHub pull request review requests, commit notifications, and CI/CD status updates.
3. **E-Commerce Receipts:** Real Amazon, Apple, and Target shipment tracking notices.
4. **Technical Newsletters:** Python Weekly, Kaggle digests, and data science updates.
5. **Personal Communications:** Friendly casual catch-ups and weekend coffee invites.
6. **Travel Reservations:** Confirmed airline flight itineraries and check-in reminders.
7. **HR Policies:** Official corporate holiday schedules and healthcare enrollment windows.

---

## 3. Feature Engineering & Extraction Pipeline

A core challenge in email security is that textual context alone can be deceived by clean writing, while URLs alone ignore social engineering coercion. We developed a **hybrid multi-modal feature pipeline** using Scikit-Learn's `FeatureUnion`:

### 3.1 URL & Domain Security Features
- **URL Count:** Total number of embedded hyperlinks.
- **Direct IP Address Detection:** Identifies URLs using raw IPv4 addresses (e.g., `http://192.168.1.104/...`), heavily leveraged to bypass domain reputation blocklists.
- **Suspicious TLD Counter:** Flags high-risk top-level domains commonly abused by malicious actors (`.xyz`, `.top`, `.tk`, `.ru`, `.cc`, `.vip`, `.click`, `.club`).
- **Brand Subdomain Spoofing:** Detects legitimate brand names appearing inside third-party domains (e.g., `login-microsoft365.auth-portal.xyz`).
- **Obfuscation Characters:** Flags presence of `@` symbols in URLs.
- **Protocol Security:** Differentiates insecure `http://` from encrypted `https://`.

### 3.2 Psychological Urgency & NLP Features
- **Urgency Keyword Density:** Counts occurrences of panic-inducing coercion words (`urgent`, `immediately`, `suspend`, `locked`, `terminate`, `action required`).
- **Credential Harvesting Indicators:** Tracks queries for `password`, `credentials`, `login`, `verify`, and `identity`.
- **Financial Lure Metrics:** Detects `invoice`, `wire transfer`, `refund`, `crypto`, `wallet`, and `payout`.
- **Stylistic Anomaly Cues:** Measures uppercase shouting ratio, exclamation marks (`!`), question marks (`?`), and dollar signs (`$`).

### 3.3 Semantic Textual Representation
- **TF-IDF Vectorizer:** Captures contextual vocabulary across unigrams and bigrams (`ngram_range=(1, 2)`), utilizing sublinear term frequency scaling (`sublinear_tf=True`) to diminish the impact of repetitive tokens.

---

## 4. Model Training & Benchmarking Results

Four machine learning algorithms were trained on an 80% stratified training split (1,200 samples) and evaluated against a held-out 20% test split (300 samples):

| Classifier Algorithm | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier (Selected Champion)** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** |
| **Logistic Regression** | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 |
| **Linear SVM (Calibrated)** | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 |
| **Multinomial Naive Bayes** | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 |

### 4.1 Detailed Classification Report (Champion: Random Forest)

```
              precision    recall  f1-score   support

    Safe (0)     1.0000    1.0000    1.0000       150
Phishing (1)     1.0000    1.0000    1.0000       150

    accuracy                         1.0000       300
   macro avg     1.0000    1.0000    1.0000       300
weighted avg     1.0000    1.0000    1.0000       300
```

---

## 5. Confusion Matrix & Graphical Evaluation

### 5.1 Confusion Matrix Breakdown
On the 300 held-out test samples:
- **True Negatives (TN):** **150** (50.0% of total) — Legitimate emails correctly classified as Safe.
- **False Positives (FP):** **0** (0.0%) — Zero false alarms on safe workplace communication.
- **False Negatives (FN):** **0** (0.0%) — Zero undetected phishing threats.
- **True Positives (TP):** **150** (50.0% of total) — Phishing attacks successfully intercepted.

Generated visual charts saved directly in the project:
1. **Confusion Matrix Heatmap:** [`model/confusion_matrix.png`](model/confusion_matrix.png)
2. **ROC-AUC Curve:** [`model/roc_curve.png`](model/roc_curve.png)

---

## 6. Verification & Step-by-Step Usage

### 6.1 Re-Train the Model
```powershell
cd "C:\Users\vijay\Desktop\New folder (2)"
py src/train.py
```

### 6.2 Evaluate & Re-Generate Plots
```powershell
py src/evaluate.py
```

### 6.3 Test Predictions via Command Line
```powershell
# Analyze sample test email
py src/predict.py "URGENT: Your account is locked. Verify at http://192.168.1.1/login"
```

### 6.4 Launch the Interactive Web Dashboard
```powershell
py app.py
```
Open **`http://localhost:8000`** in your browser to interact with the full web dashboard, test one-click presets, and view live classification results.
