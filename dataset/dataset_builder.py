"""
=============================================================================
DATASET BUILDER & SYNTHESIZER FOR PHISHING EMAIL DETECTION
=============================================================================
Generates a comprehensive, realistic, and balanced dataset of Phishing and
Legitimate (Safe) emails containing real-world lures, URLs, keywords, and metadata.
=============================================================================
"""

import os
import random
import pandas as pd

# Set random seed for exact reproducibility
random.seed(42)

PHISHING_TEMPLATES = [
    # 1. Urgent Account Suspension / Security Alert
    {
        "subject": "URGENT: Your {service} account has been suspended due to unauthorized activity",
        "body": "Dear Customer,\n\nWe detected suspicious login attempts on your {service} account from an unrecognized IP address. To prevent permanent termination, you must verify your identity within 24 hours.\n\nClick the link below to confirm your credentials immediately:\n{phish_url}\n\nFailure to take immediate action will result in permanent account locking.\n\nSincerely,\n{service} Security Department",
        "services": ["PayPal", "Microsoft 365", "Apple ID", "Amazon", "Wells Fargo", "Netflix", "Google Workspace"]
    },
    # 2. Fake Billing / Invoice Fraud
    {
        "subject": "Invoice #{invoice_num} Pending Payment - Immediate Action Required",
        "body": "Hello,\n\nPlease find attached the outstanding invoice #{invoice_num} for your recent purchase of ${amount}. Your card ending in 4921 has been charged.\n\nIf you did not authorize this transaction, please contest and cancel the transaction immediately by logging in at:\n{phish_url}\n\nFailure to report unauthorized billing within 12 hours will finalize this charge.\n\nBilling Support Team",
        "services": ["Geek Squad", "Norton Antivirus", "McAfee Total Protection", "Adobe Creative Cloud"]
    },
    # 3. IT Helpdesk / Password Expiration
    {
        "subject": "Action Required: Your work email password expires in 2 hours",
        "body": "Employee Notice,\n\nYour organizational password is scheduled to expire today at 5:00 PM. To keep your current password and avoid losing access to your company mailbox and intranet, please complete the password synchronization portal now:\n\n{phish_url}\n\nIT Service Desk & Administration",
        "services": ["Corporate IT", "Helpdesk Support", "Global IT Operations"]
    },
    # 4. Package Delivery Issue
    {
        "subject": "Delivery Exception: Package #{tracking_id} could not be delivered",
        "body": "Dear Recipient,\n\nYour scheduled courier package #{tracking_id} was held at our distribution center due to an incomplete delivery address and unpaid customs surcharge of $2.99.\n\nUpdate your delivery address and pay the fee to reschedule delivery:\n{phish_url}\n\nIf unclaimed within 48 hours, the parcel will be returned to sender.\n\n{service} Dispatch Office",
        "services": ["FedEx Logistics", "DHL Express", "USPS Postal Service", "UPS Delivery"]
    },
    # 5. Financial Windfall / Crypto / Prize Scam
    {
        "subject": "Congratulations! You have received a grant / reward of ${amount}",
        "body": "Dear Winner,\n\nYou have been selected as a beneficiary of the Global Relief Fund worth ${amount}. To process your payout and wire transfer, claim your wallet reward at:\n\n{phish_url}\n\nProvide your wallet address and verification documents to release funds.\n\nFund Administration",
        "services": ["Coinbase Rewards", "Binance Airdrop", "International Trust"]
    },
    # 6. Urgent Wire Transfer / CEO Fraud
    {
        "subject": "URGENT & CONFIDENTIAL: Wire transfer needed before end of day",
        "body": "Hi,\n\nI am currently in an executive board meeting and unable to take calls. We need to finalize an urgent client acquisition payment today.\n\nPlease review the transfer documents and process the initial wire of ${amount} here:\n{phish_url}\n\nKeep this strictly confidential until the public announcement.\n\nBest regards,\nExecutive Management",
        "services": ["CEO Office", "Finance Director", "Managing Partner"]
    },
    # 7. Tax / Government Refund Lure
    {
        "subject": "Internal Revenue Service: Notification of Pending Tax Refund #{invoice_num}",
        "body": "Taxpayer Alert,\n\nOur records indicate you are entitled to a recalculation tax refund of ${amount} from the previous fiscal quarter. Due to paperwork discrepancies, direct deposit failed.\n\nSubmit your banking information securely through our portal:\n{phish_url}\n\nIRS Electronic Filing Division",
        "services": ["IRS Treasury", "National Revenue Service"]
    }
]

SAFE_TEMPLATES = [
    # 1. Team Collaboration & Project Standup
    {
        "subject": "Sprint Planning Agenda & Review for Q3 Milestone",
        "body": "Hi team,\n\nTomorrow at 10:00 AM we will hold our bi-weekly sprint planning meeting. Please review the updated Jira board and backlog items beforehand.\n\nYou can access the team documentation here:\n{safe_url}\n\nLet me know if you have any questions or agenda additions.\n\nBest,\nSarah Jenkins\nEngineering Manager",
        "services": ["Jira", "GitHub", "Notion", "Confluence"]
    },
    # 2. Code Review & Pull Request
    {
        "subject": "[GitHub] Pull Request #142: Fix authentication token refresh bug",
        "body": "Hi @alex,\n\nDavid has requested your review on pull request #142 in repository frontend-core.\n\nChanges summary:\n- Handled expired JWT token retry\n- Added unit tests for session timeout\n\nReview the diff on GitHub:\n{safe_url}\n\nThanks,\nGitHub Notifications",
        "services": ["GitHub", "GitLab", "Bitbucket"]
    },
    # 3. Legitimate E-Commerce / Service Receipt
    {
        "subject": "Your order #{order_id} has shipped!",
        "body": "Hello,\n\nGreat news! Your recent order #{order_id} has been packaged and is on its way to you.\n\nExpected arrival: 3 business days.\nYou can track your shipment status anytime using our official tracking portal:\n{safe_url}\n\nThank you for shopping with us!\nCustomer Care Team",
        "services": ["Amazon", "Target", "Best Buy", "Apple Store"]
    },
    # 4. Weekly Industry Newsletter
    {
        "subject": "Python & Data Science Weekly - Issue #412",
        "body": "Welcome to this week's edition of Python Weekly!\n\nTop stories this week:\n1. Scikit-learn 1.9 release highlights and new transformers.\n2. Best practices for production ML pipelines.\n3. Optimizing Pandas memory usage on large datasets.\n\nRead the full article online:\n{safe_url}\n\nTo manage your newsletter preferences or unsubscribe, click here.\n\nHappy Coding!\nThe Editorial Team",
        "services": ["Python Weekly", "Data Elixir", "Kaggle Digest"]
    },
    # 5. Travel & Flight Reservation
    {
        "subject": "Flight Confirmation - Booking Reference #{ref_id}",
        "body": "Dear Passenger,\n\nYour flight reservation is confirmed. Flight AA-284 departs on Friday at 08:30 AM from Terminal 2.\n\nPlease check in online 24 hours prior to departure and download your mobile boarding pass:\n{safe_url}\n\nSafe travels!\nCustomer Reservations",
        "services": ["Delta Airlines", "United", "American Airlines", "Lufthansa"]
    },
    # 6. Friendly Personal Message
    {
        "subject": "Coffee catch-up this weekend?",
        "body": "Hey,\n\nHope your week is going well! It has been a while since we caught up. Are you free this Saturday afternoon for coffee or a quick lunch around downtown?\n\nLet me know what time works best for you.\n\nCheers,\nMark",
        "services": ["Personal", "Family"]
    },
    # 7. HR Policy & Office Update
    {
        "subject": "Company Holiday Schedule & Year-End Benefits Information",
        "body": "Dear Colleagues,\n\nPlease take a moment to review the upcoming statutory holiday schedule and year-end healthcare flexible spending account deadlines.\n\nFull details have been published on the employee handbook portal:\n{safe_url}\n\nWarm regards,\nPeople & Culture Team",
        "services": ["Human Resources", "People Ops"]
    }
]

# Suspicious Phishing URL elements
PHISH_DOMAINS = [
    "http://192.168.1.104/secure-login/verify.php",
    "http://10.0.0.12/paypal-verification-center/account",
    "http://login-microsoft365-verify.auth-portal.xyz/login.php",
    "http://paypal-security-alert.verify-user.net/auth",
    "http://amazon-account-suspension.update-billing.top/service",
    "http://appleid-security-verification.cloud-device.info/confirm",
    "http://wellsfargo-online-secure.banking-alert.cc/signin",
    "http://dhl-express-parcel-tracking.pickup-parcel.ru/reschedule",
    "http://fedex-package-redelivery.customs-fee.biz/track",
    "http://netflix-billing-update.stream-account.vip/payment",
    "http://secure-irs-tax-refund.gov-portal.tk/rebate",
    "http://binance-airdrop-bonus.claim-token.club/wallet"
]

SAFE_DOMAINS = [
    "https://www.paypal.com/signin",
    "https://login.microsoftonline.com",
    "https://www.amazon.com/your-orders",
    "https://appleid.apple.com",
    "https://www.github.com/pulls",
    "https://jira.atlassian.com/browse",
    "https://www.fedex.com/tracking",
    "https://www.dhl.com/en/express/tracking.html",
    "https://www.netflix.com/youraccount",
    "https://www.wellsfargo.com/help",
    "https://docs.python.org/3/",
    "https://en.wikipedia.org/wiki/Machine_learning"
]

def generate_dataset(num_samples=1400):
    rows = []
    
    # 50% Phishing, 50% Safe for a clean balanced dataset
    half = num_samples // 2

    # --- Generate Phishing Samples ---
    for i in range(half):
        template = random.choice(PHISHING_TEMPLATES)
        service = random.choice(template["services"])
        phish_url = random.choice(PHISH_DOMAINS)
        invoice_num = str(random.randint(10000, 99999))
        tracking_id = f"TRK{random.randint(1000000, 9999999)}"
        amount = f"{random.randint(49, 4500):,}"

        subject = template["subject"].format(
            service=service,
            invoice_num=invoice_num,
            tracking_id=tracking_id,
            amount=amount
        )
        
        body = template["body"].format(
            service=service,
            phish_url=phish_url,
            invoice_num=invoice_num,
            tracking_id=tracking_id,
            amount=amount
        )

        rows.append({
            "subject": subject,
            "body": body,
            "full_text": f"{subject}\n\n{body}",
            "label": "Phishing",
            "is_phishing": 1
        })

    # --- Generate Safe Samples ---
    for i in range(half):
        template = random.choice(SAFE_TEMPLATES)
        service = random.choice(template["services"])
        safe_url = random.choice(SAFE_DOMAINS) if "{safe_url}" in template["body"] else ""
        order_id = str(random.randint(100000, 999999))
        ref_id = f"REF-{random.randint(1000, 9999)}"

        subject = template["subject"].format(
            service=service,
            order_id=order_id,
            ref_id=ref_id
        )
        
        body = template["body"].format(
            service=service,
            safe_url=safe_url,
            order_id=order_id,
            ref_id=ref_id
        )

        rows.append({
            "subject": subject,
            "body": body,
            "full_text": f"{subject}\n\n{body}",
            "label": "Safe",
            "is_phishing": 0
        })

    # Shuffle the dataset
    random.shuffle(rows)
    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(out_dir, "emails.csv")
    
    print("Building comprehensive email dataset...")
    df = generate_dataset(num_samples=1500)
    df.to_csv(csv_path, index=False)
    
    print(f"Dataset successfully saved to: {csv_path}")
    print(f"Total samples: {len(df)}")
    print("Class distribution:")
    print(df["label"].value_counts())
