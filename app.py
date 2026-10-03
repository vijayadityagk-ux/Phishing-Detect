"""
=============================================================================
FASTAPI WEB APPLICATION & CLASSIFICATION API SERVER
=============================================================================
Provides:
1. Interactive Web Interface (static/index.html)
2. REST API endpoint (/api/analyze) for real-time email security scoring
3. Evaluation metrics and visual charts delivery
=============================================================================
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure src/ is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "src"))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from predict import predict_email

app = FastAPI(
    title="PhishGuard ML Detection API",
    description="Scikit-Learn Machine Learning Phishing Email Detection Pipeline",
    version="1.0.0"
)

# Mount model directory to serve generated charts (confusion_matrix.png, roc_curve.png)
model_dir = os.path.join(current_dir, "model")
static_dir = os.path.join(current_dir, "static")

if os.path.exists(model_dir):
    app.mount("/model", StaticFiles(directory=model_dir), name="model")

class EmailAnalysisRequest(BaseModel):
    email_text: str

@app.get("/")
def serve_dashboard():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "PhishGuard API is running. Access /docs for API documentation."}

@app.post("/api/analyze")
def analyze_email_endpoint(payload: EmailAnalysisRequest):
    if not payload.email_text.strip():
        raise HTTPException(status_code=400, detail="email_text cannot be empty.")
    
    try:
        result = predict_email(payload.email_text, model_dir=model_dir)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health_check():
    model_file = os.path.join(model_dir, "phishing_detector.pkl")
    return {
        "status": "online",
        "model_loaded": os.path.exists(model_file)
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  🚀 Starting PhishGuard ML Web Dashboard")
    print("  🌐 Dashboard URL: http://localhost:8000")
    print("  📖 API Docs URL : http://localhost:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
