import os, json
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
 
from model_loader import load_all_artifacts, get_registry
from predict import predict as run_predict
 
# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path.cwd()   # current working directory
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR      = BASE_DIR / "data"
 
# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title       = "Sarcoji Sarcasm Detection API",
    description = (
        "Detect sarcasm, sentiment, emotion, and bullying "
        "in social media text using Hybrid CNN-BiLSTM-Attention "
        "with GloVe + Word2Vec + Emoji2Vec fusion."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS (allow Streamlit frontend on any origin) ─────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)
 
# ── Load model at startup ─────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    load_all_artifacts()
 
 
# ── Request/Response schemas ──────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length = 1,
        max_length = 1000,
        example    = "Wow amazing customer service 😂 I only waited 3 hours!",
        description = "Input review text (may contain emojis)"
    )
 
class HighlightToken(BaseModel):
    token     : str
    importance: float
    type      : str
 
class EmojiDetail(BaseModel):
    emoji  : str
    meaning: str
 
class PredictResponse(BaseModel):
    input_text         : str
    cleaned_text       : str
    detected_emojis    : List[str]
    emoji_meanings     : List[str]
    emoji_details      : List[EmojiDetail]
    sarcasm_prediction : str
    sarcasm_probability: float
    confidence         : float
    threshold_used     : float
    sentiment          : str
    vader_score        : float
    emotion            : str
    bully              : str
    bully_confidence   : float
    highlight_tokens   : List[HighlightToken]
    token_count        : int
    timestamp          : str
    model_version      : str
 
 
# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════
 
@app.get("/", tags=["General"])
def root():
    """Welcome endpoint."""
    return {
        "message"    : "🎯 Sarcoji API is running",
        "model"      : "CNN-BiLSTM-Attention (GloVe+W2V+Emoji2Vec)",
        "docs"       : "/docs",
        "health"     : "/health",
        "predict"    : "/predict",
        "version"    : "1.0.0",
    }
 
 
@app.get("/health", tags=["General"])
def health_check(reg = Depends(get_registry)):
    """
    Health check — returns model loaded status.
    Frontend polls this on startup.
    """
    return {
        "status"        : "ready" if reg.is_ready else "not_ready",
        "model_loaded"  : reg.is_ready,
        "vocab_size"    : len(reg.word2idx) if reg.word2idx else 0,
        "threshold"     : reg.threshold,
        "error"         : reg.load_error,
        "config"        : reg.config,
    }
 
 
@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_endpoint(request: PredictRequest, reg = Depends(get_registry)):
    """
    Full sarcasm detection prediction.
 
    Input  : { "text": "Your review text here 😂" }
    Output : Sarcasm prediction, confidence, sentiment, emotion,
             bully detection, attention highlights, emoji meanings.
    """
    if not reg.is_ready:
        raise HTTPException(
            status_code = 503,
            detail      = f"Model not ready: {reg.load_error}. "
                          f"Check /health for details."
        )
    try:
        result = run_predict(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 
@app.get("/metrics", tags=["Analytics"])
def get_metrics():
    """
    Return model performance metrics from metrics_matrix.csv.
    Displayed in the 'Model Performance' tab of Streamlit frontend.
    """
    try:
        path = DATA_DIR / "metrics_matrix.csv"
        if not path.exists():
            return {"error": "metrics_matrix.csv not found", "data": []}
        df = pd.read_csv(path)
        return {"data": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 
@app.get("/stats", tags=["Analytics"])
def get_correlation_stats():
    """
    Return statistical correlation results from correlation_summary.csv.
    """
    try:
        path = DATA_DIR / "correlation_summary.csv"
        if not path.exists():
            return {"error": "correlation_summary.csv not found", "data": []}
        df = pd.read_csv(path)
        return {"data": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 
@app.get("/emojis", tags=["Analytics"])
def get_emoji_analysis(top_n: int = 20):
    """
    Return top N emojis from emoji_analysis.csv.
    Default: top 20.
    """
    try:
        path = DATA_DIR / "emoji_analysis.csv"
        if not path.exists():
            return {"error": "emoji_analysis.csv not found", "data": []}
        df = pd.read_csv(path)
        top = df.head(top_n)
        return {"data": top.to_dict(orient="records"), "total_unique": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
