
# ================================================================
#  frontend/utils.py
#  ──────────────────
#  Helper functions for the Streamlit frontend:
#    - API calls to FastAPI backend
#    - Color utilities for sentiment/sarcasm labels
#    - Prediction history management
#    - Chart/table helpers
# ================================================================
 
import requests
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List
import streamlit as st
 
# ── Backend URL ───────────────────────────────────────────────
# Change this when deploying to Hugging Face Spaces.
# Local development: "http://localhost:8000"
# HF Spaces combined: "http://localhost:8000"  (same container)
BACKEND_URL = "http://localhost:8000"
 
 
# ══════════════════════════════════════════════════════════════
# API CALLS
# ══════════════════════════════════════════════════════════════
 
def check_health() -> Dict[str, Any]:
    """Poll /health endpoint. Returns status dict."""
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"status": "not_ready", "error": "Cannot connect to backend"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
 
 
def call_predict(text: str) -> Optional[Dict[str, Any]]:
    """
    POST /predict with user text.
    Returns prediction dict or None on error.
    """
    try:
        r = requests.post(
            f"{BACKEND_URL}/predict",
            json    = {"text": text},
            timeout = 30,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Make sure the API is running.")
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The model may still be loading.")
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        st.error(f"🚨 API Error: {detail}")
    return None
 
 
# ── Cached Fetch Helpers (only cache successful non-empty requests) ──
 
@st.cache_data(ttl=300)
def _fetch_metrics() -> List[Dict[str, Any]]:
    r = requests.get(f"{BACKEND_URL}/metrics", timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        raise ValueError("No metrics data returned")
    return data
 
@st.cache_data(ttl=300)
def _fetch_correlation_stats() -> List[Dict[str, Any]]:
    r = requests.get(f"{BACKEND_URL}/stats", timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        raise ValueError("No correlation stats data returned")
    return data
 
@st.cache_data(ttl=300)
def _fetch_emoji_data(top_n: int) -> List[Dict[str, Any]]:
    r = requests.get(f"{BACKEND_URL}/emojis?top_n={top_n}", timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        raise ValueError("No emoji data returned")
    return data
 
# ── Public APIs ──
 
def get_metrics() -> pd.DataFrame:
    """Fetch model metrics (cached 5 min)."""
    try:
        data = _fetch_metrics()
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()
 
def get_correlation_stats() -> pd.DataFrame:
    """Fetch correlation stats (cached 5 min)."""
    try:
        data = _fetch_correlation_stats()
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()
 
def get_emoji_data(top_n: int = 20) -> pd.DataFrame:
    """Fetch top emoji analysis (cached 5 min)."""
    try:
        data = _fetch_emoji_data(top_n)
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()
 
 
# ══════════════════════════════════════════════════════════════
# COLOR & STYLE UTILITIES
# ══════════════════════════════════════════════════════════════
 
COLORS = {
    "sarcastic"    : "#C62828",    # deep red
    "non_sarcastic": "#1B5E20",    # deep green
    "positive"     : "#1565C0",    # blue
    "negative"     : "#BF360C",    # dark orange
    "neutral"      : "#455A64",    # grey
    "bully"        : "#880E4F",    # dark pink
    "not_bully"    : "#1A237E",    # dark blue
    "confidence_hi": "#00695C",    # teal
    "confidence_lo": "#E65100",    # amber
    "highlight_max": "#FF1744",    # bright red
    "highlight_mid": "#FF9100",    # amber
    "highlight_low": "#FFEA00",    # yellow
    "card_bg"      : "#F5F5F5",
    "card_border"  : "#BDBDBD",
}
 
ICONS = {
    "Sarcastic"    : "😏",
    "Non-Sarcastic": "😊",
    "Positive"     : "😀",
    "Negative"     : "😡",
    "Neutral"      : "😐",
    "Bully"        : "⚠️",
    "Not-Bully"    : "✅",
}
 
def sarcasm_color(label: str) -> str:
    return COLORS["sarcastic"] if label == "Sarcastic" else COLORS["non_sarcastic"]
 
def sentiment_color(label: str) -> str:
    if label == "Positive":   return COLORS["positive"]
    if label == "Negative":   return COLORS["negative"]
    return COLORS["neutral"]
 
def confidence_color(conf: float) -> str:
    return COLORS["confidence_hi"] if conf >= 0.7 else COLORS["confidence_lo"]
 
def highlight_color(importance: float) -> str:
    if importance >= 0.7:  return COLORS["highlight_max"]
    if importance >= 0.45: return COLORS["highlight_mid"]
    return COLORS["highlight_low"]
 
def importance_to_bg(importance: float) -> str:
    """Convert importance [0,1] to a CSS rgba background string."""
    r = int(255 * importance)
    g = int(200 * (1 - importance))
    return f"rgba({r},{g},50,0.25)"
 
 
# ══════════════════════════════════════════════════════════════
# PREDICTION HISTORY
# ══════════════════════════════════════════════════════════════
 
def init_history():
    """Initialize session state for prediction history."""
    if "history" not in st.session_state:
        st.session_state.history = []
 
 
def add_to_history(result: Dict[str, Any]):
    """Add a prediction result to session history (max 50 entries)."""
    init_history()
    entry = {
        "timestamp"          : result.get("timestamp", datetime.now().isoformat()),
        "input_text"         : result["input_text"][:60] + "..." if len(result["input_text"]) > 60 else result["input_text"],
        "sarcasm_prediction" : result["sarcasm_prediction"],
        "confidence"         : f"{result['confidence']:.1%}",
        "sentiment"          : result["sentiment"],
        "emotion"            : result["emotion"],
        "bully"              : result["bully"],
        "emojis"             : "".join(result.get("detected_emojis", [])) or "—",
    }
    st.session_state.history.insert(0, entry)       # newest first
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[:50]
 
 
def get_history_df() -> pd.DataFrame:
    """Return prediction history as a formatted DataFrame."""
    init_history()
    if not st.session_state.history:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.history).rename(columns={
        "timestamp"          : "Time",
        "input_text"         : "Input Text",
        "sarcasm_prediction" : "Sarcasm",
        "confidence"         : "Confidence",
        "sentiment"          : "Sentiment",
        "emotion"            : "Emotion",
        "bully"              : "Bully",
        "emojis"             : "Emojis",
    })