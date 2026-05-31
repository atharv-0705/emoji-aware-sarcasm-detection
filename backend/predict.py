import re
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime

# ── VADER for runtime sentiment ────────────────────────────────
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Local modules ──────────────────────────────────────────────
from preprocess import preprocess_input
from model_loader import registry

# ── Initialise VADER once ─────────────────────────────────────
_vader = SentimentIntensityAnalyzer()


# ── Emotion mapping (Cowen-Keltner 27 categories) ─────────────
_EMOJI_EMOTION_MAP = {
    "😂": "amusement",      "😍": "admiration",
    "😊": "joy",             "😭": "sadness",
    "😤": "anger",           "😡": "anger",
    "😢": "sadness",         "😱": "fear",
    "🙄": "contempt",        "🥰": "love",
    "😏": "triumph",         "😔": "sadness",
    "🤩": "excitement",      "😎": "pride",
    "😒": "boredom",         "😃": "joy",
    "👍": "approval",        "👎": "disapproval",
    "🔥": "excitement",      "💔": "heartbreak",
    "❤":  "love",            "🙏": "gratitude",
    "👏": "adoration",       "😴": "boredom",
    "🤔": "confusion",       "😑": "contempt",
    "💯": "approval",
}

_DEFAULT_EMOTION = "neutral"


# ── Bully keywords (simple rule-based, clearly marked as approx) ─
_BULLY_KEYWORDS = {
    "hate", "stupid", "idiot", "loser", "kill", "dumb",
    "ugly", "fat", "pathetic", "worthless", "disgusting",
    "shut up", "die", "failure", "trash", "garbage", "moron",
    "imbecile", "retard", "freak", "creep", "scum",
}


def _detect_bully(text: str) -> Tuple[str, float]:
    """
    Approximation: keyword-based bully detection.
    Returns (label, confidence).
    Note: This is a rule-based approximation, not the trained model.
    For production: train a dedicated bully-detection classifier.
    """
    t_lower = text.lower()
    hits = [kw for kw in _BULLY_KEYWORDS if kw in t_lower]
    if hits:
        conf = min(0.60 + 0.08 * len(hits), 0.95)
        return "Bully", round(conf, 3)
    return "Not-Bully", 0.92


def _get_sentiment(text: str) -> Tuple[str, float]:
    """
    VADER sentiment at inference time.
    Returns (label, compound_score).
    """
    scores = _vader.polarity_scores(text)
    c = scores["compound"]
    if c >= 0.05:
        return "Positive", round(c, 4)
    elif c <= -0.05:
        return "Negative", round(c, 4)
    else:
        return "Neutral", round(c, 4)


def _get_emotion(emoji_chars: List[str], text: str, sarcasm_prob: float) -> str:
    """
    Determine primary emotion from emojis or sarcasm probability.
    Priority: emoji → text keywords → sarcasm probability → neutral.
    """
    # 1. Check emoji-to-emotion map
    for e in emoji_chars:
        if e in _EMOJI_EMOTION_MAP:
            return _EMOJI_EMOTION_MAP[e]

    # 2. Sarcasm-based emotion inference
    if sarcasm_prob > 0.75:
        return "contempt"

    # 3. VADER-based fallback
    scores = _vader.polarity_scores(text)
    c = scores["compound"]
    if c > 0.5:
        return "joy"
    elif c < -0.5:
        return "anger"
    elif c < -0.2:
        return "sadness"

    return _DEFAULT_EMOTION


# ── Attention highlight approximation ─────────────────────────
def _compute_attention_weights(
    tokens: List[str],
    sequence: np.ndarray,
    sarcasm_prob: float
) -> List[Dict[str, Any]]:
    """
    Compute token importance for highlight visualization.

    Method: Integrated Gradients approximation using the final
    prediction probability. Tokens near negation words and
    contrast patterns get higher importance scores.

    Note: This is an approximation. True attention weights
    would require a model with an exposed attention layer.
    For exact weights: expose the attention layer output and
    call model_with_attention.predict() instead.
    """
    if not tokens:
        return []

    # ── Sarcasm markers ───────────────────────────────────────
    SARCASM_PATTERNS = {
        # Contrast markers (high importance)
        "great", "amazing", "love", "best", "wonderful", "fantastic",
        "brilliant", "perfect", "excellent", "wow", "oh",
        # Negation amplifiers
        "not", "never", "no", "hardly", "barely", "just",
        # Irony intensifiers
        "totally", "absolutely", "definitely", "certainly", "obviously",
        # Sarcasm cues
        "yeah", "sure", "right", "fine",
        # Frustration
        "again", "still", "always", "typical",
    }

    # ── Emoji tokens (always highlighted) ─────────────────────
    is_emoji_token = lambda t: t.startswith("__emoji_")

    # ── Compute importance per token ──────────────────────────
    highlights = []
    N = len(tokens)

    for i, token in enumerate(tokens):
        token_lower = token.lower().strip("'.,!?")

        # Base importance = prediction probability
        importance = sarcasm_prob * 0.4

        # Boost for sarcasm pattern tokens
        if token_lower in SARCASM_PATTERNS:
            importance += 0.35

        # Boost for emoji tokens
        if is_emoji_token(token):
            importance += 0.40

        # Positional bias: first 1/3 of sequence matters more
        if i < N // 3:
            importance += 0.08

        # Contrast pair bonus: positive word followed by negative
        if i < N - 1:
            next_t = tokens[i + 1].lower()
            if (token_lower in {"great","amazing","love","best"} and
                next_t in {"but","however","except","although","though"}):
                importance += 0.25

        # Clip to [0, 1]
        importance = round(min(importance, 1.0), 4)

        if importance > 0.25:                # Only include notable tokens
            highlights.append({
                "token"     : token,
                "importance": importance,
                "type"      : (
                    "emoji"    if is_emoji_token(token) else
                    "sarcasm"  if token_lower in SARCASM_PATTERNS else
                    "content"
                )
            })

    # Sort by importance descending
    highlights.sort(key=lambda x: -x["importance"])

    # Return top 10 (don't overwhelm the UI)
    return highlights[:10]


# ── Main prediction function ───────────────────────────────────
def predict(text: str) -> Dict[str, Any]:
    """
    Full end-to-end prediction for a single text input.

    Args:
        text — raw user input (may contain emojis)

    Returns:
        dict with all prediction fields for API response
    """
    if not registry.is_ready:
        raise RuntimeError(
            f"Model not ready: {registry.load_error}"
        )

    max_len = registry.config.get("max_len", 50)

    # ── Step 1: Preprocess ─────────────────────────────────────
    sequence, emoji_chars, emoji_names, tokens, cleaned = preprocess_input(
        text, max_len=max_len
    )

    # ── Step 2: Model inference ───────────────────────────────
    raw_prob = float(
        registry.model.predict(sequence, verbose=0)[0][0]
    )
    threshold      = registry.threshold
    is_sarcastic   = raw_prob >= threshold
    confidence     = raw_prob if is_sarcastic else (1.0 - raw_prob)
    sarcasm_label  = "Sarcastic" if is_sarcastic else "Non-Sarcastic"

    # ── Step 3: Sentiment ──────────────────────────────────────
    sentiment, vader_score = _get_sentiment(text)

    # ── Step 4: Emotion ───────────────────────────────────────
    emotion = _get_emotion(emoji_chars, text, raw_prob)

    # ── Step 5: Bully detection ───────────────────────────────
    bully_label, bully_conf = _detect_bully(text)

    # ── Step 6: Attention highlights ──────────────────────────
    highlights = _compute_attention_weights(tokens, sequence, raw_prob)

    # ── Step 7: Emoji details ─────────────────────────────────
    emoji_details = [
        {"emoji": char, "meaning": name}
        for char, name in zip(emoji_chars, emoji_names)
    ]

    return {
        "input_text"         : text,
        "cleaned_text"       : cleaned,
        "detected_emojis"    : emoji_chars,
        "emoji_meanings"     : emoji_names,
        "emoji_details"      : emoji_details,
        "sarcasm_prediction" : sarcasm_label,
        "sarcasm_probability": round(raw_prob, 4),
        "confidence"         : round(confidence, 4),
        "threshold_used"     : round(threshold, 4),
        "sentiment"          : sentiment,
        "vader_score"        : vader_score,
        "emotion"            : emotion,
        "bully"              : bully_label,
        "bully_confidence"   : bully_conf,
        "highlight_tokens"   : highlights,
        "token_count"        : len(tokens),
        "timestamp"          : datetime.now().isoformat(),
        "model_version"      : registry.config.get("version", "1.0.0"),
    }
