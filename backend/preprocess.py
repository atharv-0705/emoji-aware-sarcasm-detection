import re
import pickle
import numpy as np
from typing import List, Tuple, Dict

# ── Lazy-load emoji to avoid MemoryError ───────────────────────
# The emoji 2.10.0 data_dict.py is very large (~50MB of dicts).
# Loading it at import time alongside TensorFlow can exceed
# available RAM. We defer the import to first actual use.
_emoji_lib = None

def _get_emoji_lib():
    """Lazy loader for the emoji package."""
    global _emoji_lib
    if _emoji_lib is None:
        import emoji as _mod
        _emoji_lib = _mod
    return _emoji_lib

# ── Load word2idx at module level (loaded once on startup) ─────
_word2idx: Dict[str, int] = {}

def load_vocab(word2idx_path: str) -> None:
    """Load word2idx mapping from pickle file."""
    global _word2idx
    with open(word2idx_path, "rb") as f:
        _word2idx = pickle.load(f)
    print(f"[Preprocess] Vocabulary loaded: {len(_word2idx):,} tokens")

def get_vocab_size() -> int:
    return len(_word2idx)


# ── Step 1: Extract emojis ─────────────────────────────────────
def extract_emojis(text: str) -> Tuple[List[str], List[str]]:
    """
    Returns:
        emoji_chars  — list of emoji characters found
        emoji_names  — list of human-readable names
    Example:
        "great 😂" → (["😂"], ["face with tears of joy"])
    """
    emoji_lib = _get_emoji_lib()
    chars = [c for c in text if c in emoji_lib.EMOJI_DATA]
    names = []
    for c in chars:
        try:
            raw  = emoji_lib.demojize(c)
            name = raw.replace(":", "").replace("_", " ").strip()
        except Exception:
            name = "unknown"
        names.append(name)
    return chars, names


# ── Step 2: Convert emojis to tokens ──────────────────────────
def _emoji_to_token(char: str) -> str:
    """😂 → __emoji_face_with_tears_of_joy__"""
    try:
        raw = _get_emoji_lib().demojize(char)
        name = raw.replace(":", "").replace(" ", "_").strip()
        return f"__emoji_{name}__"
    except Exception:
        return "__emoji_unknown__"


# ── Step 3: Full text cleaning ─────────────────────────────────
def clean_text(text: str) -> Tuple[str, List[str], List[str]]:
    """
    Clean text, extract emojis, replace them with tokens.

    Returns:
        cleaned_text  — lowercased, no URLs/mentions, emojis replaced
        emoji_chars   — raw emoji characters
        emoji_names   — human-readable emoji names
    """
    t = str(text)

    # Extract emojis BEFORE any cleaning (regex might strip them)
    emoji_chars, emoji_names = extract_emojis(t)

    # Replace emojis with tokens
    for char in set(emoji_chars):
        token = _emoji_to_token(char)
        t = t.replace(char, f" {token} ")

    # Lowercase
    t = t.lower()

    # Remove URLs
    t = re.sub(r"http\S+|www\S+", " ", t)

    # Remove @mentions and #hashtags
    t = re.sub(r"@\w+|#\w+", " ", t)

    # Keep only alphanumeric, apostrophes, underscores (for emoji tokens)
    t = re.sub(r"[^a-z0-9'_ ]", " ", t)

    # Collapse multiple spaces
    t = re.sub(r"\s+", " ", t).strip()

    return t, emoji_chars, emoji_names


# ── Step 4: Tokenize ──────────────────────────────────────────
def tokenize(clean_text: str) -> List[str]:
    """Split cleaned text into list of tokens."""
    return clean_text.split()


# ── Step 5: Map tokens to indices ─────────────────────────────
def tokens_to_indices(tokens: List[str]) -> List[int]:
    """
    Convert tokens to integer indices using word2idx.
    Unknown tokens → index 1 (<UNK>).
    """
    if not _word2idx:
        raise RuntimeError("Vocabulary not loaded. Call load_vocab() first.")
    UNK = _word2idx.get("<UNK>", 1)
    return [_word2idx.get(t, UNK) for t in tokens]


# ── Step 6: Pad/Truncate ──────────────────────────────────────
def pad_sequence(indices: List[int], max_len: int = 50) -> np.ndarray:
    """
    Pad to max_len (post-padding with 0s).
    Truncate sequences longer than max_len (post-truncation).
    """
    if len(indices) >= max_len:
        return np.array(indices[:max_len], dtype=np.int32)
    else:
        padded = indices + [0] * (max_len - len(indices))
        return np.array(padded, dtype=np.int32)


# ── Main pipeline ─────────────────────────────────────────────
def preprocess_input(
    text: str,
    max_len: int = 50
) -> Tuple[np.ndarray, List[str], List[str], List[str], str]:
    """
    Full preprocessing pipeline for inference.

    Args:
        text    — raw user input text (may contain emojis)
        max_len — sequence length (must match training config)

    Returns:
        sequence     — padded integer array, shape (1, max_len)
        emoji_chars  — extracted emoji characters
        emoji_names  — emoji human-readable names
        tokens       — cleaned token list (for attention highlight)
        cleaned_text — cleaned string
    """
    # Clean + extract emojis
    cleaned, emoji_chars, emoji_names = clean_text(text)

    # Tokenize
    tokens = tokenize(cleaned)

    # Convert to indices
    indices = tokens_to_indices(tokens)

    # Pad
    padded = pad_sequence(indices, max_len=max_len)

    # Add batch dimension: (max_len,) → (1, max_len)
    sequence = padded.reshape(1, max_len)

    return sequence, emoji_chars, emoji_names, tokens, cleaned
