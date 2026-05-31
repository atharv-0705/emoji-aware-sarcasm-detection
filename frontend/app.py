
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
 
from utils import (
    check_health, call_predict, get_metrics,
    get_correlation_stats, get_emoji_data,
    add_to_history, get_history_df,
    sarcasm_color, sentiment_color, confidence_color,
    highlight_color, importance_to_bg, COLORS, ICONS
)
 
# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "Sarcoji — Sarcasm Detection",
    page_icon  = "😏",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)
 
# ══════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Fonts & base ──────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
 
/* ── Header ────────────────────────────────────────────── */
.main-header {
    background: linear-gradient(135deg, #1565C0 0%, #283593 50%, #4A148C 100%);
    color: white; padding: 28px 32px; border-radius: 10px;
    margin-bottom: 24px;
}
.main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
.main-header p  { font-size: 1rem; opacity: 0.88; margin: 6px 0 0 0; }
 
/* ── Metric cards ──────────────────────────────────────── */
.metric-card {
    background: #fff; border-radius: 10px; padding: 20px 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
    border-top: 4px solid;
}
.metric-value { font-size: 2rem; font-weight: 700; }
.metric-label { font-size: 0.85rem; color: #666; margin-top: 4px; }
 
/* ── Result cards ──────────────────────────────────────── */
.result-card {
    background: #fff; border-radius: 8px; padding: 18px 20px;
    margin: 6px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border-left: 5px solid;
}
.result-label { font-size: 0.78rem; font-weight: 600;
                text-transform: uppercase; letter-spacing: 0.05em;
                color: #888; margin-bottom: 4px; }
.result-value { font-size: 1.35rem; font-weight: 700; }
 
/* ── Highlight tokens ──────────────────────────────────── */
.token-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
.token-chip {
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.88rem; font-weight: 500;
    border: 1px solid rgba(0,0,0,0.12);
}
 
/* ── Confidence bar ────────────────────────────────────── */
.conf-bar-outer {
    background: #eee; border-radius: 8px; height: 12px;
    overflow: hidden; margin: 8px 0;
}
.conf-bar-inner {
    height: 100%; border-radius: 8px;
    transition: width 0.5s ease;
}
 
/* ── Status badge ──────────────────────────────────────── */
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 0.80rem; font-weight: 600; color: white;
}
 
/* ── Input box ─────────────────────────────────────────── */
.stTextArea textarea {
    font-size: 1.05rem !important;
    border: 2px solid #E0E0E0 !important;
    border-radius: 8px !important;
}
.stTextArea textarea:focus { border-color: #1565C0 !important; }
 
/* ── Tabs ──────────────────────────────────────────────── */
.stTabs [data-baseweb="tab"] {
    font-size: 1rem; font-weight: 500; padding: 10px 20px;
}
.stTabs [aria-selected="true"] {
    color: #1565C0 !important; border-bottom: 3px solid #1565C0 !important;
}
 
/* ── Hide streamlit branding ───────────────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════════════════════
# HELPER: RESULT DISPLAY FUNCTIONS
# ══════════════════════════════════════════════════════════════
 
def render_metric_card(label, value, color, subtitle=""):
    st.markdown(f"""
    <div class="metric-card" style="border-top-color:{color}">
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-label">{label}</div>
        {'<div style="font-size:0.75rem;color:#aaa;margin-top:3px">'+subtitle+'</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)
 
 
def render_result_card(label, icon, value, color, subtitle=""):
    st.markdown(f"""
    <div class="result-card" style="border-left-color:{color}">
        <div class="result-label">{icon} {label}</div>
        <div class="result-value" style="color:{color}">{value}</div>
        {'<div style="font-size:0.80rem;color:#aaa;margin-top:3px">'+subtitle+'</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)
 
 
def render_confidence_bar(conf: float, color: str):
    pct = int(conf * 100)
    st.markdown(f"""
    <div style="margin: 4px 0 8px 0">
        <div style="font-size:0.80rem;color:#666;margin-bottom:4px">
            Confidence: <b>{pct}%</b>
        </div>
        <div class="conf-bar-outer">
            <div class="conf-bar-inner"
                 style="width:{pct}%; background:{color}">
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
 
def render_highlight_tokens(highlights):
    if not highlights:
        st.info("No notable tokens detected for this input.")
        return
 
    st.markdown("**Key tokens influencing the prediction:**")
    chips_html = '<div class="token-row">'
    for h in highlights:
        token = h["token"]
        imp   = h["importance"]
        typ   = h["type"]
        bg    = importance_to_bg(imp)
        color = highlight_color(imp)
        label = "🎭" if typ == "sarcasm" else ("😊" if typ == "emoji" else "📝")
        chips_html += (
            f'<div class="token-chip" '
            f'style="background:{bg};border-color:{color};color:{color}">'
            f'{label} {token.replace("__emoji_","").replace("__","").replace("_"," ")}'
            f' <span style="font-size:0.75rem;opacity:0.8">({imp:.0%})</span>'
            f'</div>'
        )
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)
 
    st.caption(
        "⚠️ Approximate highlights based on token importance scoring. "
        "For exact attention weights, expose the model's attention layer."
    )
 
 
# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>😏 Sarcoji — Sarcasm Detection</h1>
    <p>
        Hybrid Deep Learning &nbsp;|&nbsp;
        GloVe + Word2Vec + Emoji2Vec &nbsp;|&nbsp;
        CNN-BiLSTM-Attention &nbsp;|&nbsp;
        Stepwise Transfer Learning
    </p>
</div>
""", unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════════════════════
# SIDEBAR — health & quick stats
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Model Status")
    health = check_health()
    if health.get("status") == "ready":
        st.success("✅ Model Ready")
        cfg = health.get("config", {})
        st.markdown(f"""
        - **Version:** {cfg.get('version','—')}
        - **Vocab:** {health.get('vocab_size',0):,} tokens
        - **Threshold:** {health.get('threshold',0.5):.4f}
        - **Max Len:** {cfg.get('max_len',50)} tokens
        """)
    else:
        st.error(f"❌ {health.get('error','Not ready')}")
        st.markdown("**Troubleshooting:**")
        st.code("cd backend\nuvicorn main:app --reload")
 
    st.divider()
    st.markdown("### 📌 Quick Links")
    st.markdown("""
    - [🔗 API Docs](http://localhost:8000/docs)
    - [📁 GitHub](https://github.com/atharv-0705/sarcoji)
    - [📄 Paper](#)
    """)
 
 
# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab_predict, tab_analytics, tab_model, tab_history, tab_about = st.tabs([
    "🎯  Predict",
    "📊  Analytics",
    "🧠  Model Info",
    "📋  History",
    "ℹ️   About"
])
 
 
# ════════════════════════════════════════════════════════════
# TAB 1: PREDICT
# ════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("### 🎯 Sarcasm Detection")
    st.markdown(
        "Enter any text — include emojis! The model will detect sarcasm, "
        "sentiment, emotion, and more."
    )
 
    # ── Example buttons ───────────────────────────────────────
    st.markdown("**Try an example:**")
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    EXAMPLES = [
        "Oh wow, absolutely LOVE it when apps crash 😂",
        "This is genuinely the best app I've ever used 😍",
        "Yeah sure, waiting 3 hours for support is totally acceptable 🙄",
        "Great update! Now it doesn't work at all 👏",
    ]
    example_clicked = None
    for col, example in zip([col_e1, col_e2, col_e3, col_e4], EXAMPLES):
        with col:
            if st.button(example[:30] + "...", use_container_width=True, key=f"ex_{example[:10]}"):
                example_clicked = example
 
    # ── Text input ────────────────────────────────────────────
    default_text = example_clicked if example_clicked else ""
    if "last_example" not in st.session_state:
        st.session_state.last_example = ""
    if example_clicked:
        st.session_state.last_example = example_clicked
 
    user_input = st.text_area(
        label       = "Enter review text",
        value       = st.session_state.get("last_example", ""),
        height      = 120,
        max_chars   = 1000,
        placeholder = "Type your review here (emojis welcome!)...",
        label_visibility = "collapsed"
    )
 
    char_count = len(user_input)
    st.caption(f"{char_count}/1000 characters")
 
    # ── Predict button ────────────────────────────────────────
    predict_btn = st.button(
        "🔍  Analyse Text",
        type             = "primary",
        use_container_width = True,
        disabled         = (not user_input.strip() or char_count < 3)
    )
 
    if predict_btn and user_input.strip():
        with st.spinner("🧠 Analysing..."):
            result = call_predict(user_input.strip())
 
        if result:
            add_to_history(result)
 
            st.markdown("---")
            st.markdown("### 📊 Results")
 
            # ── Top-level: sarcasm result (full width) ────────
            s_label = result["sarcasm_prediction"]
            s_color = sarcasm_color(s_label)
            s_icon  = ICONS.get(s_label, "")
            s_conf  = result["confidence"]
 
            st.markdown(f"""
            <div class="result-card" style="border-left-color:{s_color};
                 background: linear-gradient(135deg, #fff 85%, {s_color}18 100%)">
                <div class="result-label">🎭 SARCASM PREDICTION</div>
                <div class="result-value" style="color:{s_color};font-size:1.8rem">
                    {s_icon} {s_label}
                </div>
                <div style="font-size:0.85rem;color:#777;margin-top:4px">
                    Raw probability: {result['sarcasm_probability']:.4f} 
                    &nbsp;|&nbsp; Threshold: {result['threshold_used']:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            render_confidence_bar(s_conf, s_color)
 
            st.divider()
 
            # ── Row 2: sentiment / emotion / bully ───────────
            col1, col2, col3 = st.columns(3)
            with col1:
                sent   = result["sentiment"]
                v_score = result["vader_score"]
                render_result_card(
                    "Sentiment", ICONS.get(sent, "💬"), sent,
                    sentiment_color(sent),
                    subtitle=f"VADER score: {v_score:+.4f}"
                )
            with col2:
                render_result_card(
                    "Emotion", "🎭",
                    result["emotion"].title(), "#6A1B9A"
                )
            with col3:
                bully = result["bully"]
                b_col = COLORS["bully"] if bully == "Bully" else COLORS["not_bully"]
                render_result_card(
                    "Bully Detection", ICONS.get(bully, "🛡️"),
                    bully, b_col,
                    subtitle=f"Confidence: {result['bully_confidence']:.1%}"
                )
 
            st.divider()
 
            # ── Row 3: emojis + highlights ────────────────────
            col_emj, col_hl = st.columns([1, 2])
 
            with col_emj:
                st.markdown("**😊 Detected Emojis**")
                details = result.get("emoji_details", [])
                if details:
                    for d in details:
                        st.markdown(
                            f"`{d['emoji']}`  &nbsp; **{d['meaning']}**"
                        )
                else:
                    st.info("No emojis detected in this text.")
 
            with col_hl:
                st.markdown("**🔦 Attention Highlights**")
                render_highlight_tokens(result.get("highlight_tokens", []))
 
            # ── Raw JSON expander ─────────────────────────────
            with st.expander("🔍 View Raw JSON Response", expanded=False):
                st.json(result)
 
 
# ════════════════════════════════════════════════════════════
# TAB 2: ANALYTICS
# ════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown("### 📊 Dataset Analytics")
 
    # ── Emoji frequency chart ─────────────────────────────────
    st.markdown("#### 😊 Top Emojis — Sarcastic vs Non-Sarcastic Distribution")
    emoji_df = get_emoji_data(top_n=15)
    if not emoji_df.empty and "Count" in emoji_df.columns:
        # Try to find positive/negative columns
        pos_col = next((c for c in emoji_df.columns
                        if "positive" in c.lower()), None)
        neg_col = next((c for c in emoji_df.columns
                        if "negative" in c.lower()), None)
        emoji_label_col = next((c for c in emoji_df.columns
                                if "emoji" in c.lower() and "char" not in c.lower()
                                and "name" not in c.lower().replace("emoji_name","")),
                               emoji_df.columns[1] if len(emoji_df.columns) > 1 else None)
 
        if pos_col and neg_col and emoji_label_col:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name = "Non-Sarcastic / Positive",
                x    = emoji_df[emoji_label_col].astype(str).str[:25],
                y    = emoji_df[pos_col],
                marker_color = COLORS["positive"],
            ))
            fig.add_trace(go.Bar(
                name = "Sarcastic / Negative",
                x    = emoji_df[emoji_label_col].astype(str).str[:25],
                y    = emoji_df[neg_col],
                marker_color = COLORS["sarcastic"],
            ))
            fig.update_layout(
                barmode      = "group",
                title        = "Top 15 Emojis — Sentiment Breakdown",
                xaxis_title  = "Emoji",
                yaxis_title  = "Count",
                plot_bgcolor = "white",
                height       = 420,
                legend       = dict(orientation="h", y=1.08),
                margin       = dict(l=40, r=40, t=60, b=80),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(emoji_df.head(15), use_container_width=True)
    else:
        st.info("Emoji data not available. Check that emoji_analysis.csv is in backend/data/")
 
    # ── Correlation stats ─────────────────────────────────────
    st.markdown("#### 📐 Statistical Correlation Analysis")
    corr_df = get_correlation_stats()
    if not corr_df.empty:
        # Key metrics in cards
        stat_cols = st.columns(4)
        for i, row in corr_df.iterrows():
            if i >= 4:
                break
            stat_name  = str(row.get("Statistic", row.get("statistic", f"Metric {i+1}")))
            stat_val   = row.get("Value", row.get("value", "—"))
            stat_str   = row.get("Strength", row.get("strength", ""))
            col = stat_cols[i % 4]
            with col:
                render_metric_card(
                    stat_name[:25],
                    f"{float(stat_val):.4f}" if isinstance(stat_val, (int, float)) else str(stat_val),
                    COLORS["sarcastic"] if "χ" in stat_name else COLORS["positive"],
                    subtitle = stat_str
                )
        st.markdown("")
        st.dataframe(corr_df, use_container_width=True, hide_index=True)
        with st.expander("ℹ️ How to interpret these statistics"):
            st.markdown("""
            | Statistic | Formula | Interpretation |
            |---|---|---|
            | **Tetrachoric r** | cos(π/(1+√(ad/bc))) | Latent continuous association between binary variables |
            | **Chi-Square χ²** | Σ(O−E)²/E | Tests independence (p<0.05 → not independent) |
            | **MCC (φ)** | (ad−bc)/√((a+b)(a+c)(b+d)(c+d)) | Balanced binary association [−1,+1] |
            | **Cramér's V** | √(χ²/N(q−1)) | Effect size [0,1]: 0=none, 1=perfect |
 
            **Key finding:** Weak MCC (~0.05–0.10) despite significant χ² confirms that
            emoji usage is statistically associated with sarcasm but the *practical*
            effect is small — justifying the need for deep learning.
            """)
    else:
        st.info("Correlation data not available. Check backend/data/correlation_summary.csv")
 
 
# ════════════════════════════════════════════════════════════
# TAB 3: MODEL INFO
# ════════════════════════════════════════════════════════════
with tab_model:
    st.markdown("### 🧠 Model Architecture & Performance")
 
    # ── Architecture summary ──────────────────────────────────
    col_arch, col_perf = st.columns([1, 1])
 
    with col_arch:
        st.markdown("#### Architecture: CNN-BiLSTM-Attention")
        st.markdown("""
        ```
        Input Text + Emojis
               ↓
        ┌─────────────────────────────────┐
        │  Triple Embedding Fusion        │
        │  GloVe-Twitter (200-dim)        │
        │  Word2Vec (200-dim)             │
        │  Emoji2Vec (200-dim)            │
        │  → Concat: F = W⊕G⊕E (600-dim) │
        └─────────────────────────────────┘
               ↓ SpatialDropout(0.40)
        ┌─────────────────────────────────┐
        │  Multi-Scale CNN                │
        │  k=2,3,4 | 64 filters each     │
        │  → 192-dim | MaxPool(2)         │
        └─────────────────────────────────┘
               ↓
        ┌─────────────────────────────────┐
        │  BiLSTM (64+64 units)           │
        │  return_sequences=True          │
        └─────────────────────────────────┘
               ↓
        ┌─────────────────────────────────┐
        │  Multi-Head Self-Attention      │
        │  4 heads | key_dim=32           │
        │  + Residual + LayerNorm         │
        └─────────────────────────────────┘
               ↓
        Dense(128)→BN→Drop(0.5)
        Dense(64)→BN→Drop(0.4)
        Dense(1) → Sigmoid
               ↓
        Sarcasm Probability y ∈ (0,1)
        Threshold: Youden's J (~0.449)
        ```
        """)
 
    with col_perf:
        st.markdown("#### Training: Stepwise Transfer Learning")
        st.markdown("""
        | Phase | LR | Epochs | Frozen |
        |---|---|---|---|
        | Phase 1 | 1e-3 | 15 | All Embeddings |
        | Phase 2 | 3e-4 | ~14 | All Embeddings |
        | Phase 3 | 5e-5 | ~10 | None |
 
        **Key Regularization:**
        - SpatialDropout1D (0.40) after embedding
        - L2 weight decay (λ=1e-4)
        - Label smoothing (ε=0.10)
        - Class weighting (balanced 39%/61%)
        - EarlyStopping (patience=5, monitor=val_auc)
 
        **Dataset:**
        - Total: 29,377 reviews
        - Sarcastic: 11,448 (39%)
        - Non-Sarcastic: 17,929 (61%)
        - Vocab size: 27,677 tokens
        """)
 
    # ── Performance metrics table ─────────────────────────────
    st.markdown("#### 📈 Model Comparison (Ablation Study)")
    metrics_df = get_metrics()
    if not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Try to plot if numeric columns available
        num_cols = metrics_df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            fig = px.bar(
                metrics_df, barmode="group",
                y=num_cols[:5],
                title="Model Metrics Comparison",
                color_discrete_sequence=[
                    COLORS["positive"], COLORS["sarcastic"],
                    "#6A1B9A", "#E65100", "#00695C"
                ],
            )
            fig.update_layout(
                height=400, plot_bgcolor="white",
                legend=dict(orientation="h", y=1.08),
                margin=dict(l=40, r=40, t=60, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Model metrics not available. Add metrics_matrix.csv to backend/data/")
 
        # Hardcoded fallback metrics
        st.markdown("**Reference results from paper:**")
        fallback = pd.DataFrame([
            {"Model": "A — Word2Vec only",       "Accuracy": 0.7729, "F1": 0.7282, "ROC-AUC": 0.8748, "MCC": 0.5383, "Overfit": "16%"},
            {"Model": "B — GloVe + TL",          "Accuracy": 0.8275, "F1": 0.7907, "ROC-AUC": 0.9159, "MCC": 0.6476, "Overfit": "5%"},
            {"Model": "C — Word2Vec + TL",        "Accuracy": 0.810,  "F1": 0.770,  "ROC-AUC": 0.905,  "MCC": 0.610,  "Overfit": "8%"},
            {"Model": "D — GloVe+W2V+E (Ours) ⭐","Accuracy": 0.845,  "F1": 0.815,  "ROC-AUC": 0.930,  "MCC": 0.670,  "Overfit": "4%"},
            {"Model": "E — BERTweet (SOTA ref.)", "Accuracy": 0.890,  "F1": 0.860,  "ROC-AUC": 0.955,  "MCC": 0.740,  "Overfit": "3%"},
        ])
        st.dataframe(fallback, use_container_width=True, hide_index=True)
 
 
# ════════════════════════════════════════════════════════════
# TAB 4: HISTORY
# ════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("### 📋 Prediction History")
    st.caption("Predictions from this session (max 50 entries)")
 
    history_df = get_history_df()
    if history_df.empty:
        st.info("No predictions yet. Go to the 🎯 Predict tab to analyse some text!")
    else:
        # Quick summary stats
        total    = len(history_df)
        sarc_pct = (history_df["Sarcasm"] == "Sarcastic").sum() / total * 100
        pos_pct  = (history_df["Sentiment"] == "Positive").sum() / total * 100
        bully_pct = (history_df["Bully"] == "Bully").sum() / total * 100
 
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:  render_metric_card("Total Predictions", str(total), COLORS["positive"])
        with sc2:  render_metric_card("Sarcastic %", f"{sarc_pct:.0f}%", COLORS["sarcastic"])
        with sc3:  render_metric_card("Positive Sentiment %", f"{pos_pct:.0f}%", COLORS["positive"])
        with sc4:  render_metric_card("Bully Detected %", f"{bully_pct:.0f}%", COLORS["bully"])
 
        st.markdown("")
        st.dataframe(
            history_df.style.apply(
                lambda row: [
                    f"background-color: {COLORS['sarcastic']}22"
                    if row.get("Sarcasm") == "Sarcastic" else ""
                    for _ in row
                ], axis=1
            ),
            use_container_width = True,
            hide_index          = True,
        )
 
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.history = []
            st.rerun()
 
 
# ════════════════════════════════════════════════════════════
# TAB 5: ABOUT
# ════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### ℹ️ About Sarcoji")
 
    col_a1, col_a2 = st.columns([2, 1])
    with col_a1:
        st.markdown("""
        **Sarcoji** is a research project on sarcasm detection in social media text,
        combining sentiment analysis, emoji interpretation, and statistical correlation
        studies into a unified deep learning framework.
 
        #### 🔬 Research Contributions
        1. **Triple-fusion embedding** (GloVe-Twitter + Word2Vec + Emoji2Vec → 600-dim)
           combining complementary embedding paradigms
        2. **Hybrid architecture** (Multi-scale CNN + BiLSTM + Self-Attention)
           capturing local, sequential, and global sarcasm patterns
        3. **Stepwise Transfer Learning** — 3-phase strategy reducing overfitting
           from 16% (baseline) to 4% (final model)
        4. **Statistical correlation study** using Chi-square, MCC, Cramér's V,
           and tetrachoric correlation to quantify sarcasm-emoji-sentiment associations
        5. **Dataset**: Sarcoji — 29,377 social media reviews
           (11,448 sarcastic, 17,929 non-sarcastic), 1,017 unique emojis
 
        #### 📊 Key Results
        | Metric | Value |
        |---|---|
        | Accuracy | 84.5% |
        | ROC-AUC | 93.0% |
        | F1 Score | 81.5% |
        | MCC | 0.670 |
        | Overfitting Gap | 4% |
 
        #### ⚠️ Limitations
        - Bully detection is rule-based (not deep learning)
        - Attention highlights are approximate (gradient-based proxy)
        - Model trained primarily on English social media text
        """)
 
    with col_a2:
        st.markdown("""
        #### 🛠️ Tech Stack
        - Python 3.10+
        - TensorFlow / Keras
        - FastAPI
        - Streamlit
        - Plotly
        - VADER Sentiment
        - GloVe-Twitter-200
        - Word2Vec (domain)
        - Emoji2Vec
 
        #### 👤 Author
        **Atharv Gupta**
        AI & Robotics, 2nd Year
 
        [GitHub](https://github.com/atharv-0705)
 
        #### 📄 Paper Status
        *Research paper in preparation*
        """)
 
    st.divider()
    st.markdown(
        "<center><small>Sarcoji v1.0.0 · "
        "Built with ❤️ using Streamlit + FastAPI · "
        "Research-grade sentiment analysis</small></center>",
        unsafe_allow_html=True
    )