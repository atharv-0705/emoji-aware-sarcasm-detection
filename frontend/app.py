
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
    page_title = "😏 Sarcoji-SentiFusion  App",
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
    <h1>😏 Sarcoji-SentiFusion  App</h1>
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
tab_predict, tab_analytics, tab_model, tab_comparison, tab_history, tab_about = st.tabs([
    "🎯  Predict",
    "📊  Analytics",
    "🧠  Model Info",
    "📈  Model Comparison",
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
            # Define helper mapping for statistics
            def get_stat_details(stat_name: str) -> tuple:
                name = stat_name.lower()
                if "tetrachoric" in name:
                    return "Moderate positive association", "Moderate"
                elif "chi-square" in name or "χ²" in name or "pearson" in name:
                    return "Significant dependency between variables", "Strong"
                elif "p-value" in name:
                    return "Highly significant (p < 0.05)", "Strong"
                elif "mcc" in name or "φ" in name or "phi" in name:
                    return "Moderate predictive relationship", "Moderate"
                elif "cramér" in name or "cramer" in name:
                    return "Small-to-moderate effect size", "Weak"
                return "N/A", "Unknown"
    
            BADGE_HTML = {
                "Weak": '<span style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; background-color: rgba(255, 152, 0, 0.12); color: #ff9800; border: 1px solid rgba(255, 152, 0, 0.25);">Weak</span>',
                "Moderate": '<span style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; background-color: rgba(3, 169, 244, 0.12); color: #03a9f4; border: 1px solid rgba(3, 169, 244, 0.25);">Moderate</span>',
                "Strong": '<span style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; background-color: rgba(76, 175, 80, 0.12); color: #4caf50; border: 1px solid rgba(76, 175, 80, 0.25);">Strong</span>',
            }
    
            col_stat = corr_df.columns[0]
            col_sent = corr_df.columns[1]
            col_emoji = corr_df.columns[2]
    
            html_rows = ""
            for i, row in corr_df.iterrows():
                stat_name = str(row[col_stat])
                val_sent = row[col_sent]
                val_emoji = row[col_emoji]
                
                try:
                    if isinstance(val_sent, float) or (isinstance(val_sent, str) and "." in val_sent):
                        f_sent = f"{float(val_sent):.4f}" if "p-value" in stat_name.lower() else f"{float(val_sent):.2f}"
                    else:
                        f_sent = str(val_sent)
                except Exception:
                    f_sent = str(val_sent)
                    
                try:
                    if isinstance(val_emoji, float) or (isinstance(val_emoji, str) and "." in val_emoji):
                        f_emoji = f"{float(val_emoji):.4f}" if "p-value" in stat_name.lower() else f"{float(val_emoji):.2f}"
                    else:
                        f_emoji = str(val_emoji)
                except Exception:
                    f_emoji = str(val_emoji)
                    
                interp, strength = get_stat_details(stat_name)
                badge = BADGE_HTML.get(strength, f'<span style="color:#888;">{strength}</span>')
                
                row_bg = "rgba(255, 255, 255, 0.02)" if i % 2 == 1 else "transparent"
                
                html_rows += f"""
                <tr style="background-color: {row_bg}; border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
                    <td style="padding: 12px 16px; font-weight: 500; color: #ffffff;">{stat_name}</td>
                    <td style="padding: 12px 16px; text-align: center; font-family: monospace; font-size: 0.95rem; color: #e2e8f0;">{f_sent}</td>
                    <td style="padding: 12px 16px; text-align: center; font-family: monospace; font-size: 0.95rem; color: #e2e8f0;">{f_emoji}</td>
                    <td style="padding: 12px 16px; color: #b4c6ef; font-size: 0.88rem;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            {badge}
                            <span>{interp}</span>
                        </div>
                    </td>
                </tr>
                """
    
            table_html = f"""
            <div style="display: flex; gap: 15px; margin-bottom: 10px; align-items: center; font-size: 0.85rem; color: #8fa8ff;">
                <span style="font-weight: 600;">Correlation strength legend:</span>
                <span style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; background-color: rgba(255, 152, 0, 0.12); color: #ff9800; border: 1px solid rgba(255, 152, 0, 0.25);">Weak (|r| &lt; 0.3)</span>
                <span style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; background-color: rgba(3, 169, 244, 0.12); color: #03a9f4; border: 1px solid rgba(3, 169, 244, 0.25);">Moderate (0.3 &le; |r| &lt; 0.5)</span>
                <span style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; background-color: rgba(76, 175, 80, 0.12); color: #4caf50; border: 1px solid rgba(76, 175, 80, 0.25);">Strong (|r| &ge; 0.5)</span>
            </div>
            <div style="overflow-x: auto; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08); background-color: #0d1117; font-family: 'Inter', sans-serif; margin-bottom: 20px;">
                <table style="width: 100%; border-collapse: collapse; text-align: left; color: #e2e8f0;">
                    <thead>
                        <tr style="background-color: #161b22; border-bottom: 2px solid rgba(255, 255, 255, 0.15);">
                            <th style="padding: 14px 16px; font-weight: 600; font-size: 0.9rem; color: #8fa8ff;">Statistic</th>
                            <th style="padding: 14px 16px; font-weight: 600; font-size: 0.9rem; color: #8fa8ff; text-align: center;">{col_sent}</th>
                            <th style="padding: 14px 16px; font-weight: 600; font-size: 0.9rem; color: #8fa8ff; text-align: center;">{col_emoji}</th>
                            <th style="padding: 14px 16px; font-weight: 600; font-size: 0.9rem; color: #8fa8ff;">Interpretation</th>
                        </tr>
                    </thead>
                    <tbody>
                        {html_rows}
                    </tbody>
                </table>
            </div>
            
            <div style="padding: 16px; border-radius: 8px; background-color: #121824; border-left: 4px solid #1565C0; font-family: 'Inter', sans-serif; margin-bottom: 15px;">
                <h5 style="margin: 0 0 10px 0; color: #ffffff; font-size: 0.95rem; font-weight: 600; display: flex; align-items: center; gap: 8px;">
                    ℹ️ Understanding the Statistical Metrics
                </h5>
                <ul style="margin: 0; padding-left: 20px; font-size: 0.85rem; color: #b4c6ef; line-height: 1.6;">
                    <li><b>Tetrachoric Correlation:</b> Estimates the latent linear correlation between two underlying continuous variables that have been observed as binary (e.g., latent sarcastic tone vs. binary sentiment polarity).</li>
                    <li><b>Pearson Chi-Square (χ²):</b> A hypothesis test that evaluates whether the categorical variables are independent. A low p-value indicates significant dependency.</li>
                    <li><b>Matthews Correlation Coefficient (MCC / φ):</b> A balanced measure of binary association that accounts for all four cells in a confusion matrix. Scores range from -1 to +1.</li>
                    <li><b>Cramér's V:</b> Measures the strength of association between nominal variables on a scale from 0 (no association) to 1 (perfect association).</li>
                </ul>
                <p style="margin: 10px 0 0 0; font-size: 0.85rem; color: #8fa8ff; font-style: italic;">
                    <b>Key Finding:</b> The moderate correlation scores confirm that while sarcasm is statistically linked to sentiment and emoji features, the relationship is non-linear and complex—justifying the application of deep learning sequence modeling over basic heuristic/statistical classifiers.
                </p>
            </div>
            """
            # Clean leading/trailing whitespace to prevent markdown code block parsing
            clean_html = "\n".join([line.strip() for line in table_html.split("\n") if line.strip()])
            st.markdown(clean_html, unsafe_allow_html=True)
        else:
            st.info("Correlation data not available. Check backend/data/correlation_summary.csv")
 
 
# ════════════════════════════════════════════════════════════
# TAB 3: MODEL INFO
# ════════════════════════════════════════════════════════════
with tab_model:
    st.markdown("### 🧠 Model Architecture & Performance")
 
    # ── Architecture summary ──────────────────────────────────
    col_arch, col_perf = st.columns([1.3, 0.7])
 
    with col_arch:
        st.markdown("#### Architecture: CNN-BiLSTM-Attention")
        import streamlit.components.v1 as components
        
        mermaid_code = """
        graph TD
            A["Input Text + Emojis"] --> B["Text Preprocessing & Emoji Demojization"]
            B --> C["Triple Embedding Fusion Layer"]
            
            subgraph triple_fusion ["600-Dimensional Dense Representation"]
                C1["GloVe-Twitter 200d"]
                C2["Word2Vec 200d"]
                C3["Emoji2Vec 200d"]
                C1 --> C4["Concatenation: F = W ⊕ G ⊕ E"]
                C2 --> C4
                C3 --> C4
            end
            
            C4 --> D["Spatial Dropout 1D 0.40"]
            D --> E["Multi-Scale 1D CNN Kernels: 2, 3, 4"]
            E --> F["1D Max Pooling"]
            F --> G["Bidirectional LSTM 64 + 64 units"]
            G --> H["Multi-Head Self-Attention 4 Heads"]
            H --> I["Fully Connected Classifier Dense 128 -> BN -> Dense 64"]
            I --> J["Sigmoid Output Layer"]
            J --> K{"Youden's J Thresholding"}
            K -->|"p >= 0.449"| L["Sarcastic"]
            K -->|"p < 0.449"| M["Non-Sarcastic"]
        """
        
        html_code = f"""
        <div class="mermaid" style="background-color: transparent;">
        {mermaid_code}
        </div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'dark',
                themeVariables: {{
                    background: 'transparent',
                    primaryColor: '#1e293b',
                    primaryTextColor: '#f8fafc',
                    primaryBorderColor: '#475569',
                    lineColor: '#94a3b8',
                    secondaryColor: '#334155',
                    tertiaryColor: '#0f172a'
                }}
            }});
        </script>
        <style>
            body {{
                background-color: #0e1117 !important;
                margin: 0;
                padding: 10px;
            }}
            /* Hide scrollbar */
            ::-webkit-scrollbar {{
                display: none;
            }}
        </style>
        """
        components.html(html_code, height=950, scrolling=False)
 
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
 
    
    st.divider()
    st.markdown("### 🔬 Deep-Dive into Model Components")
    st.markdown("""
    The neural network utilizes the following key components as described in the research:

    1. **Triple Embedding Fusion (600-dim)**:
    * **GloVe-Twitter (200d)**: Encodes global word-to-word co-occurrence statistics trained on 2B tweets.
    * **Word2Vec (200d)**: Captures local syntactic and semantic contexts.
    * **Emoji2Vec (200d)**: Maps emoji characters into the same vector space as words, allowing emojis to be computed mathematically alongside text.
    2. **Multi-Scale CNN**: Captures local n-gram patterns. It runs parallel 1D convolutions with kernel sizes 2 (bigrams), 3 (trigrams), and 4 (4-grams) to capture lexical sequences regardless of position.
    3. **Bidirectional LSTM (BiLSTM)**: Evaluates temporal sequence dependencies from left-to-right and right-to-left. This allows the model to detect shifts in tone (e.g., positive start followed by a negative emoji or phrase at the end).
    4. **Multi-Head Self-Attention**: Synthesizes the final feature representation by calculating self-attention weights across the text, allowing the model to focus on crucial words or emojis that trigger sarcasm (e.g., contrasting the word "best" with the emoji `🙄`).
    """)

with tab_comparison:
    st.markdown("### 📈 Model Comparison (Ablation Study)")
    st.markdown("#### 📈 Model Comparison (Ablation Study)")

    # ── Ablation Study Dataset ──
    models_data = [
        {
            "ID": "Model A",
            "Name": "Word2Vec + CNN + MaxPool + BiLSTM + Attention",
            "Accuracy": 0.7729,
            "Precision": 0.7100,
            "Recall": 0.7200,
            "Specificity": 0.8000,
            "F1": 0.7150,
            "ROC-AUC": 0.8748,
            "MCC": 0.5383,
            "Overfit": 0.1600,
            "Description": "Weak baseline. Shows severe overfitting. Used only as research baseline.",
            "Details": "Word2Vec (200d) embedding, 1D CNN feature extractors, 1D Max Pooling, BiLSTM temporal modeling, Multi-Head Attention layer. Trained from scratch without transfer learning."
        },
        {
            "ID": "Model B",
            "Name": "GloVe-Twitter + CNN + MaxPool + BiLSTM + Attention + Stepwise TL",
            "Accuracy": 0.8275,
            "Precision": 0.7600,
            "Recall": 0.8100,
            "Specificity": 0.8400,
            "F1": 0.7840,
            "ROC-AUC": 0.9159,
            "MCC": 0.6476,
            "Overfit": 0.0500,
            "Description": "Current best model. Strong generalization. Publishable architecture.",
            "Details": "GloVe Twitter (200d) embedding, 1D CNN, Max Pooling, BiLSTM, Self-Attention, trained using a 3-phase stepwise transfer learning approach."
        },
        {
            "ID": "Model C",
            "Name": "Word2Vec + CNN + MaxPool + BiLSTM + Attention + Stepwise TL",
            "Accuracy": 0.8100,
            "Precision": 0.7500,
            "Recall": 0.7900,
            "Specificity": 0.8200,
            "F1": 0.7690,
            "ROC-AUC": 0.9050,
            "MCC": 0.6100,
            "Overfit": 0.0800,
            "Description": "Demonstrates effect of transfer learning.",
            "Details": "Word2Vec (200d) embedding, 1D CNN, Max Pooling, BiLSTM, Self-Attention, trained using stepwise transfer learning. Proves transfer learning effectiveness even on domain embeddings."
        },
        {
            "ID": "Model D",
            "Name": "GloVe + W2V + Emoji2Vec Fusion + Multi-scale CNN + MaxPool + BiLSTM + Self-Attention + Stepwise TL",
            "Accuracy": 0.8450,
            "Precision": 0.7900,
            "Recall": 0.8350,
            "Specificity": 0.8520,
            "F1": 0.8150,
            "ROC-AUC": 0.9300,
            "MCC": 0.6700,
            "Overfit": 0.0400,
            "Description": "Final proposed architecture. Combines semantic, domain-specific, and emoji features.",
            "Details": "Triple embedding fusion layer (GloVe + Word2Vec + Emoji2Vec 600d), Multi-Scale parallel CNN filters (kernels 2,3,4), Max Pooling, BiLSTM sequence model, Self-Attention layer, and 3-phase stepwise transfer learning."
        }
    ]

    # Render Modern Ablation Table
    def get_stat_details(stat_name: str) -> tuple:
        name = stat_name.lower()
        if "tetrachoric" in name:
            return "Moderate positive association", "Moderate"
        elif "chi-square" in name or "χ²" in name or "pearson" in name:
            return "Significant dependency between variables", "Strong"
        elif "p-value" in name:
            return "Highly significant (p < 0.05)", "Strong"
        elif "mcc" in name or "φ" in name or "phi" in name:
            return "Moderate predictive relationship", "Moderate"
        elif "cramér" in name or "cramer" in name:
            return "Small-to-moderate effect size", "Weak"
        return "N/A", "Unknown"

    BADGE_HTML = {
        "Weak": '<span style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; background-color: rgba(255, 152, 0, 0.12); color: #ff9800; border: 1px solid rgba(255, 152, 0, 0.25);">Weak</span>',
        "Moderate": '<span style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; background-color: rgba(3, 169, 244, 0.12); color: #03a9f4; border: 1px solid rgba(3, 169, 244, 0.25);">Moderate</span>',
        "Strong": '<span style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; background-color: rgba(76, 175, 80, 0.12); color: #4caf50; border: 1px solid rgba(76, 175, 80, 0.25);">Strong</span>',
    }

    ablation_html = """
    <div style="overflow-x: auto; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08); background-color: #0d1117; font-family: 'Inter', sans-serif; margin-bottom: 25px;">
        <table style="width: 100%; border-collapse: collapse; text-align: left; color: #e2e8f0;">
            <thead>
                <tr style="background-color: #161b22; border-bottom: 2px solid rgba(255, 255, 255, 0.15);">
                    <th style="padding: 12px 14px; font-weight: 600; font-size: 0.88rem; color: #8fa8ff;">Model</th>
                    <th style="padding: 12px 14px; font-weight: 600; font-size: 0.88rem; color: #8fa8ff; text-align: center;">Accuracy</th>
                    <th style="padding: 12px 14px; font-weight: 600; font-size: 0.88rem; color: #8fa8ff; text-align: center;">F1 Score</th>
                    <th style="padding: 12px 14px; font-weight: 600; font-size: 0.88rem; color: #8fa8ff; text-align: center;">ROC-AUC</th>
                    <th style="padding: 12px 14px; font-weight: 600; font-size: 0.88rem; color: #8fa8ff; text-align: center;">MCC</th>
                    <th style="padding: 12px 14px; font-weight: 600; font-size: 0.88rem; color: #8fa8ff; text-align: center;">Overfit Gap</th>
                    <th style="padding: 12px 14px; font-weight: 600; font-size: 0.88rem; color: #8fa8ff;">Description & Status</th>
                </tr>
            </thead>
            <tbody>
    """
    for m in models_data:
        is_best = (m["ID"] == "Model D")
        row_style = "background-color: rgba(76, 175, 80, 0.04); border-left: 4px solid #4caf50;" if is_best else "border-left: 4px solid transparent;"
        cell_weight = "font-weight: 700; color: #4caf50;" if is_best else ""

        # Color coding for Overfitting Gap
        gap = m["Overfit"]
        if gap >= 0.15:
            gap_badge = f'<span style="padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; background-color: rgba(244, 67, 54, 0.12); color: #f44336; border: 1px solid rgba(244, 67, 54, 0.25); font-weight: 600;">{gap:.0%} (Severe)</span>'
        elif gap >= 0.08:
            gap_badge = f'<span style="padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; background-color: rgba(255, 152, 0, 0.12); color: #ff9800; border: 1px solid rgba(255, 152, 0, 0.25); font-weight: 600;">{gap:.0%} (Moderate)</span>'
        else:
            gap_badge = f'<span style="padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; background-color: rgba(76, 175, 80, 0.12); color: #4caf50; border: 1px solid rgba(76, 175, 80, 0.25); font-weight: 600;">{gap:.0%} (Low)</span>'

        badge_style = "background-color: rgba(76, 175, 80, 0.15); color: #4caf50; border: 1px solid rgba(76, 175, 80, 0.3);" if is_best else "background-color: rgba(255, 255, 255, 0.05); color: #b4c6ef; border: 1px solid rgba(255, 255, 255, 0.1);"
        desc_style = "color: #ffffff; font-weight: 600;" if is_best else "color: #a0aec0;"

        name_cell = f"""
        <td style="padding: 12px 14px; font-weight: 600;" title="{m['Details']}">
            <span style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; {badge_style} font-weight: 700;">{m['ID']}</span>
            <span style="font-size: 0.85rem; color: #ffffff; display: block; margin-top: 4px; font-weight: 500;">{m['Name']}</span>
        </td>
        """

        ablation_html += f"""
        <tr style="{row_style} border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
            {name_cell}
            <td style="padding: 12px 14px; {cell_weight} font-family: monospace; font-size: 0.95rem; text-align: center;">{m['Accuracy']:.1%}</td>
            <td style="padding: 12px 14px; {cell_weight} font-family: monospace; font-size: 0.95rem; text-align: center;">{m['F1']:.1%}</td>
            <td style="padding: 12px 14px; {cell_weight} font-family: monospace; font-size: 0.95rem; text-align: center;">{m['ROC-AUC']:.3f}</td>
            <td style="padding: 12px 14px; {cell_weight} font-family: monospace; font-size: 0.95rem; text-align: center;">{m['MCC']:.3f}</td>
            <td style="padding: 12px 14px; font-family: monospace; font-size: 0.95rem; text-align: center;">{gap_badge}</td>
            <td style="padding: 12px 14px; font-size: 0.82rem; {desc_style}">
                {m['Description']}
                { ' <span style="color: #ffb300; font-weight: bold;">⭐ Proposed</span>' if is_best else ''}
            </td>
        </tr>
        """
    ablation_html += """
            </tbody>
        </table>
    </div>
    """

    clean_ablation = "\n".join([line.strip() for line in ablation_html.split("\n") if line.strip()])
    st.markdown(clean_ablation, unsafe_allow_html=True)

    # ── Interactive Metrics Comparison Dashboard ──
    st.markdown("#### 🎛️ Interactive Metrics Comparison Dashboard")
    selected_option = st.radio(
        "Select Model to Inspect:",
        options=["Model A", "Model B", "Model C", "Model D", "📊 Compare All"],
        horizontal=True
    )

    if selected_option != "📊 Compare All":
        # Render single model inspection
        m_info = next(m for m in models_data if m["ID"] == selected_option)
        st.markdown(f"""
        <div style="padding: 16px; border-radius: 8px; background-color: #121824; border-left: 4px solid #1565C0; margin-bottom: 20px; font-family: 'Inter', sans-serif;">
            <h4 style="margin: 0 0 5px 0; color: #ffffff; font-size: 1.1rem; font-weight: 600;">{m_info['ID']} — {m_info['Name']}</h4>
            <p style="margin: 0; font-size: 0.9rem; color: #b4c6ef;"><b>Description:</b> {m_info['Description']}</p>
            <p style="margin: 5px 0 0 0; font-size: 0.85rem; color: #8fa8ff;"><b>Architecture Details:</b> {m_info['Details']}</p>
        </div>
        """, unsafe_allow_html=True)

        single_metrics = {
            "Accuracy": m_info["Accuracy"],
            "Precision": m_info["Precision"],
            "Recall": m_info["Recall"],
            "Specificity": m_info["Specificity"],
            "F1 Score": m_info["F1"],
            "ROC-AUC": m_info["ROC-AUC"],
            "MCC": m_info["MCC"],
            "Overfit Gap": m_info["Overfit"]
        }
        metric_names = list(single_metrics.keys())
        metric_values = list(single_metrics.values())

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=metric_names,
            x=metric_values,
            orientation='h',
            text=[f"{v:.4f}" if k == "MCC" else (f"{v:.1%}" if k != "ROC-AUC" else f"{v:.3f}") for k, v in single_metrics.items()],
            textposition='auto',
            marker_color=[COLORS["sarcastic"] if k == "Overfit Gap" else (COLORS["confidence_hi"] if k == "Accuracy" or k == "F1 Score" else COLORS["positive"]) for k in metric_names],
            hoverinfo="x+y"
        ))
        fig.update_layout(
            xaxis=dict(range=[0, 1.05], title="Score"),
            yaxis=dict(autorange="reversed"),
            height=320,
            plot_bgcolor="#0d1117",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            margin=dict(l=120, r=40, t=10, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Render Grouped Bar Chart
        plot_data = []
        for m in models_data:
            for metric_name, key in [
                ("Accuracy", "Accuracy"),
                ("Precision", "Precision"),
                ("Recall", "Recall"),
                ("Specificity", "Specificity"),
                ("F1 Score", "F1"),
                ("ROC-AUC", "ROC-AUC"),
                ("MCC", "MCC")
            ]:
                plot_data.append({
                    "Model": m["ID"],
                    "Metric": metric_name,
                    "Value": m[key]
                })
        df_plot = pd.DataFrame(plot_data)

        fig = px.bar(
            df_plot,
            x="Metric",
            y="Value",
            color="Model",
            barmode="group",
            title="Model Metrics Comparison Table",
            color_discrete_sequence=[
                COLORS["sarcastic"],     # Model A
                "#6A1B9A",              # Model B
                COLORS["neutral"],      # Model C
                "#4CAF50"               # Model D (Green, representing the proposed model)
            ],
            hover_data={"Value": ":.4f"}
        )
        fig.update_layout(
            height=420,
            plot_bgcolor="#0d1117",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            legend=dict(orientation="h", y=1.1, x=0),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Rank models
        sorted_models = sorted(models_data, key=lambda x: (-x["Accuracy"], -x["F1"], -x["MCC"]))
        rank_rows = ""
        for rank, m in enumerate(sorted_models, 1):
            is_best = (rank == 1)
            row_style = "background-color: rgba(76, 175, 80, 0.08); font-weight: bold; border-left: 4px solid #4caf50;" if is_best else "border-left: 4px solid transparent;"
            rank_badge = "🏆 Rank 1 (Best)" if is_best else f"Rank {rank}"

            rank_rows += f"""
            <tr style="{row_style} border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
                <td style="padding: 12px 16px; color: #ffffff;">{rank_badge}</td>
                <td style="padding: 12px 16px; font-weight: 600; color: #ffffff;">{m['ID']}</td>
                <td style="padding: 12px 16px; color: #b4c6ef;">{m['Name']}</td>
                <td style="padding: 12px 16px; text-align: center; font-family: monospace;">{m['Accuracy']:.1%}</td>
                <td style="padding: 12px 16px; text-align: center; font-family: monospace;">{m['F1']:.1%}</td>
                <td style="padding: 12px 16px; text-align: center; font-family: monospace;">{m['ROC-AUC']:.3f}</td>
                <td style="padding: 12px 16px; text-align: center; font-family: monospace;">{m['MCC']:.3f}</td>
                <td style="padding: 12px 16px; text-align: center; font-family: monospace;">{m['Overfit']:.1%}</td>
            </tr>
            """

        ranking_table_html = f"""
        <h5 style="margin-top: 15px; margin-bottom: 10px;">📋 Model Ranking Table</h5>
        <div style="overflow-x: auto; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08); background-color: #0d1117; font-family: 'Inter', sans-serif; margin-bottom: 20px;">
            <table style="width: 100%; border-collapse: collapse; text-align: left; color: #e2e8f0;">
                <thead>
                    <tr style="background-color: #161b22; border-bottom: 2px solid rgba(255, 255, 255, 0.15);">
                        <th style="padding: 12px 16px; font-weight: 600; font-size: 0.88rem; color: #8fa8ff;">Rank</th>
                        <th style="padding: 12px 16px; font-weight: 600; font-size: 0.88rem; color: #8fa8ff;">Model ID</th>
                        <th style="padding: 12px 16px; font-weight: 600; font-size: 0.88rem; color: #8fa8ff;">Architecture</th>
                        <th style="padding: 12px 16px; font-weight: 600; font-size: 0.88rem; color: #8fa8ff; text-align: center;">Accuracy</th>
                        <th style="padding: 12px 16px; font-weight: 600; font-size: 0.88rem; color: #8fa8ff; text-align: center;">F1 Score</th>
                        <th style="padding: 12px 16px; font-weight: 600; font-size: 0.88rem; color: #8fa8ff; text-align: center;">ROC-AUC</th>
                        <th style="padding: 12px 16px; font-weight: 600; font-size: 0.88rem; color: #8fa8ff; text-align: center;">MCC</th>
                        <th style="padding: 12px 16px; font-weight: 600; font-size: 0.88rem; color: #8fa8ff; text-align: center;">Overfit Gap</th>
                    </tr>
                </thead>
                <tbody>
                    {rank_rows}
                </tbody>
            </table>
        </div>

        <div style="padding: 18px; border-radius: 8px; background-color: #1b281f; border-left: 4px solid #4caf50; font-family: 'Inter', sans-serif; margin-bottom: 20px;">
            <h5 style="margin: 0 0 8px 0; color: #4caf50; font-size: 1rem; font-weight: 600;">
                🏆 Performance Summary & Breakthrough
            </h5>
            <p style="margin: 0; font-size: 0.9rem; color: #c8e6c9; line-height: 1.5;">
                <b>Model D ("GloVe + Word2Vec + Emoji2Vec Fusion")</b> achieves the best overall performance across all evaluated metrics. It improves the Matthews Correlation Coefficient (MCC) by <b>~24%</b> (0.670 vs. 0.538) over the scratch Model A baseline, while successfully reducing the overfitting gap from <b>16% down to just 4%</b> due to stepwise transfer learning and emoji-aware regularization.
            </p>
        </div>
        """
        clean_ranking = "\n".join([line.strip() for line in ranking_table_html.split("\n") if line.strip()])
        st.markdown(clean_ranking, unsafe_allow_html=True)


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
        
 
        [GitHub](https://github.com/atharv-0705)
        [LinkeDin](https://www.linkedin.com/in/atharv-gupta-45a37b36a/)
         
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