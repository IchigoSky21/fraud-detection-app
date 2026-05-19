import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ── Load model & scaler ───────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ── Session state ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Fraud Detection App",
    page_icon  = "🔍",
    layout     = "centered"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ══ BACKGROUND ONLY — adaptive ══ */
@media (prefers-color-scheme: light) {
    .stApp {
        background: #dbeafe !important;
        background-image:
            radial-gradient(ellipse 80% 50% at 50% -10%, rgba(59,130,246,0.20) 0%, transparent 70%),
            radial-gradient(ellipse 60% 40% at 85% 85%, rgba(96,165,250,0.14) 0%, transparent 60%) !important;
    }
}
@media (prefers-color-scheme: dark) {
    .stApp {
        background: #0c1a3a !important;
        background-image:
            radial-gradient(ellipse 80% 50% at 50% -10%, rgba(29,78,216,0.22) 0%, transparent 70%),
            radial-gradient(ellipse 60% 40% at 85% 85%, rgba(37,99,235,0.14) 0%, transparent 60%) !important;
    }
}

/* ══ CARDS — fixed light style regardless of theme ══ */
.card {
    background: rgba(255, 255, 255, 0.82) !important;
    border: 1px solid rgba(59,130,246,0.20) !important;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.1rem;
    backdrop-filter: blur(10px);
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #1d4ed8 !important;
    margin-bottom: 0.9rem;
}

/* ══ FORCE LIGHT THEME ON ALL STREAMLIT INPUTS ══ */
/* Number inputs */
div[data-testid="stNumberInput"] input {
    background-color: #ffffff !important;
    color: #1e3a5f !important;
    border: 1px solid rgba(59,130,246,0.25) !important;
    border-radius: 8px !important;
}
div[data-testid="stNumberInput"] label {
    color: #475569 !important;
    font-size: 0.82rem !important;
}
div[data-testid="stNumberInput"] button {
    background-color: #eff6ff !important;
    color: #1d4ed8 !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
}

/* Radio buttons */
div[data-testid="stRadio"] label {
    color: #1e3a5f !important;
    font-size: 0.9rem !important;
}
div[data-testid="stRadio"] > div {
    background: transparent !important;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.7) !important;
    border-radius: 10px !important;
}
div[data-testid="stFileUploader"] label {
    color: #475569 !important;
}

/* Caption text */
div[data-testid="stCaptionContainer"] p {
    color: #64748b !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ══ HERO ══ */
.hero {
    text-align: center;
    padding: 2.2rem 1rem 1.4rem;
    margin-bottom: 1rem;
}
.hero-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.28rem 0.9rem;
    border-radius: 999px;
    border: 1px solid rgba(59,130,246,0.40);
    background: rgba(255,255,255,0.55);
    color: #1d4ed8;
    margin-bottom: 0.9rem;
    backdrop-filter: blur(6px);
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 0.55rem;
    letter-spacing: -0.02em;
    color: #1e3a5f;
}
.hero-sub {
    font-size: 0.92rem;
    font-weight: 300;
    max-width: 400px;
    margin: 0 auto;
    line-height: 1.6;
    color: #475569;
}

/* ══ RESULT BOXES ══ */
.result-fraud {
    background: linear-gradient(135deg, rgba(220,38,38,0.10), rgba(185,28,28,0.05));
    border: 1px solid rgba(220,38,38,0.35);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    margin: 0.9rem 0;
}
.result-legit {
    background: linear-gradient(135deg, rgba(22,163,74,0.10), rgba(15,118,55,0.05));
    border: 1px solid rgba(22,163,74,0.35);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    margin: 0.9rem 0;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: 0.03em;
}
.result-fraud .result-label { color: #dc2626; }
.result-legit .result-label { color: #16a34a; }

/* ══ METRICS ══ */
.metrics-row {
    display: flex;
    gap: 0.75rem;
    margin: 0.9rem 0;
}
.metric-box {
    flex: 1;
    border-radius: 10px;
    border: 1px solid rgba(59,130,246,0.15);
    background: rgba(239,246,255,0.85);
    padding: 0.8rem 0.9rem;
    text-align: center;
}
.metric-label {
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
    color: #64748b;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #1e3a5f;
}
.metric-value.speed { color: #1d4ed8; }
.metric-value.fraud { color: #dc2626; }
.metric-value.legit { color: #16a34a; }

/* ══ PROGRESS BAR ══ */
.prob-bar-wrap  { margin: 0.7rem 0 0.2rem; }
.prob-bar-label { font-size: 0.72rem; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.35rem; color: #64748b; }
.prob-bar-bg    { border-radius: 999px; height: 7px; overflow: hidden; background: rgba(59,130,246,0.12); }
.prob-bar-fill  { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #ef4444, #f97316); }

/* ══ PILLS ══ */
.pill-row { display: flex; gap: 0.5rem; margin-bottom: 0.9rem; flex-wrap: wrap; }
.pill     { padding: 0.28rem 0.8rem; border-radius: 999px; font-size: 0.76rem; font-weight: 600; border: 1px solid; }
.pill-total { background: rgba(59,130,246,0.10); color: #1d4ed8; border-color: rgba(59,130,246,0.3); }
.pill-fraud { background: rgba(220,38,38,0.10);  color: #dc2626; border-color: rgba(220,38,38,0.3); }
.pill-legit { background: rgba(22,163,74,0.10);  color: #16a34a; border-color: rgba(22,163,74,0.3); }

/* ══ HISTORY ══ */
.history-header {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
    color: #1e3a5f;
}
.history-empty {
    font-size: 0.86rem;
    text-align: center;
    padding: 1.4rem;
    border: 1px dashed rgba(59,130,246,0.25);
    border-radius: 10px;
    color: #64748b;
    background: rgba(255,255,255,0.5);
}

/* ══ DIVIDER & BUTTON ══ */
.section-divider { border: none; border-top: 1px solid rgba(59,130,246,0.15); margin: 1.6rem 0; }

.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    padding: 0.6rem 1.1rem !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ══ FOOTER ══ */
.footer {
    text-align: center;
    font-size: 0.73rem;
    padding: 1.8rem 0 0.8rem;
    letter-spacing: 0.03em;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge">Machine Learning · Binary Classification</div>
    <div class="hero-title">Transaction Fraud Detection</div>
    <div class="hero-sub">Detect whether a credit card transaction is fraudulent or legitimate using a trained Random Forest model.</div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MODE SELECTOR
# ════════════════════════════════════════════════════════════
st.markdown('<div class="card"><div class="card-title">Input Mode</div>', unsafe_allow_html=True)
mode = st.radio("", ["Manual Input", "Upload CSV"], horizontal=True, label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MODE 1 — Manual Input
# ════════════════════════════════════════════════════════════
if mode == "Manual Input":
    st.markdown('<div class="card"><div class="card-title">Transaction Features</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    feature_values = {}

    with col1:
        for i in range(1, 15):
            feature_values[f"V{i}"] = st.number_input(
                f"V{i}", value=0.0, format="%.6f", key=f"v{i}"
            )
    with col2:
        for i in range(15, 29):
            feature_values[f"V{i}"] = st.number_input(
                f"V{i}", value=0.0, format="%.6f", key=f"v{i}"
            )

    amount = st.number_input("Amount (transaction value)", value=0.0, min_value=0.0, format="%.2f")
    feature_values["Amount"] = amount
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Predict", use_container_width=True):
        df_input = pd.DataFrame([feature_values])
        df_input["Amount"] = scaler.transform(df_input[["Amount"]])

        start       = time.time()
        prediction  = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]
        latency     = (time.time() - start) * 1000

        if prediction == 1:
            st.markdown("""
            <div class="result-fraud">
                <div class="result-label">FRAUDULENT TRANSACTION</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-legit">
                <div class="result-label">LEGITIMATE TRANSACTION</div>
            </div>""", unsafe_allow_html=True)

        pred_class = "fraud" if prediction == 1 else "legit"
        pred_label = "Fraud" if prediction == 1 else "Legitimate"

        st.markdown(f"""
        <div class="metrics-row">
            <div class="metric-box">
                <div class="metric-label">Prediction</div>
                <div class="metric-value {pred_class}">{pred_label}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Fraud Probability</div>
                <div class="metric-value {pred_class}">{probability:.2%}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Latency</div>
                <div class="metric-value speed">{latency:.1f} ms</div>
            </div>
        </div>
        <div class="prob-bar-wrap">
            <div class="prob-bar-label">Fraud Risk Level</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{probability*100:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        record = {"Result": "Fraud" if prediction == 1 else "Legitimate"}
        record.update({f"V{i}": round(feature_values[f"V{i}"], 4) for i in range(1, 29)})
        record["Amount"] = amount
        st.session_state.history.append(record)

# ════════════════════════════════════════════════════════════
# MODE 2 — Upload CSV
# ════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="card"><div class="card-title">Batch Prediction via CSV</div>', unsafe_allow_html=True)
    st.caption("CSV must contain columns V1 to V28 and Amount (no id or Class column)")
    uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        df_upload     = pd.read_csv(uploaded_file)
        required_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        missing       = [c for c in required_cols if c not in df_upload.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            df_process = df_upload[required_cols].copy()
            df_process["Amount"] = scaler.transform(df_process[["Amount"]])

            start   = time.time()
            preds   = model.predict(df_process)
            probs   = model.predict_proba(df_process)[:, 1]
            latency = (time.time() - start) * 1000

            df_upload["Prediction"]        = ["Fraud" if p == 1 else "Legitimate" for p in preds]
            df_upload["Fraud_Probability"] = [f"{p:.2%}" for p in probs]

            n_fraud = int(sum(preds))
            n_legit = len(preds) - n_fraud

            st.markdown(f"""
            <div class="pill-row">
                <span class="pill pill-total">Total: {len(preds)}</span>
                <span class="pill pill-fraud">Fraud: {n_fraud}</span>
                <span class="pill pill-legit">Legitimate: {n_legit}</span>
                <span class="pill pill-total">Latency: {latency:.1f} ms</span>
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(
                df_upload[["Prediction", "Fraud_Probability"] + required_cols],
                use_container_width=True
            )

            csv_out = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button(
                label               = "Download Results CSV",
                data                = csv_out,
                file_name           = "fraud_prediction_results.csv",
                mime                = "text/csv",
                use_container_width = True
            )

            for pred, row in zip(preds, df_upload[required_cols].itertuples(index=False)):
                record = {"Result": "Fraud" if pred == 1 else "Legitimate"}
                record.update({f"V{i}": round(getattr(row, f"V{i}"), 4) for i in range(1, 29)})
                record["Amount"] = row.Amount
                st.session_state.history.append(record)

# ════════════════════════════════════════════════════════════
# HISTORY
# ════════════════════════════════════════════════════════════
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="history-header">Prediction History &nbsp;<span style="font-size:0.72rem;font-weight:400;opacity:0.5;">(this session only)</span></div>', unsafe_allow_html=True)

if len(st.session_state.history) == 0:
    st.markdown('<div class="history-empty">No predictions yet. Results will appear here after each detection.</div>', unsafe_allow_html=True)
else:
    df_history = pd.DataFrame(st.session_state.history)
    df_history.index = range(1, len(df_history) + 1)
    df_history.index.name = "No"

    total = len(df_history)
    fraud = df_history["Result"].str.contains("Fraud").sum()
    legit = total - fraud

    st.markdown(f"""
    <div class="pill-row">
        <span class="pill pill-total">Total: {total}</span>
        <span class="pill pill-fraud">Fraud: {fraud}</span>
        <span class="pill pill-legit">Legitimate: {legit}</span>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(df_history, use_container_width=True)

    if st.button("Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    Transaction Fraud Detection &nbsp;·&nbsp; Machine Learning Project &nbsp;·&nbsp;
    LK01 Group 6 &nbsp;·&nbsp; Binus University 2026
</div>
""", unsafe_allow_html=True)
