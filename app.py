import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ── Load model & scaler ──────────────────────────────────────
# ── Load model & scaler ───────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("fraud_model.pkl")
@@ -13,123 +13,452 @@

model, scaler = load_model()

# ── Konfigurasi halaman ──────────────────────────────────────
# ── Session state ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Fraud Detection App",
    page_icon  = "🔍",
    page_title = "FraudShield — Fraud Detection",
    page_icon  = "🛡️",
    layout     = "centered"
)

st.title("🔍 Transaction Fraud Detection")
st.markdown("Masukkan data transaksi untuk mendeteksi apakah transaksi tersebut **fraudulent** atau **legitimate**.")
# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0e1a;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(56,139,253,0.15) 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(99,60,180,0.10) 0%, transparent 60%);
    color: #e2e8f0;
}

/* ── Hero banner ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    margin-bottom: 1rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(56,139,253,0.15);
    border: 1px solid rgba(56,139,253,0.35);
    color: #58a6ff;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 1rem;
    border-radius: 999px;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.15;
    color: #f0f6ff;
    margin-bottom: 0.6rem;
    letter-spacing: -0.02em;
}
.hero-title span {
    background: linear-gradient(90deg, #58a6ff, #a371f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: #8b949e;
    font-size: 0.95rem;
    font-weight: 300;
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Cards ── */
.card {
    background: rgba(22, 27, 42, 0.85);
    border: 1px solid rgba(56,139,253,0.12);
    border-radius: 16px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(12px);
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #58a6ff;
    margin-bottom: 1rem;
}

/* ── Result boxes ── */
.result-fraud {
    background: linear-gradient(135deg, rgba(248,81,73,0.15), rgba(200,40,40,0.08));
    border: 1px solid rgba(248,81,73,0.40);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    margin: 1rem 0;
}
.result-legit {
    background: linear-gradient(135deg, rgba(63,185,80,0.15), rgba(40,160,60,0.08));
    border: 1px solid rgba(63,185,80,0.40);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    margin: 1rem 0;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: 0.04em;
}
.result-fraud .result-label { color: #f85149; }
.result-legit .result-label { color: #3fb950; }
.result-icon { font-size: 2.2rem; margin-bottom: 0.3rem; }

/* ── Metrics row ── */
.metrics-row {
    display: flex;
    gap: 0.8rem;
    margin: 1rem 0;
}
.metric-box {
    flex: 1;
    background: rgba(30, 37, 56, 0.9);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-label {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #f0f6ff;
}
.metric-value.fraud  { color: #f85149; }
.metric-value.legit  { color: #3fb950; }
.metric-value.speed  { color: #58a6ff; }

/* ── Progress bar ── */
.prob-bar-wrap {
    margin: 0.8rem 0 0.3rem;
}
.prob-bar-label {
    font-size: 0.75rem;
    color: #8b949e;
    margin-bottom: 0.4rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.prob-bar-bg {
    background: rgba(255,255,255,0.07);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #f85149, #ff9500);
    transition: width 0.6s ease;
}

/* ── History ── */
.history-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #f0f6ff;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.history-empty {
    color: #8b949e;
    font-size: 0.88rem;
    text-align: center;
    padding: 1.5rem;
    border: 1px dashed rgba(255,255,255,0.1);
    border-radius: 12px;
}

# ── Pilih mode input ─────────────────────────────────────────
mode = st.radio("Pilih mode input:", ["Manual Input", "Upload CSV"])
/* ── Summary pills ── */
.pill-row {
    display: flex;
    gap: 0.6rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}
.pill {
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}
.pill-total  { background: rgba(88,166,255,0.15); color: #58a6ff; border: 1px solid rgba(88,166,255,0.3); }
.pill-fraud  { background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid rgba(248,81,73,0.3); }
.pill-legit  { background: rgba(63,185,80,0.15);  color: #3fb950; border: 1px solid rgba(63,185,80,0.3); }

/* ── Section divider ── */
.section-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 1.8rem 0;
}

/* ── Streamlit overrides ── */
.stRadio > label { color: #8b949e !important; font-size: 0.85rem !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: #c9d1d9 !important; }
div[data-baseweb="tab-list"] { background: rgba(22,27,42,0.8) !important; border-radius: 10px !important; }
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1.2rem !important;
    letter-spacing: 0.03em !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stNumberInput > label  { color: #8b949e !important; font-size: 0.8rem !important; }
.stFileUploader > label { color: #8b949e !important; }
div[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden !important; }
.stAlert { border-radius: 10px !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: #484f58;
    font-size: 0.75rem;
    padding: 2rem 0 1rem;
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge">🛡️ ML-Powered Security</div>
    <div class="hero-title">Fraud<span>Shield</span></div>
    <div class="hero-sub">Real-time credit card fraud detection powered by Random Forest machine learning.</div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MODE SELECTOR
# ════════════════════════════════════════════════════════════
st.markdown('<div class="card"><div class="card-title">⚙️ Input Mode</div>', unsafe_allow_html=True)
mode = st.radio("", ["✍️  Manual Input", "📂  Upload CSV"], horizontal=True, label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MODE 1 — Manual Input
# ════════════════════════════════════════════════════════════
if mode == "Manual Input":
    st.subheader("Input Fitur Transaksi")
if "Manual" in mode:
    st.markdown('<div class="card"><div class="card-title">📝 Transaction Features</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    feature_values = {}

    # V1 - V14 di kolom kiri
    with col1:
        for i in range(1, 15):
            feature_values[f"V{i}"] = st.number_input(
                f"V{i}", value=0.0, format="%.6f", key=f"v{i}"
            )

    # V15 - V28 di kolom kanan
    with col2:
        for i in range(15, 29):
            feature_values[f"V{i}"] = st.number_input(
                f"V{i}", value=0.0, format="%.6f", key=f"v{i}"
            )

    amount = st.number_input("Amount (nilai transaksi)", value=0.0, min_value=0.0, format="%.2f")
    amount = st.number_input("💰 Amount (transaction value)", value=0.0, min_value=0.0, format="%.2f")
    feature_values["Amount"] = amount
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔎 Prediksi", use_container_width=True):
        # Preprocessing
    if st.button("🔎  Run Fraud Detection", use_container_width=True):
        df_input = pd.DataFrame([feature_values])
        df_input["Amount"] = scaler.transform(df_input[["Amount"]])

        # Inference + ukur latency
        start = time.time()
        prediction = model.predict(df_input)[0]
        start       = time.time()
        prediction  = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]
        latency = (time.time() - start) * 1000  # ms
        latency     = (time.time() - start) * 1000

        # Output
        st.divider()
        # Result box
        if prediction == 1:
            st.error("🚨 FRAUDULENT TRANSACTION")
            st.markdown(f"""
            <div class="result-fraud">
                <div class="result-icon">🚨</div>
                <div class="result-label">FRAUDULENT TRANSACTION</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.success("✅ LEGITIMATE TRANSACTION")
            st.markdown(f"""
            <div class="result-legit">
                <div class="result-icon">✅</div>
                <div class="result-label">LEGITIMATE TRANSACTION</div>
            </div>""", unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Prediksi",          "Fraud" if prediction == 1 else "Legitimate")
        col_b.metric("Fraud Probability", f"{probability:.2%}")
        col_c.metric("Latency",           f"{latency:.1f} ms")
        # Metrics
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

        # Progress bar probabilitas
        st.markdown("**Fraud Probability:**")
        st.progress(float(probability))
        # Save to history
        record = {"Hasil": "🚨 Fraud" if prediction == 1 else "✅ Legitimate"}
        record.update({f"V{i}": round(feature_values[f"V{i}"], 4) for i in range(1, 29)})
        record["Amount"] = amount
        st.session_state.history.append(record)

# ════════════════════════════════════════════════════════════
# MODE 2 — Upload CSV
# ════════════════════════════════════════════════════════════
else:
    st.subheader("Upload File CSV")
    st.markdown("CSV harus memiliki kolom: `V1` hingga `V28` dan `Amount` (tanpa kolom `id` dan `Class`).")
    st.markdown('<div class="card"><div class="card-title">📂 Batch Prediction via CSV</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e;font-size:0.85rem;">CSV must contain columns: <code>V1</code> to <code>V28</code> and <code>Amount</code> (no <code>id</code> or <code>Class</code> column)</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)

        # Validasi kolom
        df_upload     = pd.read_csv(uploaded_file)
        required_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        missing = [c for c in required_cols if c not in df_upload.columns]
        missing       = [c for c in required_cols if c not in df_upload.columns]

        if missing:
            st.error(f"Kolom berikut tidak ditemukan: {missing}")
            st.error(f"Missing columns: {missing}")
        else:
            df_process = df_upload[required_cols].copy()
            df_process["Amount"] = scaler.transform(df_process[["Amount"]])

            start = time.time()
            preds = model.predict(df_process)
            probs = model.predict_proba(df_process)[:, 1]
            start   = time.time()
            preds   = model.predict(df_process)
            probs   = model.predict_proba(df_process)[:, 1]
            latency = (time.time() - start) * 1000

            df_upload["Prediction"]        = ["Fraud" if p == 1 else "Legitimate" for p in preds]
            df_upload["Prediction"]        = ["🚨 Fraud" if p == 1 else "✅ Legitimate" for p in preds]
            df_upload["Fraud_Probability"] = [f"{p:.2%}" for p in probs]

            st.success(f"✅ Prediksi selesai untuk {len(df_upload)} transaksi dalam {latency:.1f} ms")

            # Ringkasan
            n_fraud = sum(preds)
            n_fraud = int(sum(preds))
            n_legit = len(preds) - n_fraud
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Transaksi", len(preds))
            c2.metric("🚨 Fraud",        n_fraud)
            c3.metric("✅ Legitimate",   n_legit)

            st.dataframe(df_upload[["Prediction", "Fraud_Probability"] + required_cols])
            st.markdown(f"""
            <div class="pill-row">
                <span class="pill pill-total">📊 Total: {len(preds)}</span>
                <span class="pill pill-fraud">🚨 Fraud: {n_fraud}</span>
                <span class="pill pill-legit">✅ Legitimate: {n_legit}</span>
                <span class="pill pill-total">⚡ {latency:.1f} ms</span>
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(
                df_upload[["Prediction", "Fraud_Probability"] + required_cols],
                use_container_width=True
            )

            # Download hasil
            csv_out = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button(
                label     = "⬇️ Download Hasil Prediksi",
                label     = "⬇️  Download Results CSV",
                data      = csv_out,
                file_name = "fraud_prediction_results.csv",
                mime      = "text/csv"
                mime      = "text/csv",
                use_container_width=True
            )

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.caption("Transaction Fraud Detection")
            # Save to history
            for pred, row in zip(preds, df_upload[required_cols].itertuples(index=False)):
                record = {"Hasil": "🚨 Fraud" if pred == 1 else "✅ Legitimate"}
                record.update({f"V{i}": round(getattr(row, f"V{i}"), 4) for i in range(1, 29)})
                record["Amount"] = row.Amount
                st.session_state.history.append(record)

# ════════════════════════════════════════════════════════════
# HISTORY
# ════════════════════════════════════════════════════════════
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="history-header">📋 Prediction History <span style="color:#8b949e;font-size:0.75rem;font-weight:400;">(this session)</span></div>', unsafe_allow_html=True)

if len(st.session_state.history) == 0:
    st.markdown('<div class="history-empty">No predictions yet — results will appear here after each detection.</div>', unsafe_allow_html=True)
else:
    df_history = pd.DataFrame(st.session_state.history)
    df_history.index = range(1, len(df_history) + 1)
    df_history.index.name = "No"

    total = len(df_history)
    fraud = df_history["Hasil"].str.contains("Fraud").sum()
    legit = total - fraud

    st.markdown(f"""
    <div class="pill-row">
        <span class="pill pill-total">📊 Total: {total}</span>
        <span class="pill pill-fraud">🚨 Fraud: {fraud}</span>
        <span class="pill pill-legit">✅ Legitimate: {legit}</span>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(df_history, use_container_width=True)

    if st.button("🗑️  Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    FraudShield &nbsp;·&nbsp; Transaction Fraud Detection &nbsp;·&nbsp;
    Machine Learning Project &nbsp;·&nbsp; ;·&nbsp;
</div>
""", unsafe_allow_html=True)
