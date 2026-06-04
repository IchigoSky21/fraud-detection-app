import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ── Load model & artifacts ────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model         = joblib.load("fraud_model.pkl")
    scaler        = joblib.load("scaler.pkl")
    encoders      = joblib.load("encoders.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, encoders, feature_names

model, scaler, encoders, feature_names = load_artifacts()

# ── Session state ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Fraud Detection App",
    page_icon  = "🔍",
    layout     = "centered"
)

# ── Custom CSS (sama seperti versi sebelumnya) ────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
@media (prefers-color-scheme: light) {
    .stApp { background: #dbeafe !important;
        background-image: radial-gradient(ellipse 80% 50% at 50% -10%, rgba(59,130,246,0.20) 0%, transparent 70%), radial-gradient(ellipse 60% 40% at 85% 85%, rgba(96,165,250,0.14) 0%, transparent 60%) !important; }
}
@media (prefers-color-scheme: dark) {
    .stApp { background: #0c1a3a !important;
        background-image: radial-gradient(ellipse 80% 50% at 50% -10%, rgba(29,78,216,0.22) 0%, transparent 70%), radial-gradient(ellipse 60% 40% at 85% 85%, rgba(37,99,235,0.14) 0%, transparent 60%) !important; }
}
input, textarea, [data-baseweb="input"] input, div[data-testid="stNumberInput"] input {
    background-color: #ffffff !important; color: #1e3a5f !important;
    border: 1px solid rgba(59,130,246,0.22) !important; border-radius: 8px !important; box-shadow: none !important; }
[data-baseweb="input"], [data-baseweb="base-input"],
div[data-testid="stNumberInput"] > div, div[data-testid="stNumberInput"] > div > div {
    background-color: #ffffff !important; border: none !important; box-shadow: none !important; }
div[data-testid="stNumberInput"] button { background-color: #eff6ff !important; color: #1d4ed8 !important; border: 1px solid rgba(59,130,246,0.20) !important; }
div[data-baseweb="select"] > div { background-color: #ffffff !important; border: 1px solid rgba(59,130,246,0.22) !important; border-radius: 8px !important; color: #1e3a5f !important; }
label, div[data-testid="stNumberInput"] label, div[data-testid="stRadio"] p { color: #475569 !important; font-size: 0.82rem !important; }
div[data-testid="stCaptionContainer"] p { color: #64748b !important; }
.hero { text-align: center; padding: 2.2rem 1rem 1.4rem; margin-bottom: 1rem; }
.hero-badge { display: inline-block; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; padding: 0.28rem 0.9rem; border-radius: 999px; border: 1px solid rgba(59,130,246,0.40); background: rgba(255,255,255,0.55); color: #1d4ed8; margin-bottom: 0.9rem; backdrop-filter: blur(6px); }
.hero-title { font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800; line-height: 1.15; margin-bottom: 0.55rem; letter-spacing: -0.02em; color: #1e3a5f; }
.hero-sub { font-size: 0.92rem; font-weight: 300; max-width: 420px; margin: 0 auto; line-height: 1.6; color: #475569; }
.card { background: rgba(255,255,255,0.78) !important; border: 1px solid rgba(59,130,246,0.18) !important; border-radius: 14px; padding: 1.5rem; margin-bottom: 1.1rem; backdrop-filter: blur(10px); }
.card-title { font-family: 'Syne', sans-serif; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #1d4ed8; margin-bottom: 0.9rem; }
.result-fraud { background: linear-gradient(135deg,rgba(220,38,38,0.10),rgba(185,28,28,0.05)); border: 1px solid rgba(220,38,38,0.35); border-radius: 12px; padding: 1.2rem 1.4rem; text-align: center; margin: 0.9rem 0; }
.result-legit { background: linear-gradient(135deg,rgba(22,163,74,0.10),rgba(15,118,55,0.05)); border: 1px solid rgba(22,163,74,0.35); border-radius: 12px; padding: 1.2rem 1.4rem; text-align: center; margin: 0.9rem 0; }
.result-label { font-family: 'Syne', sans-serif; font-size: 1.35rem; font-weight: 800; letter-spacing: 0.03em; }
.result-fraud .result-label { color: #dc2626; }
.result-legit .result-label { color: #16a34a; }
.metrics-row { display: flex; gap: 0.75rem; margin: 0.9rem 0; }
.metric-box { flex: 1; border-radius: 10px; border: 1px solid rgba(59,130,246,0.15); background: rgba(239,246,255,0.85); padding: 0.8rem 0.9rem; text-align: center; }
.metric-label { font-size: 0.68rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.25rem; color: #64748b; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 700; color: #1e3a5f; }
.metric-value.speed { color: #1d4ed8; } .metric-value.fraud { color: #dc2626; } .metric-value.legit { color: #16a34a; }
.prob-bar-wrap { margin: 0.7rem 0 0.2rem; }
.prob-bar-label { font-size: 0.72rem; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.35rem; color: #64748b; }
.prob-bar-bg { border-radius: 999px; height: 7px; overflow: hidden; background: rgba(59,130,246,0.12); }
.prob-bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg,#ef4444,#f97316); }
.pill-row { display: flex; gap: 0.5rem; margin-bottom: 0.9rem; flex-wrap: wrap; }
.pill { padding: 0.28rem 0.8rem; border-radius: 999px; font-size: 0.76rem; font-weight: 600; border: 1px solid; }
.pill-total { background: rgba(59,130,246,0.10); color: #1d4ed8; border-color: rgba(59,130,246,0.3); }
.pill-fraud { background: rgba(220,38,38,0.10); color: #dc2626; border-color: rgba(220,38,38,0.3); }
.pill-legit { background: rgba(22,163,74,0.10); color: #16a34a; border-color: rgba(22,163,74,0.3); }
.history-header { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; margin-bottom: 0.75rem; color: #1e3a5f; }
.history-empty { font-size: 0.86rem; text-align: center; padding: 1.4rem; border: 1px dashed rgba(59,130,246,0.25); border-radius: 10px; color: #64748b; background: rgba(255,255,255,0.5); }
.section-divider { border: none; border-top: 1px solid rgba(59,130,246,0.15); margin: 1.6rem 0; }
.stButton > button { background: linear-gradient(135deg,#1d4ed8,#3b82f6) !important; color: #fff !important; border: none !important; border-radius: 10px !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 0.92rem !important; padding: 0.6rem 1.1rem !important; transition: opacity 0.2s !important; }
.stButton > button:hover { opacity: 0.85 !important; }
.footer { text-align: center; font-size: 0.73rem; padding: 1.8rem 0 0.8rem; letter-spacing: 0.03em; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge">Machine Learning · Binary Classification</div>
    <div class="hero-title">Transaction Fraud Detection</div>
    <div class="hero-sub">Enter your credit card transaction details to detect whether it is fraudulent or legitimate.</div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# KATEGORI & KONSTANTA
# ════════════════════════════════════════════════════════════
CATEGORIES = [
    'food_dining', 'gas_transport', 'grocery_net', 'grocery_pos',
    'health_fitness', 'home', 'kids_pets', 'misc_net', 'misc_pos',
    'personal_care', 'shopping_net', 'shopping_pos', 'travel',
    'entertainment'
]
STATES = [
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID',
    'IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS',
    'MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK',
    'OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV',
    'WI','WY'
]

# ════════════════════════════════════════════════════════════
# FORM INPUT
# ════════════════════════════════════════════════════════════
st.markdown('<div class="card"><div class="card-title">Transaction Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    category    = st.selectbox("Spending Category", CATEGORIES)
    amt         = st.number_input("Transaction Amount ($)", min_value=0.0, value=50.0, format="%.2f")
    gender      = st.selectbox("Cardholder Gender", ["M", "F"])
    age         = st.number_input("Cardholder Age", min_value=18, max_value=100, value=35)
    city_pop    = st.number_input("City Population", min_value=100, value=50000, step=1000)

with col2:
    state       = st.selectbox("State", STATES)
    hour        = st.slider("Transaction Hour (0-23)", 0, 23, 12)
    day_of_week = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    month       = st.selectbox("Month", ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    distance_km = st.number_input("Distance to Merchant (km)", min_value=0.0, value=5.0, format="%.1f")

st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PREDIKSI
# ════════════════════════════════════════════════════════════
if st.button("Predict", use_container_width=True):
    try:
        # Encode inputs
        day_map   = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
                     "Friday":4,"Saturday":5,"Sunday":6}
        month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
                     "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

        def safe_encode(encoder, value):
            classes = list(encoder.classes_)
            if value in classes:
                return encoder.transform([value])[0]
            return 0  # fallback jika tidak ditemukan

        # Buat dict fitur sesuai urutan feature_names
        raw = {
            'merchant'    : safe_encode(encoders['merchant'], 'fraud_Rippin, Kub and Mann'),
            'category'    : safe_encode(encoders['category'], category),
            'amt'         : amt,
            'gender'      : safe_encode(encoders['gender'], gender),
            'city'        : safe_encode(encoders['city'], 'Unknown'),
            'state'       : safe_encode(encoders['state'], state),
            'zip'         : 10000,
            'city_pop'    : city_pop,
            'job'         : safe_encode(encoders['job'], 'Unknown'),
            'hour'        : hour,
            'day_of_week' : day_map[day_of_week],
            'month'       : month_map[month],
            'age'         : age,
            'distance_km' : distance_km,
        }

        df_input = pd.DataFrame([raw])

        # Pastikan urutan kolom sama dengan training
        for col in feature_names:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[feature_names]

        # Scale
        scale_cols = ['amt', 'city_pop', 'age', 'distance_km', 'zip']
        df_input[scale_cols] = scaler.transform(df_input[scale_cols])

        # Predict
        start       = time.time()
        prediction  = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]
        latency     = (time.time() - start) * 1000

        # Output
        if prediction == 1:
            st.markdown('<div class="result-fraud"><div class="result-label">FRAUDULENT TRANSACTION</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-legit"><div class="result-label">LEGITIMATE TRANSACTION</div></div>', unsafe_allow_html=True)

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

        # Simpan ke history
        record = {
            "Result"      : "Fraud" if prediction == 1 else "Legitimate",
            "Category"    : category,
            "Amount ($)"  : amt,
            "Gender"      : gender,
            "Age"         : age,
            "State"       : state,
            "Hour"        : hour,
            "Day"         : day_of_week,
            "Distance(km)": distance_km,
            "Probability" : f"{probability:.2%}",
        }
        st.session_state.history.append(record)

    except Exception as e:
        st.error(f"Error saat prediksi: {e}")

# ════════════════════════════════════════════════════════════
# HISTORY
# ════════════════════════════════════════════════════════════
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="history-header">Prediction History <span style="font-size:0.72rem;font-weight:400;opacity:0.5;">(this session only)</span></div>', unsafe_allow_html=True)

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
    Transaction Fraud Detection &nbsp;
</div>
""", unsafe_allow_html=True)
