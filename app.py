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
    page_title = "Fraud Detection System",
    page_icon  = "💳",
    layout     = "centered"
)

# ════════════════════════════════════════════════════════════
# HEADER & DESKRIPSI
# ════════════════════════════════════════════════════════════
st.title("💳 Transaction Fraud Detection")
st.markdown("Sistem klasifikasi ini akan menganalisis data transaksi kartu kredit Anda untuk mendeteksi potensi aktivitas penipuan secara *real-time*.")
st.divider()

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
# FORM INPUT (Lebih rapi dan mencegah auto-refresh)
# ════════════════════════════════════════════════════════════
with st.form("prediction_form"):
    st.subheader("Transaction Details")
    
    # Membagi form menjadi 2 kolom agar seimbang
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Financial & Merchant Info**")
        amt         = st.number_input("Transaction Amount ($)", min_value=0.0, value=50.0, format="%.2f")
        category    = st.selectbox("Spending Category", CATEGORIES)
        distance_km = st.number_input("Distance to Merchant (km)", min_value=0.0, value=5.0, format="%.1f")
        
        st.markdown("**Time & Date**")
        month       = st.selectbox("Month", ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
        day_of_week = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        hour        = st.slider("Transaction Hour (0-23)", 0, 23, 12)

    with col2:
        st.markdown("**Cardholder Profile**")
        age         = st.number_input("Cardholder Age", min_value=18, max_value=100, value=35)
        gender      = st.selectbox("Cardholder Gender", ["M", "F"])
        
        st.markdown("**Location Info**")
        state       = st.selectbox("State", STATES)
        city_pop    = st.number_input("City Population", min_value=100, value=50000, step=1000)

    st.markdown("<br>", unsafe_allow_html=True) # Spasi kecil
    submit_button = st.form_submit_button("Analyze Transaction", use_container_width=True)

# ════════════════════════════════════════════════════════════
# LOGIKA PREDIKSI & HASIL
# ════════════════════════════════════════════════════════════
if submit_button:
    with st.spinner('Analyzing transaction patterns...'):
        try:
            # Encode inputs
            day_map   = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
            month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

            def safe_encode(encoder, value):
                classes = list(encoder.classes_)
                if value in classes:
                    return encoder.transform([value])[0]
                return 0  # fallback

            # Susun data input
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

            # Pastikan urutan kolom sesuai model
            for col in feature_names:
                if col not in df_input.columns:
                    df_input[col] = 0
            df_input = df_input[feature_names]

            # Scaling
            scale_cols = ['amt', 'city_pop', 'age', 'distance_km', 'zip']
            df_input[scale_cols] = scaler.transform(df_input[scale_cols])

            # Prediksi dan Latensi
            start       = time.time()
            prediction  = model.predict(df_input)[0]
            probability = model.predict_proba(df_input)[0][1]
            latency     = (time.time() - start) * 1000

            # ── Tampilan Hasil Evaluasi Bawaan Streamlit ──
            st.divider()
            st.subheader("Analysis Result")
            
            if prediction == 1:
                st.error("🚨 **FRAUDULENT TRANSACTION DETECTED**")
            else:
                st.success("✅ **LEGITIMATE TRANSACTION**")

            # Menggunakan st.metric untuk menampilkan angka secara rapi dan profesional
            met1, met2, met3 = st.columns(3)
            met1.metric("Status", "Fraud" if prediction == 1 else "Legitimate")
            met2.metric("Fraud Risk Probability", f"{probability:.1%}")
            met3.metric("Inference Latency", f"{latency:.1f} ms")

            # Progress bar visual untuk tingkat risiko
            st.progress(float(probability), text="Risk Level Indicator")

            # Simpan ke history
            record = {
                "Result"      : "Fraud" if prediction == 1 else "Legitimate",
                "Category"    : category,
                "Amount ($)"  : amt,
                "Gender"      : gender,
                "Age"         : age,
                "State"       : state,
                "Probability" : f"{probability:.2%}",
                "Latency"     : f"{latency:.1f} ms"
            }
            st.session_state.history.append(record)

        except Exception as e:
            st.error(f"Error saat prediksi: {e}")

# ════════════════════════════════════════════════════════════
# HISTORY (Riwayat Prediksi)
# ════════════════════════════════════════════════════════════
st.divider()
st.subheader("Session History")

if len(st.session_state.history) == 0:
    st.info("Belum ada transaksi yang diuji. Silakan jalankan prediksi melalui form di atas.")
else:
    df_history = pd.DataFrame(st.session_state.history)
    df_history.index = range(1, len(df_history) + 1)
    
    # Menampilkan dataframe dengan style bawaan yang bersih
    st.dataframe(df_history, use_container_width=True)

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
