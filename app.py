import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ── 1. PAGE CONFIGURATION ──────────────────────────────────────
# Pastikan ini berada paling atas sebelum elemen UI lainnya
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── 2. LOAD ML ARTIFACTS ───────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, encoders, feature_names

try:
    model, scaler, encoders, feature_names = load_artifacts()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file .pkl berada di folder yang sama. Error: {e}")

# ── 3. SIDEBAR NAVIGATION ──────────────────────────────────────
st.sidebar.image("https://images.unsplash.com/photo-1614064641936-3820480c5eaf?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80", use_column_width=True)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["💳 Detection Dashboard", "⚙️ How It Works", "👥 About Us"])

st.sidebar.markdown("---")
st.sidebar.info("Sistem Pendeteksi Penipuan Transaksi berbasis Random Forest Classifier.")

# ═══════════════════════════════════════════════════════════════
# PAGE 1: DETECTION DASHBOARD
# ═══════════════════════════════════════════════════════════════
if page == "💳 Detection Dashboard":
    st.title("💳 Transaction Fraud Detection")
    st.markdown("Masukkan detail transaksi di bawah ini untuk mendapatkan analisis risiko secara *real-time*.")
    
    # ── Constants ──
    CATEGORIES = ['shopping_net', 'shopping_pos', 'grocery_pos', 'grocery_net', 'gas_transport', 'travel', 'misc_net', 'misc_pos', 'food_dining', 'health_fitness', 'home', 'kids_pets', 'personal_care', 'entertainment']
    STATES = ['NY','CA','TX','FL','IL','PA','OH','GA','AL','AK','AZ','AR','CO','CT','DE','HI','ID','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NC','ND','OK','OR','RI','SC','SD','TN','UT','VT','VA','WA','WV','WI','WY']
    
    with st.form("prediction_form"):
        st.subheader("Form Detail Transaksi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**💰 Financial & Merchant**")
            amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=50.0, step=10.0)
            category = st.selectbox("Spending Category", CATEGORIES)
            distance_km = st.number_input("Distance to Merchant (km)", min_value=0.0, value=5.0, step=1.0)
            
            st.markdown("**📅 Time & Date**")
            month = st.selectbox("Month", ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], index=11)
            day_of_week = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], index=6)
            hour = st.slider("Hour (0-23)", 0, 23, 12)

        with col2:
            st.markdown("**👤 Cardholder Profile**")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["M", "F"])
            
            st.markdown("**📍 Location Info**")
            state = st.selectbox("State", STATES)
            city_pop = st.number_input("City Population", min_value=100, value=50000, step=5000)

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("Analyze Transaction", use_container_width=True)

    # ── Prediction Logic ──
    if submit_btn:
        with st.spinner('Menganalisis pola transaksi dengan AI...'):
            start_time = time.time()
            
            try:
                # Mapping manual
                day_map = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}
                month_map = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6, "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}
                
                def safe_encode(enc_name, val):
                    if val in encoders[enc_name].classes_:
                        return encoders[enc_name].transform([val])[0]
                    return 0
                
                # Susun Data
                raw = {
                    'category': safe_encode('category', category),
                    'amt': amt,
                    'age': age,
                    'distance_km': distance_km,
                    'city_pop': city_pop,
                    'gender': safe_encode('gender', gender),
                    'state': safe_encode('state', state),
                    'hour': hour,
                    'day_of_week': day_map.get(day_of_week, 0),
                    'month': month_map.get(month, 1),
                    'zip': 10000, 
                    'merchant': safe_encode('merchant', 'Unknown'),
                    'city': safe_encode('city', 'Unknown'),
                    'job': safe_encode('job', 'Unknown')
                }
                
                df_input = pd.DataFrame([raw])
                
                # Pastikan kolom sesuai model
                for col in feature_names:
                    if col not in df_input.columns:
                        df_input[col] = 0
                df_input = df_input[feature_names]

                # Scaling
                scale_cols = ['amt', 'city_pop', 'age', 'distance_km', 'zip']
                df_input[scale_cols] = scaler.transform(df_input[scale_cols])

                # Predict & Threshold Logic (15%)
                probability = model.predict_proba(df_input)[0][1]
                is_fraud = 1 if probability > 0.15 else 0
                
                latency = (time.time() - start_time) * 1000

                # ── UI Results ──
                st.divider()
                st.subheader("Hasil Analisis")
                
                if is_fraud:
                    st.error("🚨 **FRAUDULENT TRANSACTION DETECTED** - Aktivitas berisiko tinggi ditemukan!")
                else:
                    st.success("✅ **LEGITIMATE TRANSACTION** - Transaksi aman untuk diproses.")

                met1, met2, met3 = st.columns(3)
                met1.metric("System Decision", "Block / Alert" if is_fraud else "Approve")
                met2.metric("Risk Probability", f"{probability:.1%}")
                met3.metric("Inference Latency", f"{latency:.1f} ms")
                
                st.progress(float(probability), text="Risk Level Indicator")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses data: {str(e)}")


# ═══════════════════════════════════════════════════════════════
# PAGE 2: HOW IT WORKS
# ═══════════════════════════════════════════════════════════════
elif page == "⚙️ How It Works":
    st.image("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", use_column_width=True)
    st.title("Bagaimana Sistem Ini Bekerja?")
    
    st.markdown("""
    ### 1. Latar Belakang Bisnis
    Di era ekonomi digital, kejahatan siber khususnya penipuan kartu kredit (*Credit Card Fraud*) menyebabkan kerugian miliaran dolar setiap tahunnya. Sistem berbasis *Machine Learning* sangat krusial di industri perbankan karena mampu meninjau ribuan transaksi per detik dan mengambil keputusan jauh lebih cepat daripada investigator manusia.

    ### 2. Kompleksitas Data (Imbalanced Data)
    Tantangan terbesar dalam proyek ini adalah sifat data yang sangat tidak seimbang (*Imbalanced*). Dalam dunia nyata, kasus penipuan berjumlah kurang dari **1%** dari total transaksi harian. Model AI konvensional akan kesulitan mendeteksi anomali sekecil ini.

    Oleh karena itu, sistem ini dirancang menggunakan algoritma **Random Forest Classifier** dengan pendekatan rekayasa fitur (*Feature Engineering*) mendalam dan penyesuaian ambang batas keputusan (*Threshold Tuning*) menjadi 15% untuk meningkatkan sensitivitas deteksi sistem.

    ### 3. Rekayasa Fitur (Feature Engineering)
    Sistem kami tidak hanya menganalisis nominal uang, melainkan menciptakan variabel prediktif baru seperti:
    * **Geographical Distance:** Menghitung jarak fisik (*km*) antara lokasi nasabah dan *merchant*.
    * **Temporal Features:** Mengekstraksi jam, hari, dan bulan transaksi untuk mempelajari pola waktu operasional para penipu.
    * **Demographic Profiling:** Mengkalkulasi umur (*age*) pengguna berdasarkan tanggal lahir.
    """)
    
    st.info("""
    **⚙️ ALUR PEMROSESAN SISTEM (PIPELINE):**
    1. **Data Input** 👉 Web menangkap 10 parameter transaksi pengguna.
    2. **Preprocessing** 👉 Konversi data teks (Kategori/Negara) menjadi angka matematis (Label Encoding).
    3. **Standardization** 👉 Penyetaraan skala angka menggunakan *StandardScaler*.
    4. **Inference** 👉 Model *Random Forest* mengambil *voting* dari 100 pohon keputusan.
    5. **Decision Output** 👉 Jika persentase risiko > 15%, blokir transaksi dalam waktu < 100 milidetik!
    """)


# ═══════════════════════════════════════════════════════════════
# PAGE 3: ABOUT US
# ═══════════════════════════════════════════════════════════════
elif page == "👥 About Us":
    st.image("https://images.unsplash.com/photo-1522071820081-009f0129c71c?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", use_column_width=True)
    st.title("Tentang Kami")
    
    st.markdown("""
    
    Aplikasi web ini dikembangkan sebagai bagian dari tugas proyek akhir / purwarupa (*Proof of Concept*) untuk mendemonstrasikan implementasi algoritma kecerdasan buatan dalam memecahkan masalah industri finansial.
    
    ---
    **Tim Kami:**
    * Anggota 1 - [Felix Zonattan]
    * Anggota 2 - [Jason Benoit Adianto]
    * Anggota 3 - [Keivan Aliegery Indriartho]
    * Anggota 4 - [Ivander Sanusi]
    * Anggota 5 - [Haposan Emmanuel Tobias]
    
    *(Hak Cipta © 2026)*
    """)
