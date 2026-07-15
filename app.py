import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
from pathlib import Path

# ── 1. PAGE CONFIGURATION ──────────────────────────────────────
# [Rekomendasi #1] layout="wide" agar form 2 kolom & hasil analisis
# tidak berdesakan di layar besar.
st.set_page_config(
    page_title="Fraud Detect AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

ASSETS_DIR = Path(__file__).parent / "assets"


def render_svg(filename: str, height: str = "180px"):
    """[Rekomendasi #2] Render SVG lokal (bukan hotlink Unsplash/Flaticon).
    Aset disimpan di /assets sehingga tidak tergantung koneksi eksternal
    dan konsisten dengan warna brand (#2563eb)."""
    svg_path = ASSETS_DIR / filename
    if svg_path.exists():
        svg_content = svg_path.read_text()
        st.markdown(
            f'<div style="width:100%; margin-bottom:14px;">{svg_content}</div>',
            unsafe_allow_html=True
        )
    else:
        st.warning(f"Aset {filename} tidak ditemukan.")


# ── 2. GLOBAL CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .risk-card {
        border-radius: 14px;
        padding: 22px 26px;
        margin-top: 8px;
        border: 1px solid #e2e8f0;
    }
    .risk-card.fraud { background-color: #fef2f2; border-color: #fecaca; }
    .risk-card.legit { background-color: #f0fdf4; border-color: #bbf7d0; }
    .model-badge {
        display: inline-block;
        background-color: #eff6ff;
        color: #1e3a8a;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 12px;
        margin: 3px 4px 3px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── 3. INITIALIZE SESSION STATE FOR HISTORY ────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── 4. LOAD ML ARTIFACTS ───────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, encoders, feature_names

model_loaded_ok = True
try:
    model, scaler, encoders, feature_names = load_artifacts()
except Exception as e:
    model_loaded_ok = False
    st.error(f"Gagal memuat model. Pastikan file .pkl berada di folder yang sama. Error: {e}")


def get_global_feature_importance(top_n=5):
    """[Rekomendasi #3] Ambil feature_importances_ asli dari Random Forest
    (bukan skor per-prediksi/SHAP) untuk transparansi model secara umum."""
    if not model_loaded_ok or not hasattr(model, "feature_importances_"):
        return None
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False).head(top_n)


def render_risk_gauge(probability: float):
    """[Rekomendasi #3] Gauge warna dinamis (hijau/kuning/merah) menggantikan
    st.progress bawaan yang warnanya statis."""
    if probability < 0.15:
        bar_color = "#16a34a"
    elif probability < 0.5:
        bar_color = "#f59e0b"
    else:
        bar_color = "#dc2626"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 34}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 15], "color": "#dcfce7"},
                {"range": [15, 50], "color": "#fef9c3"},
                {"range": [50, 100], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "#1e293b", "width": 3},
                "thickness": 0.85,
                "value": 15,
            },
        },
        title={"text": "Tingkat Risiko", "font": {"size": 15}},
    ))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=45, b=10))
    return fig


def get_heuristic_flags(amt, distance_km, hour, category):
    """[Rekomendasi #3] Flag heuristik sederhana pada INPUT pengguna
    (bukan kontribusi model) supaya user paham indikator umum risiko fraud.
    Dilabeli eksplisit sebagai heuristik agar tidak overclaim sebagai
    penjelasan keputusan model."""
    flags = []
    if amt > 500:
        flags.append("💰 Nominal transaksi tergolong tinggi (> $500)")
    if distance_km > 50:
        flags.append("📍 Jarak ke merchant cukup jauh dari lokasi biasa (> 50 km)")
    if hour in [0, 1, 2, 3, 4, 5]:
        flags.append("🕑 Transaksi terjadi pada jam rawan (dini hari)")
    if category in ["shopping_net", "misc_net"]:
        flags.append("🌐 Kategori transaksi online cenderung berisiko lebih tinggi")
    return flags


# ── 5. SIDEBAR NAVIGATION ───────────────────────────────────────
with st.sidebar:
    col_logo, col_text = st.columns([1, 4])
    with col_logo:
        render_svg("logo.svg", height="40px")
    with col_text:
        st.markdown("<h3 style='margin-top: 2px; color: #1e293b;'>Fraud Detect AI</h3>", unsafe_allow_html=True)

    st.markdown("---")

    page = option_menu(
        menu_title=None,
        options=["Dashboard", "Cara Kerja Sistem", "Tentang Kami"],
        icons=["shield-check", "diagram-3", "people-fill"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "5px 0px",
                "border-radius": "8px",
                "font-family": "sans-serif"
            },
            "nav-link-selected": {"background-color": "#2563eb", "color": "white", "font-weight": "bold"},
        }
    )

    st.markdown("---")

    # [Rekomendasi #7] Badge status model di sidebar
    st.markdown("**Status Sistem**")
    status_icon = "✅" if model_loaded_ok else "❌"
    st.markdown(
        f"""
        <span class="model-badge">{status_icon} Model {'Aktif' if model_loaded_ok else 'Gagal Dimuat'}</span>
        <span class="model-badge">🌲 Random Forest</span>
        <span class="model-badge">🎯 Threshold 15%</span>
        """,
        unsafe_allow_html=True
    )
    st.info("Sistem Pendeteksi Penipuan Transaksi berbasis Random Forest Classifier.")

# ═══════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════
if page == "Dashboard":
    render_svg("hero_dashboard.svg")
    st.markdown("Masukkan detail transaksi di bawah ini untuk mendapatkan analisis risiko secara **real-time**.")

    # ── Constants ──
    CATEGORIES = ['shopping_net', 'shopping_pos', 'grocery_pos', 'grocery_net', 'gas_transport', 'travel', 'misc_net', 'misc_pos', 'food_dining', 'health_fitness', 'home', 'kids_pets', 'personal_care', 'entertainment']
    STATES = ['NY','CA','TX','FL','IL','PA','OH','GA','AL','AK','AZ','AR','CO','CT','DE','HI','ID','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NC','ND','OK','OR','RI','SC','SD','TN','UT','VT','VA','WA','WV','WI','WY']

    # [Rekomendasi #4] Mapping populasi kota jadi kategori yang lebih intuitif
    CITY_POP_OPTIONS = {
        "Desa kecil (< 5 ribu)": 2500,
        "Kota kecil (5 ribu - 50 ribu)": 25000,
        "Kota menengah (50 ribu - 200 ribu)": 100000,
        "Kota besar (200 ribu - 1 juta)": 500000,
        "Metropolitan (> 1 juta)": 1500000,
    }

    with st.form("prediction_form"):
        st.subheader("Form Detail Transaksi")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**💰 Finansial & Merchant**")
            amt = st.number_input(
                "Nominal Transaksi ($)", min_value=0.0, value=50.0, step=10.0,
                help="Nominal transaksi yang besar dan tidak biasa sering menjadi indikator awal fraud."
            )
            category = st.selectbox(
                "Kategori Transaksi", CATEGORIES,
                help="Kategori transaksi online (net) umumnya memiliki risiko fraud lebih tinggi dibanding offline (pos)."
            )
            distance_km = st.number_input(
                "Jarak ke Merchant (km)", min_value=0.0, value=5.0, step=1.0,
                help="Jarak fisik antara lokasi kartu terdaftar dan lokasi merchant saat transaksi terjadi."
            )

            st.divider()

            st.markdown("**📅 Waktu Transaksi**")
            month = st.selectbox(
                "Bulan", ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], index=11,
                help="Beberapa periode (mis. musim liburan) memiliki pola transaksi berbeda."
            )
            day_of_week = st.selectbox(
                "Hari", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], index=6,
                help="Pola transaksi akhir pekan bisa berbeda dari hari kerja."
            )
            hour = st.slider(
                "Jam (0-23)", 0, 23, 12,
                help="Transaksi pada dini hari (00:00-05:00) secara statistik lebih berisiko."
            )

        with col2:
            st.markdown("**👤 Profil Pemegang Kartu**")
            age = st.number_input("Usia", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Jenis Kelamin", ["M", "F"])

            st.divider()

            st.markdown("**📍 Informasi Lokasi**")
            state = st.selectbox("Negara Bagian", STATES)
            city_pop_label = st.selectbox(
                "Populasi Kota", list(CITY_POP_OPTIONS.keys()), index=2,
                help="Perkiraan jumlah penduduk kota tempat transaksi dilakukan."
            )
            city_pop = CITY_POP_OPTIONS[city_pop_label]

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("🔍 Analisis Transaksi", use_container_width=True, type="primary")

    # ── Prediction Logic ──
    if submit_btn:
        with st.spinner('Menganalisis pola transaksi dengan AI...'):
            start_time = time.time()

            try:
                day_map = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}
                month_map = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6, "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}

                def safe_encode(enc_name, val):
                    if val in encoders[enc_name].classes_:
                        return encoders[enc_name].transform([val])[0]
                    return 0

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

                for col in feature_names:
                    if col not in df_input.columns:
                        df_input[col] = 0
                df_input = df_input[feature_names]

                scale_cols = ['amt', 'city_pop', 'age', 'distance_km', 'zip']
                df_input[scale_cols] = scaler.transform(df_input[scale_cols])

                probability = model.predict_proba(df_input)[0][1]
                is_fraud = 1 if probability > 0.15 else 0
                latency = (time.time() - start_time) * 1000

                # ── UI Results ──
                st.divider()
                st.subheader("Hasil Analisis")

                # [Rekomendasi #3] Card-style result
                card_class = "fraud" if is_fraud else "legit"
                with st.container(border=True):
                    if is_fraud:
                        st.error("🚨 **TRANSAKSI TERINDIKASI FRAUD** — Aktivitas berisiko tinggi ditemukan!")
                    else:
                        st.success("✅ **TRANSAKSI SAH** — Transaksi aman untuk diproses.")

                    res_col1, res_col2 = st.columns([1, 1])
                    with res_col1:
                        met1, met2 = st.columns(2)
                        met1.metric("Keputusan Sistem", "Blokir / Waspada" if is_fraud else "Setujui")
                        met2.metric("Waktu Inferensi", f"{latency:.1f} ms")
                        st.caption(f"Ambang batas keputusan: 15% • Probabilitas terukur: {probability:.1%}")

                        flags = get_heuristic_flags(amt, distance_km, hour, category)
                        if flags:
                            with st.expander("⚠️ Indikator umum yang terdeteksi pada input Anda (heuristik)"):
                                for f in flags:
                                    st.write(f"- {f}")

                    with res_col2:
                        st.plotly_chart(render_risk_gauge(probability), use_container_width=True)

                    global_importance = get_global_feature_importance()
                    if global_importance is not None:
                        with st.expander("🔍 Fitur yang paling berpengaruh secara umum pada model ini"):
                            st.caption("Ini adalah *feature importance* global dari Random Forest, bukan penjelasan spesifik untuk transaksi ini.")
                            st.bar_chart(global_importance)

                record = {
                    "Hasil": "🚨 Fraud" if is_fraud else "✅ Sah",
                    "Risiko": f"{probability:.1%}",
                    "Nominal": f"${amt:.2f}",
                    "Kategori": category,
                    "Usia": age,
                    "Jarak": f"{distance_km} km",
                    "Latensi": f"{latency:.1f} ms"
                }
                st.session_state.history.append(record)

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses data: {str(e)}")

    # ── Riwayat Sesi ──
    st.divider()
    st.subheader("🕰️ Riwayat Sesi")

    if len(st.session_state.history) == 0:
        st.info("Belum ada transaksi yang diuji pada sesi ini.")
    else:
        df_history = pd.DataFrame(st.session_state.history)
        df_history.index = range(1, len(df_history) + 1)

        # [Rekomendasi #5] Ringkasan analitik di atas tabel riwayat
        total_tx = len(df_history)
        fraud_count = df_history["Hasil"].str.contains("Fraud").sum()
        avg_risk = df_history["Risiko"].str.rstrip("%").astype(float).mean()

        sum1, sum2, sum3 = st.columns(3)
        sum1.metric("Total Transaksi Diuji", total_tx)
        sum2.metric("Terdeteksi Fraud", fraud_count)
        sum3.metric("Rata-rata Risiko", f"{avg_risk:.1f}%")

        # [Rekomendasi #5] Color-coding baris fraud/sah
        def highlight_result(row):
            color = "#fee2e2" if "Fraud" in row["Hasil"] else "#dcfce7"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            df_history.style.apply(highlight_result, axis=1),
            use_container_width=True
        )

        if st.button("Hapus Riwayat", type="secondary"):
            st.session_state.history = []
            st.rerun()

# ═══════════════════════════════════════════════════════════════
# PAGE 2: CARA KERJA SISTEM
# ═══════════════════════════════════════════════════════════════
elif page == "Cara Kerja Sistem":
    render_svg("hero_howitworks.svg")

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
    1. **Data Input** 👉 Web menangkap parameter transaksi pengguna.
    2. **Preprocessing** 👉 Konversi data teks (Kategori/Negara) menjadi angka matematis (Label Encoding).
    3. **Standardization** 👉 Penyetaraan skala angka menggunakan *StandardScaler*.
    4. **Inference** 👉 Model *Random Forest* mengambil *voting* dari seluruh pohon keputusan.
    5. **Decision Output** 👉 Jika persentase risiko > 15%, sistem memicu peringatan waspada.
    """)

    global_importance = get_global_feature_importance(top_n=8)
    if global_importance is not None:
        st.markdown("### 4. Fitur Paling Berpengaruh (Global)")
        st.caption("Diambil dari `feature_importances_` model Random Forest — menunjukkan fitur mana yang paling sering digunakan model untuk membedakan transaksi, secara keseluruhan (bukan per transaksi).")
        st.bar_chart(global_importance)

# ═══════════════════════════════════════════════════════════════
# PAGE 3: TENTANG KAMI
# ═══════════════════════════════════════════════════════════════
elif page == "Tentang Kami":
    render_svg("hero_about.svg")

    st.markdown("""

    Aplikasi web ini dikembangkan sebagai bagian dari tugas proyek akhir / purwarupa (*Proof of Concept*) untuk mendemonstrasikan implementasi algoritma kecerdasan buatan dalam memecahkan masalah industri finansial.

    ---
    **Tim Kami:**
    * Anggota 1 - Felix Zonattan
    * Anggota 2 - Jason Benoit Adianto
    * Anggota 3 - Keivan Aliegery Indriartho
    * Anggota 4 - Ivander Sanusi
    * Anggota 5 - Haposan Emmanuel Tobias

    *(Hak Cipta © 2026)*
    """)
