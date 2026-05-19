import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ── Inisialisasi Session State untuk History & Input ──────────
if "history" not in st.session_state:
    st.session_state["history"] = []

# Fungsi untuk mengisi data sampel (Fitur 5)
def set_sample_data(is_fraud=False):
    # Data ilustrasi. Pada fraud, variansi biasanya lebih ekstrim
    for i in range(1, 29):
        # Angka acak sekedar simulasi: Normal (kecil), Fraud (ekstrim)
        val = np.random.uniform(-4.0, 4.0) if is_fraud else np.random.uniform(-0.5, 0.5)
        st.session_state[f"v{i}"] = float(val)
    st.session_state["amount_input"] = 850.50 if is_fraud else 25.00

# ── Load model & scaler ──────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ── Konfigurasi halaman ──────────────────────────────────────
st.set_page_config(
    page_title = "Fraud Detection App",
    page_icon  = "🔍",
    layout     = "centered"
)

# ════════════════════════════════════════════════════════════
# FITUR 4: GLOBAL FEATURE IMPORTANCE (Sidebar)
# ════════════════════════════════════════════════════════════
st.sidebar.title("📊 Model Insights")
st.sidebar.markdown("Menampilkan fitur (variabel) mana yang paling mempengaruhi keputusan AI.")

# Mengecek atribut model untuk Feature Importance
try:
    feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    importances = None
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0]) # Ambil nilai absolut untuk Regresi Logistik/Linear
        
    if importances is not None and len(importances) == len(feature_names):
        # Ambil 10 fitur teratas
        df_imp = pd.DataFrame({"Importance": importances}, index=feature_names)
        df_imp = df_imp.sort_values(by="Importance", ascending=True).tail(10)
        st.sidebar.markdown("**Top 10 Fitur Paling Berpengaruh**")
        st.sidebar.bar_chart(df_imp, height=350)
    else:
        st.sidebar.info("Model tidak memiliki atribut Feature Importance.")
except Exception as e:
    st.sidebar.info("Gagal memuat Feature Importance.")

# ── HEADER UTAMA ─────────────────────────────────────────────
st.title("🔍 Transaction Fraud Detection")
st.markdown("Masukkan data transaksi untuk mendeteksi apakah transaksi tersebut **fraudulent** atau **legitimate**.")

mode = st.radio("Pilih mode input:", ["Manual Input", "Upload CSV"])

# ════════════════════════════════════════════════════════════
# MODE 1 — Manual Input
# ════════════════════════════════════════════════════════════
if mode == "Manual Input":
    st.subheader("Input Fitur Transaksi")
    
    # FITUR 5: Tombol Data Sampel
    st.markdown("**Gunakan Data Sampel:**")
    btn1, btn2, _ = st.columns([1, 1, 2])
    btn1.button("✅ Sampel Normal", on_click=set_sample_data, args=(False,), use_container_width=True)
    btn2.button("🚨 Sampel Fraud", on_click=set_sample_data, args=(True,), use_container_width=True)
    
    st.markdown("---")

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

    amount = st.number_input("Amount (nilai transaksi)", value=0.0, min_value=0.0, format="%.2f", key="amount_input")
    feature_values["Amount"] = amount

    if st.button("🔎 Prediksi Manual", use_container_width=True, type="primary"):
        # Preprocessing
        df_input = pd.DataFrame([feature_values])
        df_input["Amount"] = scaler.transform(df_input[["Amount"]])

        # Inference
        start = time.time()
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]
        latency = (time.time() - start) * 1000  # ms

        # Output
        st.divider()
        result_text = "Fraud" if prediction == 1 else "Legitimate"
        if prediction == 1:
            st.error("🚨 FRAUDULENT TRANSACTION")
        else:
            st.success("✅ LEGITIMATE TRANSACTION")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Prediksi",          result_text)
        col_b.metric("Fraud Probability", f"{probability:.2%}")
        col_c.metric("Latency",           f"{latency:.1f} ms")

        st.progress(float(probability))
        
        # Simpan ke History
        st.session_state["history"].append({
            "Waktu": time.strftime("%H:%M:%S"),
            "Mode": "Manual",
            "Amount": f"${amount:,.2f}",
            "Hasil": result_text,
            "Probabilitas": f"{probability:.2%}",
            "Latency": f"{latency:.1f} ms"
        })

# ════════════════════════════════════════════════════════════
# MODE 2 — Upload CSV
# ════════════════════════════════════════════════════════════
else:
    st.subheader("Upload File CSV")
    st.markdown("CSV harus memiliki kolom: `V1` hingga `V28` dan `Amount`.")

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        required_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        missing = [c for c in required_cols if c not in df_upload.columns]

        if missing:
            st.error(f"Kolom berikut tidak ditemukan: {missing}")
        else:
            df_process = df_upload[required_cols].copy()
            df_process["Amount"] = scaler.transform(df_process[["Amount"]])

            start = time.time()
            preds = model.predict(df_process)
            probs = model.predict_proba(df_process)[:, 1]
            latency = (time.time() - start) * 1000

            df_upload["Prediction"]        = ["Fraud" if p == 1 else "Legitimate" for p in preds]
            df_upload["Fraud_Probability"] = [f"{p:.2%}" for p in probs]

            st.success(f"✅ Prediksi selesai untuk {len(df_upload)} transaksi dalam {latency:.1f} ms")

            # Ringkasan Angka
            n_fraud = sum(preds)
            n_legit = len(preds) - n_fraud
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Transaksi", len(preds))
            c2.metric("🚨 Fraud",        n_fraud)
            c3.metric("✅ Legitimate",   n_legit)

            # FITUR 3: Visualisasi Hasil CSV Sederhana
            st.markdown("#### 📊 Distribusi Prediksi")
            # Membuat DataFrame kecil khusus untuk visualisasi
            chart_df = pd.DataFrame({"Jumlah": [n_legit, n_fraud]}, index=["Legitimate", "Fraud"])
            st.bar_chart(chart_df, height=250)

            st.dataframe(df_upload[["Prediction", "Fraud_Probability"] + required_cols])

            # Simpan ke History
            st.session_state["history"].append({
                "Waktu": time.strftime("%H:%M:%S"),
                "Mode": f"CSV Batch ({len(preds)} baris)",
                "Amount": "N/A",
                "Hasil": f"{n_fraud} Fraud / {n_legit} Legit",
                "Probabilitas": "N/A",
                "Latency": f"{latency:.1f} ms"
            })

            # Download hasil
            csv_out = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Hasil Prediksi", csv_out, "fraud_prediction_results.csv", "text/csv")

# ════════════════════════════════════════════════════════════
# HISTORI PREDIKSI
# ════════════════════════════════════════════════════════════
st.divider()
st.subheader("🕒 Riwayat Prediksi Sesi Ini")

if len(st.session_state["history"]) > 0:
    df_history = pd.DataFrame(st.session_state["history"])[::-1].reset_index(drop=True)
    st.dataframe(df_history, use_container_width=True)
    
    col_hist1, col_hist2 = st.columns([1, 4])
    with col_hist1:
        if st.button("🗑️ Hapus Riwayat", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()
else:
    st.info("Belum ada transaksi yang diprediksi pada sesi ini.")

st.divider()
st.caption("Transaction Fraud Detection App | Powered by Streamlit")
