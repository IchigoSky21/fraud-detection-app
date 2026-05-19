import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ── Inisialisasi Session State ────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []

# ── Fungsi Data Sampel (Diperbaiki dengan data representatif) ─
def set_sample_data(is_fraud=False):
    if is_fraud:
        # Pola umum dari transaksi Fraud (nilai negatif ekstrim pada V3, V14, V17)
        sample = {
            "v1": -2.3, "v2": 1.5, "v3": -4.2, "v4": 3.8, "v5": -1.2,
            "v6": -1.4, "v7": -2.8, "v8": 1.1, "v9": -1.5, "v10": -3.5,
            "v11": 2.2, "v12": -4.0, "v13": 0.5, "v14": -5.5, "v15": 0.2,
            "v16": -3.0, "v17": -4.8, "v18": -1.5, "v19": 0.8, "v20": 0.3,
            "v21": 0.5, "v22": -0.2, "v23": 0.1, "v24": -0.3, "v25": 0.2,
            "v26": 0.4, "v27": 0.5, "v28": 0.1
        }
        st.session_state["amount_input"] = 125.50
    else:
        # Pola umum dari transaksi Normal (nilai di sekitar 0)
        sample = {
            "v1": 1.2, "v2": 0.1, "v3": 0.5, "v4": 0.2, "v5": -0.1,
            "v6": -0.2, "v7": 0.1, "v8": -0.1, "v9": 0.3, "v10": -0.1,
            "v11": 0.1, "v12": 0.4, "v13": -0.2, "v14": 0.3, "v15": 1.0,
            "v16": 0.5, "v17": -0.2, "v18": 0.1, "v19": -0.2, "v20": 0.0,
            "v21": -0.1, "v22": -0.2, "v23": 0.1, "v24": 0.0, "v25": 0.2,
            "v26": -0.1, "v27": 0.0, "v28": 0.0
        }
        st.session_state["amount_input"] = 25.00
        
    for key, val in sample.items():
        st.session_state[key] = val

# ── Load model & scaler ──────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

st.set_page_config(page_title="Fraud Detection App", page_icon="🔍", layout="centered")

# ════════════════════════════════════════════════════════════
# FITUR 1: SLIDER THRESHOLD KEPUTUSAN (Sidebar)
# ════════════════════════════════════════════════════════════
st.sidebar.title("⚙️ Pengaturan Model")
st.sidebar.markdown("Sesuaikan batas probabilitas untuk menentukan klasifikasi *fraud*.")

# Secara default, threshold adalah 0.5 (50%)
threshold = st.sidebar.slider(
    "Decision Threshold", 
    min_value=0.01, 
    max_value=0.99, 
    value=0.50, 
    step=0.01,
    help="Tingkatkan nilai ini jika Anda ingin model lebih berhati-hati sebelum menuduh transaksi sebagai fraud (mengurangi False Positives)."
)

# ── HEADER UTAMA ─────────────────────────────────────────────
st.title("🔍 Transaction Fraud Detection")
st.markdown("Masukkan data transaksi untuk mendeteksi apakah transaksi tersebut **fraudulent** atau **legitimate**.")

mode = st.radio("Pilih mode input:", ["Manual Input", "Upload CSV"])

# ════════════════════════════════════════════════════════════
# MODE 1 — Manual Input
# ════════════════════════════════════════════════════════════
if mode == "Manual Input":
    st.subheader("Input Fitur Transaksi")
    
    st.markdown("**Gunakan Data Sampel:**")
    btn1, btn2, _ = st.columns([1, 1, 2])
    btn1.button("✅ Sampel Normal", on_click=set_sample_data, args=(False,), use_container_width=True)
    btn2.button("🚨 Sampel Fraud", on_click=set_sample_data, args=(True,), use_container_width=True)
    
    st.markdown("---")

    col1, col2 = st.columns(2)
    feature_values = {}

    with col1:
        for i in range(1, 15):
            feature_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.6f", key=f"v{i}")

    with col2:
        for i in range(15, 29):
            feature_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.6f", key=f"v{i}")

    amount = st.number_input("Amount (nilai transaksi)", value=0.0, min_value=0.0, format="%.2f", key="amount_input")
    feature_values["Amount"] = amount

    if st.button("🔎 Prediksi Manual", use_container_width=True, type="primary"):
        df_input = pd.DataFrame([feature_values])
        df_input["Amount"] = scaler.transform(df_input[["Amount"]])

        start = time.time()
        probability = model.predict_proba(df_input)[0][1]
        
        # MENGGUNAKAN THRESHOLD DARI SIDEBAR
        prediction = 1 if probability >= threshold else 0
        latency = (time.time() - start) * 1000  

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
        
        st.session_state["history"].append({
            "Waktu": time.strftime("%H:%M:%S"),
            "Mode": "Manual",
            "Amount": f"${amount:,.2f}",
            "Probabilitas": f"{probability:.2%}",
            "Threshold Digunakan": f"{threshold:.2f}",
            "Hasil": result_text,
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
            probs = model.predict_proba(df_process)[:, 1]
            
            # MENGGUNAKAN THRESHOLD DARI SIDEBAR UNTUK BATCH CSV
            preds = [1 if p >= threshold else 0 for p in probs]
            latency = (time.time() - start) * 1000

            df_upload["Prediction"]        = ["Fraud" if p == 1 else "Legitimate" for p in preds]
            df_upload["Fraud_Probability"] = [f"{p:.2%}" for p in probs]

            st.success(f"✅ Prediksi selesai untuk {len(df_upload)} transaksi dalam {latency:.1f} ms")

            n_fraud = sum(preds)
            n_legit = len(preds) - n_fraud
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Transaksi", len(preds))
            c2.metric("🚨 Fraud",        n_fraud)
            c3.metric("✅ Legitimate",   n_legit)

            st.dataframe(df_upload[["Prediction", "Fraud_Probability"] + required_cols])

            st.session_state["history"].append({
                "Waktu": time.strftime("%H:%M:%S"),
                "Mode": f"CSV Batch ({len(preds)} baris)",
                "Amount": "N/A",
                "Probabilitas": "N/A",
                "Threshold Digunakan": f"{threshold:.2f}",
                "Hasil": f"{n_fraud} Fraud / {n_legit} Legit",
                "Latency": f"{latency:.1f} ms"
            })

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
