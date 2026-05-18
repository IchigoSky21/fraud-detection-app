import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ── Load model & scaler ──────────────────────────────────────
model  = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# ── Konfigurasi halaman ──────────────────────────────────────
st.set_page_config(
    page_title = "Fraud Detection App",
    page_icon  = "🔍",
    layout     = "centered"
)

st.title("🔍 Transaction Fraud Detection")
st.markdown("Masukkan data transaksi untuk mendeteksi apakah transaksi tersebut **fraudulent** atau **legitimate**.")

# ── Pilih mode input ─────────────────────────────────────────
mode = st.radio("Pilih mode input:", ["Manual Input", "Upload CSV"])

# ════════════════════════════════════════════════════════════
# MODE 1 — Manual Input
# ════════════════════════════════════════════════════════════
if mode == "Manual Input":
    st.subheader("Input Fitur Transaksi")

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
    feature_values["Amount"] = amount

    if st.button("🔎 Prediksi", use_container_width=True):
        # Preprocessing
        df_input = pd.DataFrame([feature_values])
        df_input["Amount"] = scaler.transform(df_input[["Amount"]])

        # Inference + ukur latency
        start = time.time()
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]
        latency = (time.time() - start) * 1000  # ms

        # Output
        st.divider()
        if prediction == 1:
            st.error("🚨 FRAUDULENT TRANSACTION")
        else:
            st.success("✅ LEGITIMATE TRANSACTION")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Prediksi",          "Fraud" if prediction == 1 else "Legitimate")
        col_b.metric("Fraud Probability", f"{probability:.2%}")
        col_c.metric("Latency",           f"{latency:.1f} ms")

        # Progress bar probabilitas
        st.markdown("**Fraud Probability:**")
        st.progress(float(probability))

# ════════════════════════════════════════════════════════════
# MODE 2 — Upload CSV
# ════════════════════════════════════════════════════════════
else:
    st.subheader("Upload File CSV")
    st.markdown("CSV harus memiliki kolom: `V1` hingga `V28` dan `Amount` (tanpa kolom `id` dan `Class`).")

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)

        # Validasi kolom
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

            # Ringkasan
            n_fraud = sum(preds)
            n_legit = len(preds) - n_fraud
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Transaksi", len(preds))
            c2.metric("🚨 Fraud",        n_fraud)
            c3.metric("✅ Legitimate",   n_legit)

            st.dataframe(df_upload[["Prediction", "Fraud_Probability"] + required_cols])

            # Download hasil
            csv_out = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button(
                label     = "⬇️ Download Hasil Prediksi",
                data      = csv_out,
                file_name = "fraud_prediction_results.csv",
                mime      = "text/csv"
            )

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.caption("Transaction Fraud Detection")