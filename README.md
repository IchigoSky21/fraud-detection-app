
# 💳 Fraud Detection AI: Real-Time Credit Card Fraud Mitigation System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Analytics-3F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

**Fraud Detection AI** adalah sistem pemantauan cerdas berbasis web *real-time* yang ditenagai oleh algoritma **Random Forest Classifier** yang telah dioptimalkan. Sistem ini mengevaluasi 10 atribut transaksional utama secara simultan (seperti jarak geografis, waktu pemrosesan, dan nominal pengeluaran) untuk menghasilkan probabilitas risiko penipuan dalam hitungan milidetik guna menghentikan aktivitas ilegal sebelum mengompromikan jaringan finansial perbankan.

Aplikasi ini dibuat khusus untuk memenuhi studi kasus industri finansial yang menghadapi tantangan ketimpangan data ekstrem (*extreme class imbalance*).

---

## 🚀 Tautan Penting / Quick Links

* **Aplikasi Web Live (Streamlit):** [fraud-detection-app.streamlit.app](https://fraud-detection-app-oz629kbvyj8qzc7833krcn.streamlit.app/)
* **Eksperimen Kode Belakang Layar (Google Colab):** [Google Colab Notebook](https://colab.research.google.com/drive/1tJOP2ZK9rriLeWPyMgowvBNkodpJVxQg?usp=sharing)

---

## ✨ Fitur Utama Sistem / Key Features

1. **Real-Time Risk Inference:** Melakukan prediksi klasifikasi risiko transaksi dengan latensi rendah (kurang dari 60 milidetik).
2. **Custom Threshold Optimization (15%):** Menggeser batas keputusan bawaan (50%) menjadi 15% berdasarkan matriks risiko perbankan guna mendeteksi transaksi penipuan semaksimal mungkin (*High Recall Strategy*).
3. **Interactive Visual Dashboard:** Mengintegrasikan komponen *Risk Gauge Meter* menggunakan Plotly untuk indikator visual tingkat bahaya transaksi yang instan dan intuitif.
4. **Session History Management:** Menyediakan pencatatan log riwayat pengujian interaktif selama sesi berlangsung yang dapat dihapus secara berkala (*Clear Session*).
5. **Explainable AI Pipeline:** Transparansi alur kerja data mulai dari rekayasa fitur (*Feature Engineering*) hingga visualisasi korelasi data makro.

---

## 🛠️ Arsitektur Teknologi / Tech Stack

* **Core Language:** Python 3.9+
* **Framework Dashboard:** Streamlit, Streamlit Option Menu
* **Machine Learning Engine:** Scikit-Learn (Random Forest, StandardScaler, LabelEncoder)
* **Data Processing:** Pandas, NumPy
* **Data Visualization:** Plotly Graph Objects, Matplotlib, Seaborn
* **Model Serialization:** Joblib / Pickle

---

## 📊 Dataset & Karakteristik Data

Proyek ini memanfaatkan dataset historis transaksi kartu kredit berskala besar dari **Kaggle** (oleh Kartik2112).
* **Volume Data Train:** ~135,9K baris data
* **Volume Data Test:** ~135,9K baris data
* **Tantangan Utama:** Data sangat tidak seimbang (*Highly Imbalanced Data*) dengan tingkat kasus penipuan nyata hanya sebesar **0.94%** (1.284 kasus fraud dari total data pelatihan).
* **Solusi Penanganan:** Mengaktifkan parameter `class_weight='balanced'` di model Random Forest untuk menerapkan metode *Cost-Sensitive Learning*.

---

## ⚙️ Alur Pemrosesan Data / System Pipeline

Sistem memproses data melalui 5 tahapan terpadu sebelum mengeluarkan keputusan akhir:
1. **Data Input Capture:** Antarmuka web menangkap 10 parameter transaksi yang diisi oleh pengguna secara *real-time*.
2. **Preprocessing (Label Encoding):** Mengonversi data kategorikal teks (seperti kategori pengeluaran, gender, dan negara bagian) menjadi angka indeks matematis berbasis objek encoder.
3. **Standardization:** Menyetarakan skala data numerik kontinu (`amt`, `city_pop`, `age`, `distance_km`) menggunakan objek `StandardScaler`.
4. **Ensemble Inference:** Model Random Forest mengeksekusi data untuk mengekstrak nilai probabilitas berdasarkan keputusan kolektif (*voting*) dari 100 pohon keputusan.
5. **Decision Output:** Jika probabilitas risiko menembus angka **15% (0.15)**, sistem otomatis mengeluarkan keputusan penolakan (*Block / Alert*).

---

## 📈 Hasil Performa Model / Model Evaluation Results

Berdasarkan pengujian komparatif komprehensif terhadap berbagai arsitektur model, algoritma *Tree-Based Ensemble* menunjukkan hasil terbaik dalam menangani ketimpangan data ini:

| Metrik Evaluasi | Random Forest (Baseline) | Random Forest (Tuned) | XGBoost | Logistic Regression |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 0.9978 | 0.9981 | 0.9979 | 0.9587 |
| **Precision** | **0.9832** | 0.9197 | 0.8215 | 0.0713 |
| **Recall** | 0.4966 | 0.6037 | 0.6497 | **0.7109** |
| **ROC-AUC** | 0.9714 | **0.9816** | 0.9874 | 0.8320 |
| **PR-AUC** | **0.7922** | 0.7777 | 0.7805 | 0.1374 |

*Model utama yang di-deploy dipilih berdasarkan keseimbangan performa PR-AUC tertinggi pada kelas minoritas untuk meminimalkan tingkat False Positive.*

---

## 💻 Cara Menjalankan Secara Lokal / Installation Guide

Ikuti langkah-langkah berikut untuk menjalankan aplikasi web ini di komputer lokal Anda:

1. **Kloning Repositori:**
   ```bash
   git clone [https://github.com/username/fraud-detection-app.git](https://github.com/username/fraud-detection-app.git)
   cd fraud-detection-app
