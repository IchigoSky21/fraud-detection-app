# 💳 Fraud Detection AI

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Analytics-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

A Streamlit-based proof of concept for screening credit-card transactions with a serialized Random Forest classifier. The application accepts a set of transaction and cardholder attributes, preprocesses them with the artifacts bundled in the repository, and returns a fraud-risk probability with an application-level decision threshold.

> **Important:** This project is an educational proof of concept and should not be treated as a production banking or fraud-decision system.

## 🚀 Live Demo

**[Open the Streamlit application](https://fraud-detection-app-oz629kbvyj8qzc7833krcn.streamlit.app/)**

The repository also contains a Google Colab notebook used for the project's machine-learning experiments:

**[Open the Google Colab notebook](https://colab.research.google.com/drive/1tJOP2ZK9rriLeWPyMgowvBNkodpJVxQg?usp=sharing)**

## ✨ Features

- **Transaction risk screening** — Enter transaction details and receive a model-generated fraud probability.
- **15% decision threshold** — The application marks a transaction as fraud-indicated when the predicted probability is **greater than 0.15**.
- **Risk gauge** — Visualizes the predicted probability and the 15% decision threshold.
- **Session history** — Keeps inspected transactions in the current Streamlit session and provides summary metrics for the session.
- **Heuristic indicators** — Displays simple rule-based indicators for unusually large amounts, long merchant distance, overnight transactions, and selected online transaction categories.
- **Global feature importance** — Displays the Random Forest's global `feature_importances_` when available.
- **Light and dark modes** — The interface includes two visual themes based on the project's case-file design.
- **Responsive interface** — The custom CSS includes mobile-specific adjustments.
- **Pretrained artifacts included** — The repository contains the serialized model, scaler, encoders, and feature-name metadata required by the application.

## 🧠 How the Application Works

The application loads four serialized artifacts at startup:

- `fraud_model.pkl` — trained fraud-classification model.
- `scaler.pkl` — fitted `StandardScaler` used for numerical preprocessing.
- `encoders.pkl` — fitted label encoders for categorical values.
- `feature_names.pkl` — feature ordering expected by the model.

For each submitted transaction, the application:

1. Collects **10 user-facing inputs** covering transaction amount, category, merchant distance, month, day, hour, age, gender, state, and city population.
2. Converts categorical values with the bundled encoders.
3. Builds the model input using the feature names stored in `feature_names.pkl`.
4. Standardizes the numerical columns `amt`, `city_pop`, `age`, `distance_km`, and `zip` with the bundled scaler.
5. Calls `predict_proba()` on the serialized model.
6. Applies the application threshold: `probability > 0.15` → **Fraud Indicated**; otherwise → **Legitimate**.
7. Records the result in the current Streamlit session history.

Some model features are populated internally rather than entered directly by the user. The current application supplies default values for fields such as `zip`, `merchant`, `city`, and `job` when constructing the prediction record.

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Web App | Streamlit |
| Navigation | streamlit-option-menu |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn / serialized Random Forest model |
| Model Persistence | Joblib / Pickle artifacts |
| Visualization | Plotly |
| Frontend Styling | Custom CSS injected through Streamlit |

The runtime dependencies currently declared in `requirements.txt` are Streamlit, Pandas, NumPy, Scikit-Learn, Joblib, streamlit-option-menu, and Plotly.

## 📊 Model Evaluation

The original project documentation reports the following comparative experiment results:

| Metric | Random Forest (Baseline) | Random Forest (Tuned) | XGBoost | Logistic Regression |
|---|---:|---:|---:|---:|
| Accuracy | 0.9978 | 0.9981 | 0.9979 | 0.9587 |
| Precision | **0.9832** | 0.9197 | 0.8215 | 0.0713 |
| Recall | 0.4966 | 0.6037 | 0.6497 | **0.7109** |
| ROC-AUC | 0.9714 | 0.9816 | **0.9874** | 0.8320 |
| PR-AUC | **0.7922** | 0.7777 | 0.7805 | 0.1374 |

These figures are documented as **project experiment results**; the training dataset and model-training notebook are not stored in this repository's current file tree. The deployed application's inference model is the serialized `fraud_model.pkl` included in the repository.

## 🔍 What the Interface Shows

### Dashboard

The main dashboard provides:

- A transaction input form.
- Fraud-risk probability.
- System decision and inference latency.
- A Plotly risk gauge.
- Optional heuristic indicators.
- Global Random Forest feature importance.
- A session-only transaction log with total checks, fraud indications, and average risk.
- A button to clear the current session history.

### System Methodology

The application includes a dedicated methodology page describing the project's approach to imbalanced fraud data, feature engineering, preprocessing, model inference, and the 15% decision threshold.

### Team

The application currently lists these project members:

- Felix Zonattan
- Jason Benoit Adianto
- Keivan Aliegery Indriartho
- Ivander Sanusi
- Haposan Emmanuel Tobias

## 📁 Repository Structure

```text
fraud-detection-app/
├── .devcontainer/
│   └── devcontainer.json
├── .streamlit/
│   └── config.toml
├── assets/
│   └── logo.svg
├── app.py
├── encoders.pkl
├── feature_names.pkl
├── fraud_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

The `.pkl` files are required runtime artifacts and are intentionally part of the repository because the application loads them directly at startup.

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/IchigoSky21/fraud-detection-app.git
cd fraud-detection-app
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Streamlit

```bash
streamlit run app.py
```

The application will normally be available at `http://localhost:8501`.

### GitHub Codespaces / Dev Container

The repository also contains `.devcontainer/devcontainer.json`. It uses the Microsoft Python 3.11 Bookworm development-container image, installs the declared Python requirements, and forwards port `8501` for the Streamlit application.

## ⚠️ Limitations

- The application is a **proof of concept**, not a production fraud-detection service.
- The current repository does not contain the original training dataset or training code in its file tree.
- The README's evaluation table reflects previously documented experiments and is not automatically recomputed when the application runs.
- The 15% threshold is an application decision rule, not a universally valid banking risk threshold.
- Several model features are filled with internal defaults rather than collected from the user interface.
- Session history exists only in Streamlit session state and is cleared when the session is reset or the user presses **Clear Session**.
- Heuristic indicators are supplementary rules and should not be interpreted as model explanations for an individual prediction.

## 🔒 Security & Privacy Note

This application is designed as a demonstration. Do not use real sensitive cardholder information in a public deployment or shared development environment. The application should not be interpreted as a compliance-ready financial system.

## 📄 License

No `LICENSE` file is currently included in this repository. Until an explicit license is added, assume that the source code is **not licensed for unrestricted reuse or redistribution**.
