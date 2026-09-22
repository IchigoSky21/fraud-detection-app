import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
from pathlib import Path

# ── 1. PAGE CONFIGURATION ──────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detect AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

ASSETS_DIR = Path(__file__).parent / "assets"


def render_svg(filename: str):
    """Render SVG lokal dari /assets."""
    svg_path = ASSETS_DIR / filename
    if svg_path.exists():
        st.markdown(f'<div class="svg-wrap">{svg_path.read_text()}</div>', unsafe_allow_html=True)
    else:
        st.warning(f"Aset {filename} tidak ditemukan.")


# ── 2. DESIGN SYSTEM: "BERKAS KASUS" (CASE FILE) ───────────────
# Konsep: meja investigasi fraud — map arsip, cap tinta, buku besar —
# bukan dashboard SaaS generik. Dua suasana disediakan: "Kertas"
# (meja kerja siang di dekat jendela) dan "Malam" (meja kerja di bawah
# lampu baca). Bukan sekadar invert warna, tapi identitas yang sama
# dibaca dalam dua kondisi cahaya berbeda.
PALETTES = {
    "light": {
        "paper": "#F6F8FB", "paper_light": "#FFFFFF",
        "ink": "#0F172A", "ink_muted": "#64748B",
        "accent": "#173B63", "accent_dark": "#102B49", "accent_text": "#FFFFFF",
        "sidebar_bg": "#FFFFFF", "sidebar_border": "#DCE3EC", "on_dark": "#0F172A",
        "tape": "#64748B",
        "stamp_red": "#C0392B", "stamp_green": "#16845B", "stamp_amber": "#B7791F",
        "rule": "#DCE3EC",
        "row_fraud": "#FCEDEC", "row_legit": "#EAF7F1",
        "gauge_steps": ["#EAF7F1", "#FFF7E6", "#FCEDEC"],
        "stamp_blend": "multiply", "stamp_opacity": "0.9",
    },
    "dark": {
        "paper": "#0B1220", "paper_light": "#111B2E",
        "ink": "#E7EEF7", "ink_muted": "#94A3B8",
        "accent": "#8DB7E8", "accent_dark": "#6E9FD6", "accent_text": "#0B1220",
        "sidebar_bg": "#111B2E", "sidebar_border": "#263752", "on_dark": "#E7EEF7",
        "tape": "#94A3B8",
        "stamp_red": "#F07B70", "stamp_green": "#55C995", "stamp_amber": "#E8B75A",
        "rule": "#263752",
        "row_fraud": "#3A1D1B", "row_legit": "#123326",
        "gauge_steps": ["#123326", "#3A2D13", "#3A1D1B"],
        "stamp_blend": "normal", "stamp_opacity": "0.95",
    },
}

# Baca preferensi mode SEBELUM CSS dirakit (pola standar Streamlit:
# session_state sudah terisi dari run sebelumnya walau widget-nya
# baru didefinisikan nanti di sidebar).
dark_mode = st.session_state.get("dark_mode_toggle", False)
pal = PALETTES["dark"] if dark_mode else PALETTES["light"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root {{ --paper:{pal['paper']}; --surface:{pal['paper_light']}; --ink:{pal['ink']}; --muted:{pal['ink_muted']}; --accent:{pal['accent']}; --accent-dark:{pal['accent_dark']}; --border:{pal['rule']}; --safe:{pal['stamp_green']}; --warn:{pal['stamp_amber']}; --danger:{pal['stamp_red']}; }}
[data-testid="stAppViewContainer"],.stApp {{background:var(--paper);}}
[data-testid="stHeader"] {{background:var(--paper);}}
html,body,[class*="css"] {{font-family:'DM Sans',sans-serif;color:var(--ink);}}
.stMarkdown,.stMarkdown p,.stMarkdown li,.stCaption,.stAlert,.stText,.stTextInput label,.stNumberInput label,[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] p {{color:var(--ink)!important;}}
h1,h2,h3,h4,h5,h6 {{font-family:'DM Sans',sans-serif!important;color:var(--ink)!important;letter-spacing:-.02em;}}
hr {{border-color:var(--border)!important;}}
[data-testid="stSidebar"] {{background:var(--surface)!important;border-right:1px solid var(--border);}}
[data-testid="stSidebar"] h3,[data-testid="stSidebar"] label,[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{color:var(--ink)!important;}}
[data-testid="stSidebar"] .nav-link {{color:var(--muted)!important;border-radius:10px!important;padding:10px 12px!important;margin:3px 0!important;font-weight:600!important;letter-spacing:0!important;text-transform:none!important;}}
[data-testid="stSidebar"] .nav-link:hover {{background:var(--paper)!important;color:var(--ink)!important;}}
[data-testid="stSidebar"] .nav-link.active {{background:{'#1D2B3F' if dark_mode else '#EEF3F8'}!important;color:var(--accent)!important;border-left:3px solid #2563EB!important;}}
[data-testid="stSidebar"] .nav-link.active span,[data-testid="stSidebar"] .nav-link.active i {{color:var(--accent)!important;}}
[data-baseweb="popover"] [data-baseweb="menu"],ul[role="listbox"] {{background:var(--surface)!important;border:1px solid var(--border)!important;}}
[data-baseweb="menu"] li,[role="option"] {{color:var(--ink)!important;}}
[data-baseweb="select"] > div {{color:var(--ink)!important;}}
[data-baseweb="select"] input {{color:var(--ink)!important;}}
[data-baseweb="select"] [data-baseweb="select-value"] {{color:var(--ink)!important;}}
input,textarea {{color:var(--ink)!important;caret-color:var(--accent)!important;}}
input::placeholder,textarea::placeholder {{color:var(--muted)!important;opacity:1!important;}}
button {{color:var(--ink);}}
.stCheckbox label,.stRadio label {{color:var(--ink)!important;}}
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary p {{color:var(--ink)!important;}}
[data-testid="stDataFrame"] {{color:var(--ink)!important;}}
.stamp-block {{border:1px solid var(--border);background:var(--paper);padding:12px 14px;margin:12px 0;border-radius:12px;transform:none;}}
.stamp-row {{display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:11px;padding:4px 0;border-bottom:1px solid var(--border);}}
.stamp-row:last-child {{border-bottom:0;}}
.stamp-label {{color:var(--muted)!important;letter-spacing:0!important;}}
.stamp-value {{color:var(--ink)!important;font-weight:600;}}
.case-header {{background:transparent;border:0;border-left:0;padding:0 0 22px;margin:0;}}
.case-title {{font-family:'DM Sans',sans-serif!important;font-weight:700;font-size:2.15rem;color:var(--ink)!important;margin:0 0 7px;line-height:1.15;letter-spacing:-.03em;}}
.case-subtitle {{font-family:'DM Sans',sans-serif;font-style:normal;color:var(--muted);font-size:15px;margin:0;max-width:720px;}}
.section-tag {{display:block;font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.08em;text-transform:uppercase;background:transparent;color:var(--muted);padding:0;margin:16px 0 8px;}}
.stButton button,[data-testid="stFormSubmitButton"] button {{background:var(--accent)!important;color:white!important;border:0!important;border-radius:10px!important;min-height:46px;font-family:'DM Sans',sans-serif!important;font-weight:700;letter-spacing:0;text-transform:none;}}
.stButton button:hover,[data-testid="stFormSubmitButton"] button:hover {{background:var(--accent-dark)!important;}}
[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input,[data-baseweb="select"]>div {{background:var(--surface)!important;border-color:var(--border)!important;border-radius:9px!important;color:var(--ink)!important;font-family:'DM Sans',sans-serif!important;}}
[data-testid="stMetric"] {{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:13px;}}
[data-testid="stMetricValue"] {{font-family:'IBM Plex Mono',monospace!important;color:var(--ink)!important;}}
[data-testid="stMetricLabel"] {{color:var(--muted)!important;text-transform:none!important;letter-spacing:0!important;}}
.st-key-verdict_card {{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:16px!important;}}
.verdict-stamp {{display:inline-block;font-family:'DM Sans',sans-serif;font-size:1rem;font-weight:700;letter-spacing:0;text-transform:none;padding:7px 12px;border:0;border-radius:999px;color:white;background:var(--stamp-color);transform:none;opacity:1;mix-blend-mode:normal;margin-bottom:10px;}}
.verdict-detail {{font-family:'IBM Plex Mono',monospace;font-size:.78rem;color:var(--muted);margin-top:2px;}}
.evidence-list {{list-style:none;padding:0;margin:6px 0;}}
.evidence-list li {{padding:9px 0 9px 2px;border-bottom:1px solid var(--border);font-size:.88rem;}}
.evidence-list li:before {{content:"";}}
.sop-box {{border:1px solid var(--border);border-radius:16px;background:var(--surface);padding:20px 24px;margin:14px 0 20px;}}
.sop-header {{font-family:'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:.08em;font-size:11px;color:var(--accent);margin-bottom:10px;}}
.roster-row {{display:flex;align-items:center;gap:14px;padding:12px 14px;border:1px solid var(--border);border-radius:12px;margin:7px 0;background:var(--surface);}}
.roster-id {{font-family:'IBM Plex Mono',monospace;color:var(--accent);font-size:11px;width:26px;}}
.roster-name {{font-weight:600;}}
.file-footer {{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);text-align:right;margin-top:18px;}}
@media (max-width:800px) {{.case-title {{font-size:1.75rem;}}.verdict-stamp {{font-size:.9rem;}}}}

/*RESPONSIVE_PATCH*/
html,body { overflow-x:hidden; }
[data-testid="stAppViewContainer"] { min-width:0; }
.block-container { width:100%; max-width:1400px; padding-top:2rem; padding-left:clamp(1rem,3vw,3rem); padding-right:clamp(1rem,3vw,3rem); }
[data-testid="stSidebar"] { min-width:280px; }
[data-testid="stSidebar"] > div:first-child { padding-left:1rem; padding-right:1rem; }
[data-testid="stHorizontalBlock"] { min-width:0; }
[data-testid="stColumn"] { min-width:0!important; }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] { max-width:100%; }
.verdict-wrap { min-width:0; }
.case-subtitle { overflow-wrap:anywhere; }
.sop-box { overflow-wrap:anywhere; }
.roster-name { overflow-wrap:anywhere; }

@media (max-width: 1100px) {
  .block-container { padding-left:1.5rem; padding-right:1.5rem; }
  .case-title { font-size:1.95rem; }
  [data-testid="stSidebar"] { min-width:250px; }
  .sop-box { padding:18px 20px; }
}

@media (max-width: 768px) {
  .block-container { padding-top:1.25rem; padding-left:1rem; padding-right:1rem; }
  .case-header { padding-bottom:16px; }
  .case-title { font-size:1.65rem; line-height:1.12; }
  .case-subtitle { font-size:14px; line-height:1.5; }
  .section-tag { margin-top:12px; }
  .sop-box { padding:16px; border-radius:12px; }
  .roster-row { gap:10px; padding:10px 12px; }
  [data-testid="stMetric"] { padding:10px; }
  [data-testid="stMetricValue"] { font-size:1.35rem!important; }
  [data-testid="stMetricLabel"] { font-size:.78rem!important; }
  .verdict-stamp { font-size:.85rem; padding:6px 10px; }
  .verdict-detail { font-size:.7rem; line-height:1.5; }
  [data-testid="stFormSubmitButton"] button { min-height:48px; }
  [data-testid="stDataFrame"] { overflow-x:auto!important; }
}

@media (max-width: 640px) {
  .block-container { padding-left:.75rem; padding-right:.75rem; }
  h1 { font-size:1.5rem!important; }
  h2 { font-size:1.25rem!important; }
  h3 { font-size:1.08rem!important; }
  .case-title { font-size:1.45rem; }
  .case-subtitle { font-size:13px; }
  .stamp-block { padding:10px 12px; }
  .stamp-row { font-size:10px; gap:8px; }
  .sop-box { margin:10px 0 16px; padding:14px; }
  .sop-box ol { padding-left:20px; }
  .sop-box li { margin-bottom:7px; }
  .roster-row { padding:9px 10px; }
  .file-footer { text-align:left; }
  [data-testid="stMetric"] { padding:9px; }
  [data-testid="stMetricValue"] { font-size:1.15rem!important; }
  [data-testid="stMetricLabel"] { font-size:.72rem!important; }
  [data-testid="stPlotlyChart"] { margin-left:-6px; margin-right:-6px; width:calc(100% + 12px); }
  [data-baseweb="select"] > div { min-height:42px; }
  input { min-height:42px!important; }
  button { min-height:42px; }
}

@media (max-width: 480px) {
  .block-container { padding-left:.6rem; padding-right:.6rem; }
  .case-title { font-size:1.3rem; }
  .verdict-stamp { font-size:.78rem; }
  .verdict-detail { font-size:.64rem; }
  .sop-header { font-size:10px; }
}

</style>
""", unsafe_allow_html=True)

# ── 3. INITIALIZE SESSION STATE ─────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── 4. LOAD ML ARTIFACTS ────────────────────────────────────────
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
    if not model_loaded_ok or not hasattr(model, "feature_importances_"):
        return None
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False).head(top_n)


def render_risk_gauge(probability: float, pal: dict):
    """Gauge dinamis, direkolorasi mengikuti palet aktif (Kertas/Malam)
    alih-alih warna default web hijau/kuning/merah terang."""
    if probability < 0.15:
        bar_color = pal["stamp_green"]
    elif probability < 0.5:
        bar_color = pal["stamp_amber"]
    else:
        bar_color = pal["stamp_red"]

    steps = pal["gauge_steps"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 32, "family": "IBM Plex Mono", "color": pal["ink"]}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": pal["rule"]},
            "bar": {"color": bar_color, "thickness": 0.32},
            "bgcolor": pal["paper_light"],
            "bordercolor": pal["rule"],
            "steps": [
                {"range": [0, 15], "color": steps[0]},
                {"range": [15, 50], "color": steps[1]},
                {"range": [50, 100], "color": steps[2]},
            ],
            "threshold": {"line": {"color": pal["accent"], "width": 3}, "thickness": 0.85, "value": 15},
        },
        title={"text": "TINGKAT RISIKO", "font": {"size": 13, "family": "IBM Plex Mono", "color": pal["ink_muted"]}},
    ))
    fig.update_layout(
        height=230, margin=dict(l=20, r=20, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "IBM Plex Sans", "color": pal["ink"]},
    )
    return fig


def get_heuristic_flags(amt, distance_km, hour, category):
    flags = []
    if amt > 500:
        flags.append("Nominal transaksi tergolong tinggi (di atas $500)")
    if distance_km > 50:
        flags.append("Jarak ke merchant cukup jauh dari lokasi biasa (di atas 50 km)")
    if hour in [0, 1, 2, 3, 4, 5]:
        flags.append("Transaksi terjadi pada jam rawan (dini hari)")
    if category in ["shopping_net", "misc_net"]:
        flags.append("Kategori transaksi daring cenderung berisiko lebih tinggi")
    return flags


def render_case_header(title: str, subtitle: str):
    st.markdown(f"""
    <div class="case-header">
        <h1 class="case-title">{title}</h1>
        <p class="case-subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# ── 5. SIDEBAR NAVIGATION ───────────────────────────────────────
with st.sidebar:
    col_logo, col_text = st.columns([1, 3])
    with col_logo:
        render_svg("logo.svg")
    with col_text:
        st.markdown("<h3 style='margin-bottom:0;'>Fraud Detect AI</h3>", unsafe_allow_html=True)

    st.markdown("---")

    page = option_menu(
        menu_title=None,
        options=["Dashboard", "Cara Kerja Sistem", "Tentang Kami"],
        icons=["folder2-open", "diagram-3", "people"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": pal["tape"], "font-size": "15px"},
            "nav-link": {
                "font-family": "'IBM Plex Sans', sans-serif",
                "font-size": "13px",
                "font-weight": "600",
                "letter-spacing": "0.05em",
                "text-transform": "uppercase",
                "text-align": "left",
                "color": pal["tape"],
                "margin": "2px 0px",
                "padding": "12px 16px",
                "border-radius": "0px",
                "border-left": "4px solid transparent",
                "background-color": "transparent",
            },
            "nav-link-selected": {
                "background-color": pal["paper_light"],
                "color": pal["accent"],
                "border-left": f"4px solid {pal['stamp_red']}",
                "font-weight": "700",
            },
        }
    )

    st.markdown("---")

    st.toggle("Mode Malam", key="dark_mode_toggle")

    st.markdown(f"""
    <div class="stamp-block">
        <div class="stamp-row"><span class="stamp-label">MODEL</span><span class="stamp-value">RANDOM FOREST</span></div>
        <div class="stamp-row"><span class="stamp-label">STATUS</span><span class="stamp-value">{'AKTIF' if model_loaded_ok else 'GAGAL'}</span></div>
        <div class="stamp-row"><span class="stamp-label">AMBANG</span><span class="stamp-value">15%</span></div>
    </div>
    <p class="sidebar-note">Sistem pendeteksi indikasi penipuan transaksi kartu, berbasis Random Forest Classifier. Dipakai untuk penyaringan awal, bukan keputusan akhir.</p>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════
if page == "Dashboard":
    render_case_header(
        "Transaction Risk Analysis",
        "Enter transaction details and let the model estimate the probability of fraud."
    )

    CATEGORIES = ['shopping_net', 'shopping_pos', 'grocery_pos', 'grocery_net', 'gas_transport', 'travel', 'misc_net', 'misc_pos', 'food_dining', 'health_fitness', 'home', 'kids_pets', 'personal_care', 'entertainment']
    STATES = ['NY','CA','TX','FL','IL','PA','OH','GA','AL','AK','AZ','AR','CO','CT','DE','HI','ID','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NC','ND','OK','OR','RI','SC','SD','TN','UT','VT','VA','WA','WV','WI','WY']

    CITY_POP_OPTIONS = {
        "Desa kecil (< 5 ribu)": 2500,
        "Kota kecil (5 ribu - 50 ribu)": 25000,
        "Kota menengah (50 ribu - 200 ribu)": 100000,
        "Kota besar (200 ribu - 1 juta)": 500000,
        "Metropolitan (> 1 juta)": 1500000,
    }

    with st.form("prediction_form"):
        st.markdown("### Formulir Rincian Transaksi")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<span class="section-tag">A — Finansial &amp; Merchant</span>', unsafe_allow_html=True)
            amt = st.number_input(
                "Nominal Transaksi ($)", min_value=0.0, value=50.0, step=10.0,
                help="Nominal transaksi yang besar dan tidak biasa sering menjadi indikator awal fraud."
            )
            category = st.selectbox(
                "Kategori Transaksi", CATEGORIES,
                help="Kategori transaksi daring (net) umumnya berisiko lebih tinggi dibanding luring (pos)."
            )
            distance_km = st.number_input(
                "Jarak ke Merchant (km)", min_value=0.0, value=5.0, step=1.0,
                help="Jarak fisik antara lokasi kartu terdaftar dan lokasi merchant saat transaksi terjadi."
            )

            st.markdown('<span class="section-tag">B — Waktu Transaksi</span>', unsafe_allow_html=True)
            month = st.selectbox(
                "Bulan", ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], index=11,
                help="Beberapa periode (misalnya musim liburan) memiliki pola transaksi berbeda."
            )
            day_of_week = st.selectbox(
                "Hari", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], index=6,
                help="Pola transaksi akhir pekan bisa berbeda dari hari kerja."
            )
            hour = st.slider(
                "Jam (0-23)", 0, 23, 12,
                help="Transaksi pada dini hari (00.00-05.00) secara statistik lebih berisiko."
            )

        with col2:
            st.markdown('<span class="section-tag">C — Profil Pemegang Kartu</span>', unsafe_allow_html=True)
            age = st.number_input("Usia", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Jenis Kelamin", ["M", "F"])

            st.markdown('<span class="section-tag">D — Informasi Lokasi</span>', unsafe_allow_html=True)
            state = st.selectbox("Negara Bagian", STATES)
            city_pop_label = st.selectbox(
                "Populasi Kota", list(CITY_POP_OPTIONS.keys()), index=2,
                help="Perkiraan jumlah penduduk kota tempat transaksi dilakukan."
            )
            city_pop = CITY_POP_OPTIONS[city_pop_label]

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("Proses Pemeriksaan Transaksi", width="stretch", type="primary")

    # ── Prediction Logic ──
    if submit_btn:
        with st.spinner('Memeriksa pola transaksi...'):
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
                st.markdown("### Hasil Pemeriksaan")

                with st.container(border=True, key="verdict_card"):
                    stamp_text = "HIGH RISK — REVIEW" if is_fraud else "LOW RISK"
                    stamp_color = pal["stamp_red"] if is_fraud else pal["stamp_green"]
                    st.markdown(f"""
                        <div class="verdict-wrap">
                            <div class="verdict-stamp" style="--stamp-color:{stamp_color};">{stamp_text}</div>
                            <div class="verdict-detail">Probabilitas terukur: {probability:.1%} &nbsp;•&nbsp; Ambang batas keputusan: 15%</div>
                        </div>
                    """, unsafe_allow_html=True)

                    res_col1, res_col2 = st.columns([1, 1])
                    with res_col1:
                        met1, met2 = st.columns(2)
                        met1.metric("Screening Status", "Review Required" if is_fraud else "Low Risk")
                        met2.metric("Inference Time", f"{latency:.1f} ms")

                        flags = get_heuristic_flags(amt, distance_km, hour, category)
                        if flags:
                            with st.expander("Risk Indicators (rule-based)"):
                                st.markdown(
                                    "<ul class='evidence-list'>" + "".join(f"<li>{f}</li>" for f in flags) + "</ul>",
                                    unsafe_allow_html=True
                                )

                    with res_col2:
                        st.plotly_chart(render_risk_gauge(probability, pal), width="stretch")

                    global_importance = get_global_feature_importance()
                    if global_importance is not None:
                        with st.expander("Global Model Feature Importance"):
                            st.caption("Ini adalah *feature importance* global dari Random Forest, bukan penjelasan spesifik untuk transaksi ini.")
                            st.bar_chart(global_importance, color=pal["accent"])

                record = {
                    "Hasil": "Fraud" if is_fraud else "Sah",
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

    # ── Riwayat Sesi (ledger) ──
    st.divider()
    st.markdown("### Log Berkas — Sesi Ini")

    if len(st.session_state.history) == 0:
        st.markdown(
            "<p class='sidebar-note' style='color: var(--ink-muted) !important; border-color: var(--rule);'>"
            "Belum ada transaksi yang diperiksa pada sesi ini.</p>",
            unsafe_allow_html=True
        )
    else:
        df_history = pd.DataFrame(st.session_state.history)
        df_history.index = range(1, len(df_history) + 1)

        total_tx = len(df_history)
        fraud_count = int(df_history["Hasil"].str.contains("Fraud").sum())
        avg_risk = df_history["Risiko"].str.rstrip("%").astype(float).mean()

        sum1, sum2, sum3 = st.columns(3)
        sum1.metric("Total Transaksi Diperiksa", total_tx)
        sum2.metric("Terindikasi Fraud", fraud_count)
        sum3.metric("Rata-rata Risiko", f"{avg_risk:.1f}%")

        def highlight_result(row):
            bg = pal["row_fraud"] if "Fraud" in row["Hasil"] else pal["row_legit"]
            return [f"background-color: {bg}; color: {pal['ink']};"] * len(row)

        styled = (
            df_history.style
            .apply(highlight_result, axis=1)
            .set_properties(**{"font-family": "'IBM Plex Mono', monospace", "font-size": "13px"})
            .set_table_styles([{
                "selector": "th",
                "props": [
                    ("font-family", "'IBM Plex Mono', monospace"),
                    ("background-color", pal["accent"]),
                    ("color", pal["accent_text"]),
                    ("text-transform", "uppercase"),
                    ("font-size", "11px"),
                    ("letter-spacing", "0.05em"),
                ]
            }])
        )
        st.dataframe(styled, width="stretch")

        if st.button("Clear Session History", type="secondary"):
            st.session_state.history = []
            st.rerun()

# ═══════════════════════════════════════════════════════════════
# PAGE 2: CARA KERJA SISTEM
# ═══════════════════════════════════════════════════════════════
elif page == "Cara Kerja Sistem":
    render_case_header(
        "How the AI Works",
        "A concise view of the model, preprocessing, and screening decision."
    )

    st.markdown("""
    ### 1. Latar Belakang
    Di era ekonomi digital, penipuan kartu kredit (*credit card fraud*) menyebabkan kerugian miliaran dolar setiap tahun. Sistem berbasis *machine learning* krusial di industri perbankan karena mampu meninjau ribuan transaksi per detik — jauh lebih cepat daripada investigator manusia.

    ### 2. Tantangan: Data Tidak Seimbang
    Tantangan terbesar proyek ini adalah sifat data yang sangat tidak seimbang (*imbalanced*). Dalam dunia nyata, kasus penipuan berjumlah kurang dari **1%** dari total transaksi harian. Model konvensional akan kesulitan mendeteksi anomali sekecil ini.

    Sistem ini menggunakan algoritma **Random Forest Classifier** dengan rekayasa fitur (*feature engineering*) mendalam, dan ambang batas keputusan (*threshold*) diturunkan menjadi 15% untuk meningkatkan sensitivitas deteksi.

    ### 3. Rekayasa Fitur
    Sistem tidak hanya menganalisis nominal uang, tetapi juga menciptakan variabel prediktif baru:
    - **Jarak Geografis** — jarak fisik (km) antara lokasi nasabah dan merchant.
    - **Fitur Temporal** — jam, hari, dan bulan transaksi, untuk mempelajari pola waktu operasional penipu.
    - **Profil Demografis** — usia pengguna dihitung dari tanggal lahir.
    """)

    st.markdown("""
    <div class="sop-box">
        <div class="sop-header">Prosedur Standar — Alur Pemrosesan</div>
        <ol class="sop-list">
            <li><strong>Input Data</strong> — sistem menangkap parameter transaksi dari formulir.</li>
            <li><strong>Preprocessing</strong> — data teks (kategori, negara bagian) dikonversi menjadi angka (label encoding).</li>
            <li><strong>Standardisasi</strong> — skala angka disetarakan menggunakan StandardScaler.</li>
            <li><strong>Inferensi</strong> — model Random Forest mengambil voting dari seluruh pohon keputusan.</li>
            <li><strong>Keputusan</strong> — jika probabilitas risiko di atas 15%, sistem memicu status waspada.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    global_importance = get_global_feature_importance(top_n=8)
    if global_importance is not None:
        st.markdown("### 4. Fitur Paling Berpengaruh (Global)")
        st.caption("Diambil dari `feature_importances_` model Random Forest — menunjukkan fitur mana yang paling sering dipakai model untuk membedakan transaksi secara keseluruhan (bukan per transaksi).")
        st.bar_chart(global_importance, color=pal["accent"])

# ═══════════════════════════════════════════════════════════════
# PAGE 3: TENTANG KAMI
# ═══════════════════════════════════════════════════════════════
elif page == "Tentang Kami":
    render_case_header(
        "About Fraud Detect AI",
        "A final-project proof of concept for machine-learning-based transaction risk screening."
    )

    st.markdown("Aplikasi ini dikembangkan sebagai bagian dari tugas proyek akhir / *proof of concept*.")

    team = [
        "Felix Zonattan",
        "Jason Benoit Adianto",
        "Keivan Aliegery Indriartho",
        "Ivander Sanusi",
        "Haposan Emmanuel Tobias",
    ]
    rows = "".join(
        f'<div class="roster-row"><span class="roster-id">{i+1:02d}</span><span class="roster-name">{name}</span></div>'
        for i, name in enumerate(team)
    )
    st.markdown(f"<div style='margin-top:14px;'>{rows}</div>", unsafe_allow_html=True)
    st.markdown('<div class="file-footer">Hak Cipta &copy; 2026</div>', unsafe_allow_html=True)
