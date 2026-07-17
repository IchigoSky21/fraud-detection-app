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
    page_title="Fraud Detect AI — Berkas Kasus",
    page_icon="🗂️",
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
        "paper": "#E8E1CE", "paper_light": "#F4EFE1",
        "ink": "#2B2620", "ink_muted": "#6b6046",
        "accent": "#24344D", "accent_dark": "#17233A", "accent_text": "#F4EFE1",
        "sidebar_bg": "#24344D", "sidebar_border": "#17233A", "on_dark": "#F4EFE1",
        "tape": "#D8CBA0",
        "stamp_red": "#9C2B1B", "stamp_green": "#33512E", "stamp_amber": "#A6740A",
        "rule": "#B9AD8F",
        "row_fraud": "#EAD6D0", "row_legit": "#DFE6D6",
        "gauge_steps": ["#E3E6D9", "#EDE2C4", "#E9D4CD"],
        "stamp_blend": "multiply", "stamp_opacity": "0.9",
    },
    "dark": {
        "paper": "#1C1712", "paper_light": "#262019",
        "ink": "#E7DFC9", "ink_muted": "#A89A78",
        "accent": "#C9A24A", "accent_dark": "#8A6C2A", "accent_text": "#1C1712",
        "sidebar_bg": "#14100C", "sidebar_border": "#000000", "on_dark": "#F4EFE1",
        "tape": "#9C8D6C",
        "stamp_red": "#E0674B", "stamp_green": "#8FBB7A", "stamp_amber": "#E3BA55",
        "rule": "#45392A",
        "row_fraud": "#3A2620", "row_legit": "#263019",
        "gauge_steps": ["#2A2E22", "#33301C", "#332420"],
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
@import url('https://fonts.googleapis.com/css2?family=Zilla+Slab:ital,wght@0,400;0,600;0,700;1,400&family=Special+Elite&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {{
    --paper: {pal['paper']};
    --paper-light: {pal['paper_light']};
    --ink: {pal['ink']};
    --ink-muted: {pal['ink_muted']};
    --accent: {pal['accent']};
    --accent-dark: {pal['accent_dark']};
    --accent-text: {pal['accent_text']};
    --on-dark: {pal['on_dark']};
    --tape: {pal['tape']};
    --stamp-red: {pal['stamp_red']};
    --stamp-green: {pal['stamp_green']};
    --stamp-amber: {pal['stamp_amber']};
    --rule: {pal['rule']};
}}

/* ── Base ── */
[data-testid="stAppViewContainer"], .stApp {{ background-color: var(--paper); }}
[data-testid="stHeader"] {{ background-color: var(--paper); }}
html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; color: var(--ink); }}
h1, h2, h3 {{ font-family: 'Zilla Slab', Georgia, serif !important; color: var(--accent) !important; }}
hr {{ border-color: var(--rule) !important; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: var(--ink-muted) !important; }}

/* ── Sidebar = folder spine ── */
/* NOTE: we deliberately do NOT use a blanket `* {{ color: ... !important }}`
   here. It previously clobbered streamlit-option-menu's own inline text/icon
   colors on the selected tab (measured contrast 1.41:1 — effectively
   invisible). Instead we target only the elements that actually need a
   default color, and leave the nav's own colors alone. */
[data-testid="stSidebar"] {{ background-color: {pal['sidebar_bg']}; border-right: 1px solid {pal['sidebar_border']}; }}
[data-testid="stSidebar"] h3 {{ color: var(--on-dark) !important; }}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p:not(.sidebar-note) {{
    color: var(--on-dark) !important;
}}
[data-testid="stSidebar"] hr {{ border-color: rgba(244,239,225,0.2) !important; }}
/* Force the ACTIVE tab's text/icon to the accent color explicitly, so it
   always reads against its own light selected background regardless of
   which mode is active. */
[data-testid="stSidebar"] a.nav-link.active,
[data-testid="stSidebar"] a.nav-link.active span,
[data-testid="stSidebar"] li.active a {{ color: var(--accent) !important; }}
[data-testid="stSidebar"] a.nav-link.active i,
[data-testid="stSidebar"] a.nav-link.active svg {{ color: var(--accent) !important; fill: var(--accent) !important; }}
.svg-wrap svg {{ display: block; }}

/* ── Dropdown / select popovers (BaseWeb renders these via a portal, but
   they still sit under the same document root, so :root variables reach
   them even though they're outside the normal sidebar/main DOM branch). ── */
[data-baseweb="popover"] [data-baseweb="menu"],
ul[role="listbox"] {{ background-color: var(--paper-light) !important; }}
[data-baseweb="menu"] li, [role="option"] {{ color: var(--ink) !important; }}

/* ── Stamped status block (sidebar) ── */
.stamp-block {{
    border: 1.5px solid var(--tape);
    background-color: rgba(244,239,225,0.06);
    padding: 12px 14px;
    margin: 10px 0 14px 0;
    transform: rotate(-0.4deg);
}}
.stamp-row {{
    display: flex; justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace; font-size: 11.5px;
    padding: 3px 0; border-bottom: 1px dotted rgba(244,239,225,0.25);
}}
.stamp-row:last-child {{ border-bottom: none; }}
.stamp-label {{ letter-spacing: 0.08em; color: var(--tape) !important; }}
.stamp-value {{ font-weight: 600; color: var(--on-dark) !important; }}
.sidebar-note {{
    font-size: 12px; line-height: 1.5; color: var(--tape) !important;
    border-left: 2px solid var(--stamp-red); padding-left: 10px;
}}

/* ── Case file letterhead header ── */
.case-header {{
    background-color: var(--paper-light);
    border: 1px solid var(--rule);
    border-left: 6px solid var(--accent);
    padding: 20px 26px 22px 26px;
    margin-bottom: 22px;
}}
.case-title {{
    font-family: 'Zilla Slab', Georgia, serif !important;
    font-weight: 700; font-size: 2.1rem;
    color: var(--accent) !important;
    margin: 0 0 6px 0; line-height: 1.15;
}}
.case-subtitle {{
    font-family: 'IBM Plex Sans', sans-serif; font-style: italic;
    color: var(--ink-muted); font-size: 0.98rem; margin: 0;
}}

/* ── Form section tags (lettered — a real intake form has real sections) ── */
.section-tag {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
    background-color: var(--accent); color: var(--accent-text);
    padding: 3px 9px; margin-bottom: 8px;
}}

/* ── Buttons ── */
.stButton button, [data-testid="stFormSubmitButton"] button {{
    background-color: var(--accent) !important;
    color: var(--accent-text) !important;
    border: 1px solid var(--accent-dark) !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.06em; text-transform: uppercase; font-size: 13px !important;
}}
.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {{
    background-color: var(--stamp-red) !important; border-color: var(--stamp-red) !important;
}}

/* ── Inputs ── */
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input,
[data-baseweb="select"] > div {{
    background-color: var(--paper-light) !important;
    border-color: var(--rule) !important;
    border-radius: 2px !important;
    color: var(--ink) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}}

/* ── Metrics ── */
[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace !important; color: var(--accent) !important; }}
[data-testid="stMetricLabel"] {{
    font-family: 'IBM Plex Sans', sans-serif !important;
    text-transform: uppercase; font-size: 11.5px !important;
    letter-spacing: 0.05em; color: var(--ink-muted) !important;
}}

/* ── Bordered containers (result card) ── */
.st-key-verdict_card {{
    background-color: var(--paper-light) !important;
    border: 1px solid var(--rule) !important;
    border-radius: 2px !important;
}}

/* ── The signature element: an ink verdict stamp ── */
.verdict-stamp {{
    display: inline-block;
    font-family: 'Special Elite', 'IBM Plex Mono', monospace;
    font-size: 1.55rem; letter-spacing: 0.06em; text-transform: uppercase;
    padding: 10px 24px;
    border: 4px double var(--stamp-color);
    color: var(--stamp-color);
    transform: rotate(-6deg);
    opacity: {pal['stamp_opacity']};
    mix-blend-mode: {pal['stamp_blend']};
    margin-bottom: 10px;
}}
.verdict-detail {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: var(--ink-muted); margin-top: 2px; }}

/* ── Evidence / heuristic list ── */
.evidence-list {{ list-style: none; padding-left: 0; margin: 6px 0; }}
.evidence-list li {{ position: relative; padding: 7px 0 7px 24px; border-bottom: 1px dashed var(--rule); font-size: 0.92rem; }}
.evidence-list li:before {{ content: "§"; position: absolute; left: 2px; color: var(--stamp-amber); font-weight: 700; }}

/* ── SOP / procedure box ── */
.sop-box {{
    border: 1px solid var(--rule); border-left: 6px solid var(--accent);
    background-color: var(--paper-light); padding: 18px 24px; margin: 14px 0 20px 0;
}}
.sop-header {{
    font-family: 'IBM Plex Mono', monospace; text-transform: uppercase;
    letter-spacing: 0.08em; font-size: 12px; color: var(--accent); margin-bottom: 10px;
}}
.sop-list {{ margin: 0; padding-left: 20px; }}
.sop-list li {{ margin-bottom: 9px; line-height: 1.5; }}

/* ── Personnel roster (Tentang Kami) ── */
.roster-row {{ display: flex; align-items: baseline; gap: 14px; padding: 9px 2px; border-bottom: 1px solid var(--rule); }}
.roster-id {{ font-family: 'IBM Plex Mono', monospace; color: var(--stamp-red); font-size: 13px; width: 34px; }}
.roster-name {{ font-family: 'IBM Plex Sans', sans-serif; font-weight: 500; }}
.file-footer {{
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--ink-muted);
    text-align: right; margin-top: 18px; letter-spacing: 0.05em;
}}

/* ── Overflow safety for rotated elements ── */
.verdict-wrap {{ overflow: hidden; max-width: 100%; padding: 2px 0; }}

/* ── Mobile (phones / narrow viewports) ── */
@media (max-width: 640px) {{
    .case-header {{ padding: 14px 16px 16px 16px; }}
    .case-title {{ font-size: 1.4rem; }}
    .case-subtitle {{ font-size: 0.88rem; }}
    .verdict-stamp {{
        font-size: 1.1rem;
        padding: 7px 14px;
        transform: rotate(-4deg);
    }}
    .stamp-row {{ flex-wrap: wrap; font-size: 11px; }}
    .roster-row {{ flex-wrap: wrap; gap: 6px 14px; }}
    .sop-box, .case-header {{ margin-left: 0; margin-right: 0; }}
}}
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
        "Pemeriksaan Transaksi",
        "Masukkan detail transaksi untuk mendapatkan analisis risiko secara real-time."
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
                    stamp_text = "Terindikasi Fraud" if is_fraud else "Dinyatakan Sah"
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
                        met1.metric("Keputusan Sistem", "Blokir / Waspada" if is_fraud else "Setujui")
                        met2.metric("Waktu Inferensi", f"{latency:.1f} ms")

                        flags = get_heuristic_flags(amt, distance_km, hour, category)
                        if flags:
                            with st.expander("Indikator umum pada input Anda (heuristik)"):
                                st.markdown(
                                    "<ul class='evidence-list'>" + "".join(f"<li>{f}</li>" for f in flags) + "</ul>",
                                    unsafe_allow_html=True
                                )

                    with res_col2:
                        st.plotly_chart(render_risk_gauge(probability, pal), width="stretch")

                    global_importance = get_global_feature_importance()
                    if global_importance is not None:
                        with st.expander("Fitur yang paling berpengaruh secara umum pada model ini"):
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

        if st.button("Hapus Log Sesi", type="secondary"):
            st.session_state.history = []
            st.rerun()

# ═══════════════════════════════════════════════════════════════
# PAGE 2: CARA KERJA SISTEM
# ═══════════════════════════════════════════════════════════════
elif page == "Cara Kerja Sistem":
    render_case_header(
        "Metodologi Sistem",
        "Ringkasan teknis mengenai cara sistem ini mengambil keputusan."
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
        "Tim Pengembang",
        "Purwarupa proyek akhir untuk mendemonstrasikan penerapan machine learning di industri finansial."
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
