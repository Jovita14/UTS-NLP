# -*- coding: utf-8 -*-
"""Kelp2_app - Redesigned with modern, premium UI"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
import re
from sklearn.preprocessing import MultiLabelBinarizer

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ABSA Analytics Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap" rel="stylesheet">

<style>
/* ── ROOT VARIABLES ── */
:root {
    --bg:        #0b0f1a;
    --surface:   #111827;
    --surface2:  #1a2235;
    --border:    #1e2d45;
    --accent:    #00e5c3;
    --accent2:   #7b61ff;
    --accent3:   #ff6b6b;
    --text:      #e8edf5;
    --muted:     #8899aa;
    --gold:      #f5c842;
    --radius:    14px;
}

/* ── BASE ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1421 0%, #111827 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── SIDEBAR HEADER ── */
.sidebar-brand {
    padding: 1.5rem 0 1rem;
    text-align: center;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.sidebar-brand .logo {
    font-size: 2.8rem;
    line-height: 1;
}
.sidebar-brand h2 {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 800 !important;
    letter-spacing: .12em;
    color: var(--accent) !important;
    margin: .4rem 0 .1rem !important;
    text-transform: uppercase;
}
.sidebar-brand p {
    font-size: .72rem;
    color: var(--muted) !important;
    letter-spacing: .06em;
    text-transform: uppercase;
}

/* ── SIDEBAR NAV ── */
[data-testid="stRadio"] label,
[data-testid="stSelectbox"] label {
    font-family: 'Syne', sans-serif !important;
    font-size: .72rem !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    font-weight: 600 !important;
}

/* ── HERO HEADER ── */
.hero {
    background: linear-gradient(135deg, #0d1f35 0%, #0b1628 50%, #0f1e30 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2.4rem 2.8rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(0,229,195,.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 20%;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(123,97,255,.10) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    display: inline-block;
    background: rgba(0,229,195,.12);
    border: 1px solid rgba(0,229,195,.3);
    color: var(--accent) !important;
    font-size: .68rem;
    letter-spacing: .14em;
    text-transform: uppercase;
    padding: .25rem .8rem;
    border-radius: 99px;
    font-weight: 600;
    margin-bottom: .9rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    color: var(--text) !important;
    margin: 0 0 .5rem !important;
    line-height: 1.2 !important;
}
.hero h1 span { color: var(--accent); }
.hero p {
    color: var(--muted) !important;
    font-size: .95rem;
    margin: 0 !important;
    max-width: 520px;
}

/* ── STAT CARDS ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.8rem;
}
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.3rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color .2s;
}
.stat-card:hover { border-color: var(--accent); }
.stat-card .accent-bar {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.stat-card .s-icon {
    font-size: 1.4rem;
    margin-bottom: .5rem;
}
.stat-card .s-val {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--text);
    line-height: 1;
    margin-bottom: .25rem;
}
.stat-card .s-label {
    font-size: .72rem;
    color: var(--muted);
    letter-spacing: .08em;
    text-transform: uppercase;
}

/* ── PANEL ── */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
}
.panel-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: var(--text) !important;
    margin-bottom: 1.2rem !important;
    display: flex;
    align-items: center;
    gap: .5rem;
}
.panel-title span { color: var(--accent); }

/* ── INSIGHT BOX ── */
.insight-box {
    background: linear-gradient(135deg, rgba(0,229,195,.06), rgba(123,97,255,.06));
    border: 1px solid rgba(0,229,195,.18);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-top: 1.4rem;
}
.insight-box .i-title {
    font-family: 'Syne', sans-serif;
    font-size: .78rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 700;
    margin-bottom: .7rem;
}
.insight-box ul {
    margin: 0; padding-left: 1.1rem;
    color: var(--muted);
    font-size: .88rem;
    line-height: 1.7;
}
.insight-box ul li::marker { color: var(--accent); }
.insight-box .warn {
    margin-top: .8rem;
    padding-top: .8rem;
    border-top: 1px solid rgba(255,107,107,.15);
    color: var(--accent3);
    font-size: .82rem;
    font-weight: 500;
}

/* ── BADGE ── */
.badge {
    display: inline-block;
    padding: .2rem .7rem;
    border-radius: 99px;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .06em;
}
.badge-green  { background: rgba(0,229,195,.15); color: var(--accent); border: 1px solid rgba(0,229,195,.3); }
.badge-purple { background: rgba(123,97,255,.15); color: var(--accent2); border: 1px solid rgba(123,97,255,.3); }
.badge-red    { background: rgba(255,107,107,.15); color: var(--accent3); border: 1px solid rgba(255,107,107,.3); }

/* ── NER ENTITY ── */
.ner-wrap {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    font-size: 1.05rem;
    line-height: 2.2;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}
.legend-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: .6rem;
    margin-top: 1rem;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: .4rem;
    font-size: .8rem;
    color: var(--muted);
}
.legend-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── UPLOAD ZONE ── */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
    transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploader"] * { color: var(--text) !important; }

/* ── SELECTBOX / RADIO ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stRadio"] > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* ── SLIDER ── */
[data-testid="stSlider"] * { color: var(--text) !important; }

/* ── SUCCESS / INFO ── */
[data-testid="stAlert"] {
    background: rgba(0,229,195,.08) !important;
    border: 1px solid rgba(0,229,195,.25) !important;
    border-radius: 10px !important;
    color: var(--accent) !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── SECTION DIVIDER ── */
.section-divider {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 2.5rem 0 1.8rem;
}
.section-divider .sd-line { flex: 1; height: 1px; background: var(--border); }
.section-divider .sd-label {
    font-family: 'Syne', sans-serif;
    font-size: .72rem;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: var(--muted);
    white-space: nowrap;
}

/* ── MATPLOTLIB OVERRIDE ── */
.stPlot { border-radius: 10px; overflow: hidden; }

/* ── HIDE DEFAULT ELEMENTS ── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
DARK_BG   = "#111827"
DARK_AX   = "#1a2235"
ACCENT    = "#00e5c3"
ACCENT2   = "#7b61ff"
ACCENT3   = "#ff6b6b"
GOLD      = "#f5c842"
TEXT_COL  = "#e8edf5"
MUTED_COL = "#8899aa"

PALETTE = [ACCENT, ACCENT2, GOLD, ACCENT3, "#ff9f43", "#54a0ff", "#5f27cd", "#01abc5"]

def apply_dark_style(fig, ax):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    ax.tick_params(colors=MUTED_COL, labelsize=9)
    ax.xaxis.label.set_color(MUTED_COL)
    ax.yaxis.label.set_color(MUTED_COL)
    ax.title.set_color(TEXT_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d45")
    ax.grid(True, color="#1e2d45", linewidth=.7, linestyle='--', alpha=.6)
    ax.set_axisbelow(True)
    return fig, ax

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="logo">🌿</div>
        <h2>ABSA Lab</h2>
        <p>Analytics Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    menu = st.selectbox(
        "PILIH ANALISIS",
        ["🏠  Overview", "📊  Distribusi Label", "🏷️  Distribusi Entitas",
         "📏  Panjang Review", "🔗  Korelasi Label", "🔍  NER Viewer"],
        key="nav_menu"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="border-top:1px solid #1e2d45; padding-top:1.2rem;">
        <p style="font-size:.68rem;color:#8899aa;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.5rem;">Upload Data</p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">Aspect-Based Sentiment Analysis</div>
    <h1>EDA & IRR <span>Dashboard</span></h1>
    <p>Eksplorasi distribusi label, entitas, korelasi, dan kesepakatan antar-anotator (IRR) secara interaktif.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILE UPLOAD (main area)
# ─────────────────────────────────────────────
col_up1, col_up2 = st.columns(2)
with col_up1:
    st.markdown('<p style="font-family:Syne;font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:#8899aa;margin-bottom:.4rem;">Dataset Utama</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload file JSONL", type=["jsonl"], key="main_file", label_visibility="collapsed")
with col_up2:
    st.markdown('<p style="font-family:Syne;font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:#8899aa;margin-bottom:.4rem;">IRR Data</p>', unsafe_allow_html=True)
    irr_file = st.file_uploader("Upload IRR JSONL", type=["jsonl"], key="irr_file", label_visibility="collapsed")

# ─────────────────────────────────────────────
# PROCESS MAIN DATA
# ─────────────────────────────────────────────
df = None
if uploaded_file:
    data, errors = [], 0
    for line in uploaded_file.read().decode("utf-8").splitlines():
        try:
            data.append(json.loads(line))
        except:
            errors += 1
    df = pd.DataFrame(data)

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    df['clean_text'] = df['text'].apply(clean_text)
    df['length'] = df['clean_text'].apply(lambda x: len(x.split()))

    # Compute stats
    total_rows = len(df)
    total_errors = errors
    all_labels = df['accept'].dropna().explode()
    n_labels = all_labels.nunique()
    avg_len = int(df['length'].mean())

    # ── STAT CARDS ──
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="accent-bar" style="background:linear-gradient(90deg,#00e5c3,#7b61ff)"></div>
            <div class="s-icon">📄</div>
            <div class="s-val">{total_rows:,}</div>
            <div class="s-label">Total Review</div>
        </div>
        <div class="stat-card">
            <div class="accent-bar" style="background:linear-gradient(90deg,#7b61ff,#ff6b6b)"></div>
            <div class="s-icon">🏷️</div>
            <div class="s-val">{n_labels}</div>
            <div class="s-label">Unique Labels</div>
        </div>
        <div class="stat-card">
            <div class="accent-bar" style="background:linear-gradient(90deg,#f5c842,#00e5c3)"></div>
            <div class="s-icon">📏</div>
            <div class="s-val">{avg_len}</div>
            <div class="s-label">Avg. Panjang (kata)</div>
        </div>
        <div class="stat-card">
            <div class="accent-bar" style="background:linear-gradient(90deg,#ff6b6b,#f5c842)"></div>
            <div class="s-icon">⚠️</div>
            <div class="s-val">{total_errors}</div>
            <div class="s-label">Parse Error</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.success(f"✅  Dataset berhasil dimuat — {total_rows} baris, {total_errors} error.")

    # ── CONTENT BY MENU ──
    clean_menu = menu.split("  ")[-1] if "  " in menu else menu

    # ── OVERVIEW ──
    if "Overview" in menu:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📄 <span>Preview</span> Dataset</div>', unsafe_allow_html=True)
        st.dataframe(
            df[['text', 'accept']].head(50).style.set_properties(**{
                'background-color': '#1a2235',
                'color': '#e8edf5',
                'border-color': '#1e2d45'
            }),
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── DISTRIBUSI LABEL ──
    elif "Distribusi Label" in menu:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📊 <span>Distribusi</span> Label ABSA</div>', unsafe_allow_html=True)

        label_counts = all_labels.value_counts().sort_values(ascending=False)
        n = len(label_counts)
        bar_colors = [PALETTE[i % len(PALETTE)] for i in range(n)]

        fig, ax = plt.subplots(figsize=(10, 4.5))
        bars = ax.bar(label_counts.index, label_counts.values, color=bar_colors, width=.6, zorder=3)
        fig, ax = apply_dark_style(fig, ax)
        ax.set_xlabel("Label", fontsize=9, color=MUTED_COL)
        ax.set_ylabel("Jumlah", fontsize=9, color=MUTED_COL)
        plt.xticks(rotation=30, ha='right', fontsize=8)

        # value labels on bars
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + .5, str(int(h)),
                    ha='center', va='bottom', fontsize=8, color=TEXT_COL, fontweight='bold')

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        <div class="insight-box">
            <div class="i-title">💡 Insight</div>
            <ul>
                <li>Dataset hasil cleaning tersisa <strong style="color:#e8edf5">678 review</strong> — di bawah standar 800, dipertahankan demi kualitas anotasi.</li>
                <li>Label dominan: <span class="badge badge-green">PRODUCT_POSITIVE</span></li>
                <li>Diikuti oleh: <span class="badge badge-purple">PLACE_POSITIVE</span> dan <span class="badge badge-green">PRICE_POSITIVE</span></li>
                <li>Label negatif sangat sedikit dalam dataset.</li>
            </ul>
            <div class="warn">⚡ Dataset <strong>tidak seimbang (class imbalance)</strong> — model berpotensi bias ke sentimen positif.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── DISTRIBUSI ENTITAS ──
    elif "Distribusi Entitas" in menu:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🏷️ <span>Distribusi</span> Entitas NER</div>', unsafe_allow_html=True)

        entities = []
        for spans in df['spans'].dropna():
            if isinstance(spans, list):
                for s in spans:
                    if 'label' in s:
                        entities.append(s['label'])

        entity_counts = pd.Series(entities).value_counts()
        n = len(entity_counts)
        bar_colors = [PALETTE[i % len(PALETTE)] for i in range(n)]

        fig, ax = plt.subplots(figsize=(10, 4.5))
        bars = ax.bar(entity_counts.index, entity_counts.values, color=bar_colors, width=.6, zorder=3)
        fig, ax = apply_dark_style(fig, ax)
        ax.set_xlabel("Entitas", fontsize=9, color=MUTED_COL)
        ax.set_ylabel("Frekuensi", fontsize=9, color=MUTED_COL)
        plt.xticks(rotation=30, ha='right', fontsize=8)

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + .5, str(int(h)),
                    ha='center', va='bottom', fontsize=8, color=TEXT_COL, fontweight='bold')

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        <div class="insight-box">
            <div class="i-title">💡 Insight</div>
            <ul>
                <li>Entitas paling sering muncul: <span class="badge badge-green">PRODUCT_POSITIVE</span></li>
                <li>Diikuti: <span class="badge badge-purple">PLACE_POSITIVE</span>, <span class="badge badge-green">PRICE_POSITIVE</span></li>
                <li>Entitas lain jauh lebih sedikit, distribusi sangat tidak merata.</li>
            </ul>
            <div class="warn">⚡ Distribusi tidak merata → model kesulitan belajar entitas minor.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── PANJANG REVIEW ──
    elif "Panjang Review" in menu:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📏 <span>Distribusi</span> Panjang Review</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig, ax = plt.subplots(figsize=(8, 4))
            n_data, bins, patches = ax.hist(df['length'], bins=30, color=ACCENT, alpha=.85, zorder=3, edgecolor=DARK_BG)
            # gradient color
            for i, patch in enumerate(patches):
                frac = i / len(patches)
                r = int(0 + frac * (123))
                g = int(229 - frac * (229-97))
                b = int(195 - frac * (195-255))
                patch.set_facecolor(f"#{r:02x}{g:02x}{b:02x}")

            ax.axvline(df['length'].mean(), color=GOLD, linewidth=1.5, linestyle='--', label=f"Rata-rata: {df['length'].mean():.1f}")
            ax.axvline(df['length'].median(), color=ACCENT3, linewidth=1.5, linestyle=':', label=f"Median: {df['length'].median():.1f}")
            ax.legend(fontsize=8, labelcolor=TEXT_COL, facecolor=DARK_AX, edgecolor=MUTED_COL)

            fig, ax = apply_dark_style(fig, ax)
            ax.set_xlabel("Jumlah Kata", fontsize=9)
            ax.set_ylabel("Frekuensi", fontsize=9)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_b:
            stats = df['length'].describe()
            st.markdown(f"""
            <div style="background:#1a2235;border:1px solid #1e2d45;border-radius:10px;padding:1.2rem;">
                <p style="font-family:Syne;font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#8899aa;margin-bottom:1rem;">Statistik</p>
                {''.join([f'<div style="display:flex;justify-content:space-between;padding:.4rem 0;border-bottom:1px solid #1e2d45;"><span style="color:#8899aa;font-size:.85rem">{k.capitalize()}</span><span style="color:#e8edf5;font-weight:600;font-size:.85rem">{v:.1f}</span></div>' for k, v in stats.items()])}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
            <div class="i-title">💡 Insight</div>
            <ul>
                <li>Mayoritas review memiliki panjang pendek–sedang.</li>
                <li>Terdapat sedikit review yang sangat panjang (outlier).</li>
                <li>Distribusi cocok untuk model NLP berbasis transformer standar.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── KORELASI LABEL ──
    elif "Korelasi Label" in menu:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🔗 <span>Korelasi</span> Antar Label</div>', unsafe_allow_html=True)

        mlb = MultiLabelBinarizer()
        label_matrix = mlb.fit_transform(df['accept'])
        df_label = pd.DataFrame(label_matrix, columns=mlb.classes_)
        corr = df_label.corr()

        fig, ax = plt.subplots(figsize=(11, 7))
        sns.heatmap(
            corr, annot=True, fmt=".2f",
            cmap=sns.diverging_palette(260, 160, s=90, l=45, n=256),
            center=0, linewidths=.5, linecolor=DARK_BG,
            annot_kws={"size": 8, "color": TEXT_COL},
            ax=ax,
            cbar_kws={"shrink": .7}
        )
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_AX)
        ax.tick_params(colors=MUTED_COL, labelsize=8)
        plt.xticks(rotation=35, ha='right')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        <div class="insight-box">
            <div class="i-title">💡 Insight</div>
            <ul>
                <li>Korelasi positif kuat antara label satu aspek dengan sentimen yang sama.</li>
                <li>Label lintas aspek cenderung berkorelasi rendah.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── NER VIEWER ──
    elif "NER Viewer" in menu:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🔍 <span>NER</span> Viewer</div>', unsafe_allow_html=True)

        color_map = {
            "PRODUCT":   "#00e5c3",
            "PRICE":     "#2ecc71",
            "PLACE":     "#ff9f43",
            "PROMOTION": "#7b61ff",
            "POSITIVE":  "#54a0ff",
            "NEGATIVE":  "#ff6b6b",
            "NEUTRAL":   "#8899aa",
        }

        def get_color(label):
            for key in color_map:
                if key in label:
                    return color_map[key]
            return "#8899aa"

        idx = st.slider("Pilih Indeks Data", 0, len(df)-1, 0)

        text  = df.iloc[idx]['text']
        spans = df.iloc[idx]['spans']

        col_v1, col_v2 = st.columns([3, 1])
        with col_v1:
            st.markdown('<p style="font-family:Syne;font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:#8899aa;margin-bottom:.5rem;">Teks Asli</p>', unsafe_allow_html=True)
            colored_text = text
            if isinstance(spans, list):
                for span in sorted(spans, key=lambda x: x['start'], reverse=True):
                    s, e, lbl = span['start'], span['end'], span['label']
                    c = get_color(lbl)
                    colored_text = (
                        colored_text[:s]
                        + f"<mark style='background:{c}22;color:{c};border:1px solid {c}44;padding:1px 6px;border-radius:5px;font-weight:600'>"
                        + f"{text[s:e]}<sup style='font-size:.6rem;margin-left:3px;opacity:.8'>{lbl}</sup></mark>"
                        + colored_text[e:]
                    )
            st.markdown(f'<div class="ner-wrap">{colored_text}</div>', unsafe_allow_html=True)

        with col_v2:
            st.markdown('<p style="font-family:Syne;font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:#8899aa;margin-bottom:.5rem;">Entitas</p>', unsafe_allow_html=True)
            if isinstance(spans, list):
                for span in spans:
                    c = get_color(span['label'])
                    st.markdown(f"""
                    <div style="background:{c}15;border-left:3px solid {c};border-radius:6px;padding:.5rem .8rem;margin-bottom:.5rem">
                        <div style="font-size:.72rem;color:{c};font-weight:700;letter-spacing:.06em">{span['label']}</div>
                        <div style="font-size:.88rem;color:#e8edf5;margin-top:.2rem">"{text[span['start']:span['end']]}"</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Legend
        st.markdown('<div class="legend-wrap">', unsafe_allow_html=True)
        for key, val in color_map.items():
            st.markdown(f"""
            <div class="legend-item">
                <div class="legend-dot" style="background:{val}"></div>
                <span>{key}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# IRR SECTION
# ─────────────────────────────────────────────
st.markdown("""
<div class="section-divider">
    <div class="sd-line"></div>
    <div class="sd-label">Inter-Annotator Agreement</div>
    <div class="sd-line"></div>
</div>
""", unsafe_allow_html=True)

if irr_file:
    irr_data = []
    for line in irr_file.read().decode("utf-8").splitlines():
        try:
            irr_data.append(json.loads(line))
        except:
            pass

    df_irr = pd.DataFrame(irr_data).T
    cols_to_show = [c for c in ["kripp_alpha", "percent_agreement", "gwet_ac2"] if c in df_irr.columns]
    if cols_to_show:
        df_irr_summary = df_irr[cols_to_show]

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📈 <span>Hasil</span> IRR</div>', unsafe_allow_html=True)
        st.dataframe(df_irr_summary.style.set_properties(**{
            'background-color': '#1a2235',
            'color': '#e8edf5',
            'border-color': '#1e2d45'
        }), use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            <div class="i-title">💡 Insight IRR</div>
            <ul>
                <li>Label <strong style="color:#e8edf5">positif</strong> → agreement tinggi antar anotator.</li>
                <li>Label <strong style="color:#e8edf5">netral &amp; negatif</strong> → agreement rendah, terdapat inkonsistensi.</li>
                <li>Terdapat perbedaan interpretasi antar anotator pada kasus ambigu.</li>
            </ul>
            <div class="warn">⚡ Perlu perbaikan <strong>annotation guideline</strong> untuk meningkatkan konsistensi.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="background:#111827;border:2px dashed #1e2d45;border-radius:14px;padding:2rem;text-align:center;color:#8899aa;">
        <div style="font-size:2rem;margin-bottom:.5rem">📂</div>
        <p style="font-family:Syne;font-size:.78rem;letter-spacing:.1em;text-transform:uppercase;margin:0">
            Upload file IRR JSONL di atas untuk melihat hasil agreement
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:3rem 0 1.5rem;border-top:1px solid #1e2d45;margin-top:3rem">
    <p style="font-family:Syne;font-size:.68rem;letter-spacing:.14em;text-transform:uppercase;color:#8899aa;margin:0">
        🌿 ABSA Analytics Dashboard &nbsp;·&nbsp; Aspect-Based Sentiment Analysis
    </p>
</div>
""", unsafe_allow_html=True)