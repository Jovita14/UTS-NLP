# -*- coding: utf-8 -*-
"""ABSA EDA & IRR Dashboard - fixed UI and robust upload handling."""

import json
import re
from html import escape

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MultiLabelBinarizer

st.set_page_config(
    page_title="ABSA Analytics Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling: high contrast, cleaner layout
# -----------------------------
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#07111f; --panel:#101c2f; --panel2:#14233a; --panel3:#0d1829;
  --border:#2b3f5f; --text:#f4f7fb; --muted:#b8c4d6; --soft:#d9e3f2;
  --accent:#18e0c4; --accent2:#60a5fa; --warn:#f59e0b; --bad:#fb7185; --good:#22c55e;
  --radius:18px;
}
html, body, [data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:Inter,system-ui,sans-serif!important;}
.block-container{padding-top:1.4rem!important;max-width:1280px!important;}
#MainMenu, footer, header, [data-testid="stDecoration"]{visibility:hidden!important;display:none!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#091527,#101c2f)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
.sidebar-brand{padding:1.4rem 0 1.2rem;text-align:center;border-bottom:1px solid var(--border);margin-bottom:1rem;}
.sidebar-brand .logo{font-size:2.4rem}.sidebar-brand h2{font-size:1.35rem!important;letter-spacing:.14em;color:var(--accent)!important;margin:.35rem 0 0!important;font-weight:800!important}.sidebar-brand p{font-size:.78rem;color:var(--muted)!important;letter-spacing:.08em;text-transform:uppercase}
.hero{background:radial-gradient(circle at 85% 5%,rgba(24,224,196,.20),transparent 30%),linear-gradient(135deg,#10213a,#0b1728);border:1px solid var(--border);border-radius:var(--radius);padding:1.7rem 2rem;margin-bottom:1.2rem;box-shadow:0 12px 32px rgba(0,0,0,.20)}
.hero h1{font-size:2.05rem!important;line-height:1.15!important;margin:0 0 .45rem!important;color:var(--text)!important;font-weight:800!important}.hero span{color:var(--accent)}.hero p{color:var(--soft)!important;font-size:1rem!important;margin:0!important;max-width:760px}.tag{display:inline-flex;align-items:center;gap:.4rem;background:rgba(24,224,196,.14);border:1px solid rgba(24,224,196,.35);color:#9ffbed!important;border-radius:999px;padding:.3rem .75rem;font-weight:700;font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;margin-bottom:.75rem}
.upload-panel,.panel{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);padding:1.2rem 1.35rem;margin-bottom:1.15rem;box-shadow:0 10px 24px rgba(0,0,0,.18)}
.panel-title{font-size:1.05rem!important;font-weight:800!important;letter-spacing:.05em;text-transform:uppercase;color:var(--text)!important;margin:0 0 1rem!important}.panel-title b{color:var(--accent)}
.small-label{font-size:.78rem;color:var(--muted);font-weight:800;letter-spacing:.08em;text-transform:uppercase;margin-bottom:.45rem}
[data-testid="stFileUploader"]{background:var(--panel2)!important;border:1.5px dashed var(--border)!important;border-radius:14px!important;padding:.8rem!important}
[data-testid="stFileUploader"]:hover{border-color:var(--accent)!important;background:#182944!important}
[data-testid="stFileUploader"] *{color:var(--text)!important;opacity:1!important}.uploadedFile{background:#1b2d48!important;color:var(--text)!important;border:1px solid #36577d!important;border-radius:12px!important}.uploadedFile *{color:var(--text)!important;opacity:1!important}.uploadedFile [data-testid='stFileUploaderFileName']{color:#eaf2ff!important}.uploadedFile [data-testid='stFileUploaderFileSize']{color:#b8c4d6!important}
[data-testid="stSelectbox"] label,[data-testid="stRadio"] label,[data-testid="stSlider"] label{color:var(--soft)!important;font-weight:700!important}
[data-testid="stSelectbox"] div,[data-testid="stSlider"] *{color:var(--text)!important}.stSelectbox [data-baseweb="select"]>div{background:var(--panel2)!important;border-color:var(--border)!important;color:var(--text)!important}
.stat-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1rem;margin:1rem 0 1.2rem}.stat{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--border);border-radius:16px;padding:1.05rem;position:relative;overflow:hidden}.stat:before{content:"";position:absolute;top:0;left:0;right:0;height:4px;background:linear-gradient(90deg,var(--accent),var(--accent2))}.stat .icon{font-size:1.55rem}.stat .val{font-size:2.15rem;font-weight:800;line-height:1;margin:.55rem 0 .2rem;color:var(--text)}.stat .label{font-size:.78rem;color:var(--soft);letter-spacing:.08em;text-transform:uppercase;font-weight:700}
.notice{border-radius:14px;padding:1rem 1.1rem;margin:.75rem 0 1rem;border:1px solid rgba(34,197,94,.35);background:rgba(34,197,94,.11);color:#bbf7d0;font-weight:600}.empty{background:var(--panel3);border:1.5px dashed var(--border);border-radius:16px;padding:1.6rem;text-align:center;color:var(--soft);margin:1rem 0}.empty strong{color:var(--text)}
.insight{background:rgba(96,165,250,.10);border:1px solid rgba(96,165,250,.28);border-radius:14px;padding:1rem 1.15rem;margin-top:1rem;color:var(--soft);line-height:1.6}.insight b{color:var(--text)}.warn{background:rgba(245,158,11,.12);border-color:rgba(245,158,11,.35)}
mark.entity{padding:2px 7px;border-radius:7px;font-weight:800}.entity-wrap{background:var(--panel2);border:1px solid var(--border);border-radius:14px;padding:1.1rem;line-height:2.35;font-size:1.05rem;color:var(--text)}
.badge{display:inline-flex;margin:.15rem;padding:.25rem .65rem;border-radius:999px;font-size:.78rem;font-weight:800;background:#e5e7eb;color:#0f172a;border:1px solid #cbd5e1;white-space:nowrap}.chip{display:inline-block;background:rgba(24,224,196,.12);border:1px solid rgba(24,224,196,.28);color:#9ffbed;border-radius:999px;padding:.2rem .55rem;font-weight:700;font-size:.78rem;margin:.1rem}
[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:14px;overflow:hidden}.stPlotlyChart,.stPlot{background:var(--panel)!important;border-radius:14px!important}hr{border-color:var(--border)!important}
@media(max-width:900px){.stat-grid{grid-template-columns:repeat(2,1fr)}.hero h1{font-size:1.6rem!important}}
</style>
""",
    unsafe_allow_html=True,
)

DARK_BG = "#07111f"
AX_BG = "#14233a"
TEXT = "#f4f7fb"
MUTED = "#b8c4d6"
GRID = "#2b3f5f"
PALETTE = ["#18e0c4", "#60a5fa", "#a78bfa", "#f59e0b", "#fb7185", "#22c55e", "#f472b6", "#38bdf8"]


def style_axis(fig, ax):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, alpha=.45, linestyle="--")
    ax.set_axisbelow(True)
    return fig, ax


def parse_uploaded_json(uploaded_file):
    """Accepts JSON, JSONL, or malformed JSONL with a few bad lines."""
    if uploaded_file is None:
        return None, 0, []
    raw = uploaded_file.getvalue().decode("utf-8-sig", errors="replace").strip()
    if not raw:
        return pd.DataFrame(), 0, []
    errors = []
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return pd.DataFrame(obj), 0, []
        if isinstance(obj, dict):
            # IRR summary is often a dict of labels -> metrics.
            if all(isinstance(v, dict) for v in obj.values()):
                return pd.DataFrame(obj).T.reset_index(names="label"), 0, []
            return pd.DataFrame([obj]), 0, []
    except Exception:
        pass

    rows = []
    for i, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception as exc:
            errors.append({"line": i, "error": str(exc), "preview": line[:100]})
    return pd.DataFrame(rows), len(errors), errors


def normalize_main_df(df):
    df = df.copy()
    if "text" not in df.columns:
        df["text"] = ""
    if "accept" not in df.columns:
        df["accept"] = [[] for _ in range(len(df))]
    if "spans" not in df.columns:
        df["spans"] = [[] for _ in range(len(df))]

    def as_list(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        if isinstance(x, str):
            try:
                val = json.loads(x)
                return val if isinstance(val, list) else [x]
            except Exception:
                return [x]
        return []

    df["accept"] = df["accept"].apply(as_list)
    df["spans"] = df["spans"].apply(lambda x: x if isinstance(x, list) else [])
    df["clean_text"] = df["text"].astype(str).str.lower().str.replace(r"[^a-zA-Z\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    df["word_count"] = df["clean_text"].apply(lambda x: len(str(x).split()))
    return df


def interpret_alpha(alpha):
    if pd.isna(alpha):
        return "Tidak ada data"
    if alpha >= .80:
        return "Sangat baik"
    if alpha >= .60:
        return "Baik"
    if alpha >= .40:
        return "Cukup"
    if alpha >= .20:
        return "Rendah"
    return "Sangat rendah"


def file_list(files):
    if not files:
        return "<span class='chip'>Belum ada file</span>"
    if not isinstance(files, list):
        files = [files]
    return "".join(f"<span class='chip'>📄 {escape(f.name)}</span>" for f in files)


with st.sidebar:
    st.markdown("""<div class='sidebar-brand'><div class='logo'>🌿</div><h2>ABSA Lab</h2><p>EDA & IRR Dashboard</p></div>""", unsafe_allow_html=True)
    menu = st.selectbox(
        "Pilih halaman",
        ["Overview", "Distribusi Label", "Distribusi Entitas", "Panjang Review", "Korelasi Label", "NER Viewer", "IRR Agreement"],
    )
    st.caption("Upload dataset utama dan file IRR dari area atas dashboard.")

st.markdown("""
<div class='hero'>
  <div class='tag'>🌿 Aspect-Based Sentiment Analysis</div>
  <h1>EDA & IRR <span>Dashboard</span></h1>
  <p>Dashboard eksplorasi dataset: preview data, distribusi label, distribusi entitas, panjang review, korelasi label, NER viewer, dan hasil Inter-Annotator Agreement.</p>
</div>
""", unsafe_allow_html=True)

up1, up2 = st.columns(2)
with up1:
    st.markdown("<div class='small-label'>Dataset Utama</div>", unsafe_allow_html=True)
    main_files = st.file_uploader("Upload satu atau beberapa file JSON/JSONL", type=["json", "jsonl"], accept_multiple_files=True, key="main_files")
    st.markdown(file_list(main_files), unsafe_allow_html=True)
with up2:
    st.markdown("<div class='small-label'>File IRR</div>", unsafe_allow_html=True)
    irr_files = st.file_uploader("Upload file IRR JSON/JSONL", type=["json", "jsonl"], accept_multiple_files=True, key="irr_files")
    st.markdown(file_list(irr_files), unsafe_allow_html=True)

st.markdown("""
<div class='insight' style='margin-top:.25rem;margin-bottom:1rem'>
  <b>Keterangan upload:</b> Dataset utama berisi kolom <code>text</code>, <code>accept</code>, dan opsional <code>spans</code>. File IRR berisi metrik agreement seperti <code>kripp_alpha</code>, <code>percent_agreement</code>, atau <code>gwet_ac2</code>. Semua file yang diupload akan muncul sebagai chip di bawah area upload dan digabung otomatis.
</div>
""", unsafe_allow_html=True)

# Load all main files and keep all uploaded files visible
frames, total_errors, all_errors = [], 0, []
for f in main_files or []:
    parsed, err_count, errors = parse_uploaded_json(f)
    if parsed is not None and not parsed.empty:
        parsed["source_file"] = f.name
        frames.append(parsed)
    total_errors += err_count
    all_errors += [{"file": f.name, **e} for e in errors]

df = normalize_main_df(pd.concat(frames, ignore_index=True)) if frames else None

# Load all IRR files robustly
irr_frames, irr_errors = [], []
for f in irr_files or []:
    parsed, err_count, errors = parse_uploaded_json(f)
    if parsed is not None and not parsed.empty:
        parsed["source_file"] = f.name
        irr_frames.append(parsed)
    irr_errors += [{"file": f.name, **e} for e in errors]

df_irr = pd.concat(irr_frames, ignore_index=True) if irr_frames else None

if df is None:
    st.markdown("""<div class='empty'>📂 <strong>Silakan upload dataset utama.</strong><br>Format yang didukung: JSON atau JSONL. Semua file yang diupload akan ditampilkan dan digabung otomatis.</div>""", unsafe_allow_html=True)
else:
    labels = df["accept"].explode().dropna()
    avg_len = 0 if df.empty else df["word_count"].mean()
    st.markdown(f"""
    <div class='stat-grid'>
      <div class='stat'><div class='icon'>📄</div><div class='val'>{len(df):,}</div><div class='label'>Total review</div></div>
      <div class='stat'><div class='icon'>🏷️</div><div class='val'>{labels.nunique()}</div><div class='label'>Unique labels</div></div>
      <div class='stat'><div class='icon'>📏</div><div class='val'>{avg_len:.1f}</div><div class='label'>Rata-rata kata</div></div>
      <div class='stat'><div class='icon'>⚠️</div><div class='val'>{total_errors}</div><div class='label'>Parse error</div></div>
    </div>
    <div class='notice'>✅ Dataset berhasil dimuat: {len(df):,} baris dari {len(main_files or [])} file. Semua file upload sudah digabung dan ditampilkan di daftar file.</div>
    """, unsafe_allow_html=True)

    if total_errors:
        with st.expander("Lihat detail parse error"):
            st.dataframe(pd.DataFrame(all_errors), use_container_width=True)

    if menu == "Overview":
        st.markdown("<div class='panel'><div class='panel-title'>📄 <b>Preview</b> Dataset</div>", unsafe_allow_html=True)
        show_cols = [c for c in ["source_file", "text", "accept", "spans", "word_count"] if c in df.columns]
        st.dataframe(df[show_cols].head(100), use_container_width=True, height=430)
        st.markdown("<div class='insight'><b>Keterangan:</b> halaman ini dipakai untuk mengecek isi data setelah file digabung. Kolom <code>accept</code> adalah daftar label ABSA, <code>spans</code> adalah posisi entitas NER, dan <code>word_count</code> adalah jumlah kata hasil cleaning sederhana.</div></div>", unsafe_allow_html=True)

    elif menu == "Distribusi Label":
        st.markdown("<div class='panel'><div class='panel-title'>📊 <b>Distribusi</b> Label ABSA</div>", unsafe_allow_html=True)
        counts = labels.value_counts().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(counts.index.astype(str), counts.values, color=[PALETTE[i % len(PALETTE)] for i in range(len(counts))])
        style_axis(fig, ax)
        ax.set_xlabel("Label"); ax.set_ylabel("Jumlah")
        plt.xticks(rotation=35, ha="right")
        for i, v in enumerate(counts.values):
            ax.text(i, v + max(counts.values) * .01, str(int(v)), ha="center", color=TEXT, fontsize=9, fontweight="bold")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        top_label = counts.index[0] if not counts.empty else "-"
        st.markdown(f"<div class='insight'><b>Keterangan:</b> grafik ini menunjukkan jumlah kemunculan setiap label dari kolom <code>accept</code>. Label terbanyak saat ini adalah <b>{escape(str(top_label))}</b>. Jika ada label yang terlalu dominan, berarti dataset cenderung tidak seimbang dan model bisa lebih mudah memprediksi label mayoritas.</div></div>", unsafe_allow_html=True)

    elif menu == "Distribusi Entitas":
        st.markdown("<div class='panel'><div class='panel-title'>🏷️ <b>Distribusi</b> Entitas NER</div>", unsafe_allow_html=True)
        entities = [s.get("label") for spans in df["spans"] for s in (spans if isinstance(spans, list) else []) if isinstance(s, dict) and s.get("label")]
        if entities:
            counts = pd.Series(entities).value_counts()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(counts.index.astype(str), counts.values, color=[PALETTE[i % len(PALETTE)] for i in range(len(counts))])
            style_axis(fig, ax); ax.set_xlabel("Entitas"); ax.set_ylabel("Jumlah")
            plt.xticks(rotation=35, ha="right")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        else:
            counts = pd.Series(dtype=int)
            st.info("Kolom spans tidak berisi entitas yang dapat dihitung.")
        top_entity = counts.index[0] if len(counts) else "-"
        st.markdown(f"<div class='insight'><b>Keterangan:</b> distribusi entitas dihitung dari label dalam kolom <code>spans</code>. Entitas paling sering muncul adalah <b>{escape(str(top_entity))}</b>. Bagian ini membantu melihat aspek mana yang paling banyak diberi anotasi dan aspek mana yang masih kurang data.</div></div>", unsafe_allow_html=True)

    elif menu == "Panjang Review":
        st.markdown("<div class='panel'><div class='panel-title'>📏 <b>Distribusi</b> Panjang Review</div>", unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.hist(df["word_count"], bins=30, color=PALETTE[0], edgecolor=DARK_BG, alpha=.9)
            ax.axvline(df["word_count"].mean(), color="#f59e0b", linestyle="--", linewidth=2, label=f"Mean {df['word_count'].mean():.1f}")
            ax.axvline(df["word_count"].median(), color="#fb7185", linestyle=":", linewidth=2, label=f"Median {df['word_count'].median():.1f}")
            style_axis(fig, ax); ax.set_xlabel("Jumlah kata"); ax.set_ylabel("Frekuensi"); ax.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        with c2:
            st.dataframe(df["word_count"].describe().to_frame("nilai"), use_container_width=True)
        st.markdown(f"<div class='insight'><b>Keterangan:</b> histogram ini menunjukkan sebaran panjang review dalam jumlah kata. Rata-rata panjang review adalah <b>{df['word_count'].mean():.1f}</b> kata dan median <b>{df['word_count'].median():.1f}</b> kata. Data yang sangat panjang dapat dianggap outlier dan perlu dicek ulang sebelum modeling.</div></div>", unsafe_allow_html=True)

    elif menu == "Korelasi Label":
        st.markdown("<div class='panel'><div class='panel-title'>🔗 <b>Korelasi</b> Antar Label</div>", unsafe_allow_html=True)
        if labels.empty:
            st.info("Belum ada label untuk dihitung.")
        else:
            mlb = MultiLabelBinarizer()
            label_df = pd.DataFrame(mlb.fit_transform(df["accept"]), columns=mlb.classes_)
            fig, ax = plt.subplots(figsize=(11, 7))
            sns.heatmap(label_df.corr(), cmap="coolwarm", center=0, linewidths=.5, linecolor=GRID, annot=True, fmt=".2f", annot_kws={"size":8}, ax=ax)
            fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(AX_BG); ax.tick_params(colors=MUTED, labelsize=8)
            plt.xticks(rotation=35, ha="right"); plt.yticks(rotation=0)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.markdown("<div class='insight'><b>Keterangan:</b> heatmap ini menunjukkan hubungan antar label setelah kolom <code>accept</code> diubah menjadi format multi-label biner. Nilai mendekati <b>1</b> berarti dua label sering muncul bersama, nilai mendekati <b>-1</b> berarti cenderung berlawanan, dan nilai sekitar <b>0</b> berarti hubungannya lemah.</div></div>", unsafe_allow_html=True)

    elif menu == "NER Viewer":
        st.markdown("<div class='panel'><div class='panel-title'>🔍 <b>NER</b> Viewer</div>", unsafe_allow_html=True)
        idx = st.slider("Pilih indeks data", 0, max(len(df)-1, 0), 0)
        row = df.iloc[idx]
        text = str(row["text"])
        spans = row["spans"] if isinstance(row["spans"], list) else []
        colors = {"PRODUCT":"#18e0c4", "PRICE":"#22c55e", "PLACE":"#f59e0b", "PROMOTION":"#a78bfa", "POSITIVE":"#60a5fa", "NEGATIVE":"#fb7185", "NEUTRAL":"#94a3b8"}
        def get_color(label):
            return next((v for k, v in colors.items() if k in str(label)), "#94a3b8")
        rendered = escape(text)
        # Render by original offsets safely. Escape non-span segments separately.
        if spans:
            parts, pos = [], 0
            for s in sorted([x for x in spans if isinstance(x, dict) and "start" in x and "end" in x], key=lambda x: x["start"]):
                start, end, label = int(s["start"]), int(s["end"]), str(s.get("label", "LABEL"))
                if start < pos or start > len(text):
                    continue
                parts.append(escape(text[pos:start]))
                col = get_color(label)
                parts.append(f"<mark class='entity' style='background:{col}22;color:{col};border:1px solid {col}66'>{escape(text[start:end])}<sup> {escape(label)}</sup></mark>")
                pos = max(pos, end)
            parts.append(escape(text[pos:]))
            rendered = "".join(parts)
        st.markdown(f"<div class='entity-wrap'>{rendered}</div>", unsafe_allow_html=True)
        if spans:
            st.markdown("<br>" + "".join(f"<span class='badge'>{escape(str(s.get('label','')))}: {escape(text[int(s.get('start',0)):int(s.get('end',0))])}</span>" for s in spans if isinstance(s, dict)), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# IRR is always shown on its own menu or in a helpful empty state
if menu == "IRR Agreement":
    st.markdown("<div class='panel'><div class='panel-title'>📈 <b>Inter-Annotator</b> Agreement</div>", unsafe_allow_html=True)
    if df_irr is None:
        st.markdown("<div class='empty'>📂 <strong>Upload file IRR di bagian atas.</strong><br>Format JSON object seperti output EDA <code>nlp2_iaaa_textcat.jsonl</code> juga sudah didukung, jadi file IRR tidak harus JSONL per baris.</div>", unsafe_allow_html=True)
    else:
        metric_cols = [c for c in ["label", "kripp_alpha", "percent_agreement", "gwet_ac2", "source_file"] if c in df_irr.columns]
        show = df_irr[metric_cols].copy() if metric_cols else df_irr.copy()
        if "kripp_alpha" in show.columns:
            show["kategori"] = pd.to_numeric(show["kripp_alpha"], errors="coerce").apply(interpret_alpha)
            show = show.sort_values("kripp_alpha", ascending=False, na_position="last")
        st.dataframe(show, use_container_width=True, height=430)
        if "kripp_alpha" in show.columns:
            plot_df = show.dropna(subset=["kripp_alpha"]).copy()
            if not plot_df.empty:
                y = plot_df["label"] if "label" in plot_df.columns else plot_df.index.astype(str)
                fig, ax = plt.subplots(figsize=(9, max(3.5, len(plot_df) * .35)))
                ax.barh(y.astype(str), plot_df["kripp_alpha"], color=PALETTE[1])
                style_axis(fig, ax); ax.set_xlabel("Krippendorff Alpha"); ax.set_ylabel("Label")
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        if irr_errors:
            with st.expander("Lihat detail parse error IRR"):
                st.dataframe(pd.DataFrame(irr_errors), use_container_width=True)
        st.markdown("<div class='insight warn'><b>Catatan IRR:</b> kategori otomatis dibuat dari nilai Krippendorff Alpha. Agreement rendah menandakan guideline anotasi perlu diperjelas untuk label tersebut.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr><div style='text-align:center;color:var(--muted);padding:1.2rem 0;font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-size:.75rem'>🌿 ABSA Analytics Dashboard · EDA Notebook Aligned</div>", unsafe_allow_html=True)