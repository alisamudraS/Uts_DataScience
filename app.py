# app.py — Streamlit Dashboard UTS DS • Insomnia (5-Level)
# Versi stabil: pakai gdown untuk download dataset besar dari Google Drive

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import skew
import plotly.express as px
import gdown

# ======================= CONFIG =======================
st.set_page_config(
    page_title="UTS DS • Insomnia (5-Level)",
    layout="wide",
    page_icon="😴",
    initial_sidebar_state="collapsed",
)

# ID file dari link Google Drive (pastikan file share-nya "Anyone with the link")
GDRIVE_FILE_ID = "1LpDfPkiXOzdj-DimvN0YIae-sdwpexSP"
LOCAL_CSV = "insomnia_dataset.csv"

SEVERITY_BINS   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001]
SEVERITY_LABELS = ["Sangat Rendah", "Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]

# ======================= LOAD & PREP ===================
@st.cache_data(show_spinner=True)
def load_csv(gdrive_id: str) -> pd.DataFrame:
    """
    Download CSV besar dari Google Drive via gdown dan baca dengan pandas.
    """
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    gdown.download(url, LOCAL_CSV, quiet=False)
    return pd.read_csv(LOCAL_CSV)

@st.cache_data(show_spinner=False)
def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    d = df_raw.copy()

    # Normalisasi nama kolom -> snake_case
    d.columns = (
        d.columns
         .str.strip().str.replace(r"\s+", " ", regex=True)
         .str.replace("[^0-9a-zA-Z_ ]", "", regex=True)
         .str.replace(" ", "_").str.lower()
    )

    # Cast numerik ringan
    for c in d.columns:
        if d[c].dtype == "object":
            try:
                d[c] = pd.to_numeric(d[c])
            except Exception:
                pass

    # Normalisasi kolom sex (jika ada)
    if "sex" in d.columns:
        m = {"f":"Female","female":"Female","m":"Male","male":"Male"}
        d["sex"] = (
            d["sex"].astype(str).str.lower()
              .str.replace(r"[^a-z]","",regex=True).map(m).fillna("Unknown")
        )

    # Imputasi median pada kolom numerik
    for c in d.select_dtypes(include=np.number).columns:
        d[c] = d[c].fillna(d[c].median())

    # Siapkan target biner (opsional)
    if "insomnia_class" in d.columns:
        d["insomnia_target"] = d["insomnia_class"].astype(int)
    elif "insomnia_probability" in d.columns:
        d["insomnia_target"] = (d["insomnia_probability"] >= 0.5).astype(int)
    else:
        d["insomnia_target"] = 0
    d["is_insomnia"] = d["insomnia_target"] == 1

    # Kategori 5-level berbasis probability
    if "insomnia_probability" in d.columns:
        d["insomnia_severity"] = pd.cut(
            d["insomnia_probability"],
            bins=SEVERITY_BINS,
            labels=SEVERITY_LABELS,
            include_lowest=True,
            right=False,
        )
    else:
        d["insomnia_severity"] = pd.Categorical(
            ["Unknown"] * len(d), categories=SEVERITY_LABELS
        )

    return d


# ======================= MAIN =======================
with st.spinner("📥 Mengunduh dan memuat data dari Google Drive…"):
    df_raw = load_csv(GDRIVE_FILE_ID)
    df = preprocess(df_raw)

num_df = df.select_dtypes(include=np.number)

# ======================= HEADER =======================
st.title("😴 Dashboard Analisis Insomnia — 5 Level Probabilitas")

# ======================= 1) DESKRIPSI =================
st.subheader("1) Ringkasan Dataset")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Jumlah Data", f"{len(df):,}")
k2.metric("Jumlah Kolom", f"{df.shape[1]}")
k3.metric("Missing (%)", f"{df_raw.isna().mean().mean()*100:.2f}%")
k4.metric("Kolom Numerik", f"{num_df.shape[1]}")

st.dataframe(df.head(), use_container_width=True, height=220)

with st.expander("Deskripsi kolom (tipe & missing)"):
    desc = pd.DataFrame({
        "dtype": df_raw.dtypes.astype(str),
        "missing_%": (df_raw.isna().mean()*100).round(2),
    }).reset_index().rename(columns={"index":"column"})
    st.dataframe(desc, use_container_width=True, height=320)

miss_rate = (df_raw.isna().mean()*100).sort_values(ascending=False).head(15)
st.plotly_chart(
    px.bar(
        miss_rate,
        labels={"index":"Kolom","value":"Missing (%)"},
        title="Top-15 Missing Value per Kolom"
    ),
    use_container_width=True
)

st.divider()

# ======================= 3) ETNIS × 5 LEVEL ===========
st.subheader("3) Distribusi 5-Level Probabilitas per Etnis")
if "ethnicity" in df.columns and "insomnia_severity" in df.columns:
    eth_ct = (
        pd.crosstab(df["ethnicity"], df["insomnia_severity"], normalize="index") * 100
    ).reindex(columns=SEVERITY_LABELS).fillna(0)

    st.dataframe(eth_ct.round(2), use_container_width=True, height=320)

    st.plotly_chart(
        px.bar(
            eth_ct.reset_index().melt(id_vars="ethnicity",
                                      var_name="Severity", value_name="Persentase"),
            x="ethnicity", y="Persentase", color="Severity", barmode="stack",
            title="Stacked % 5-Level Probabilitas per Etnis"
        ).update_layout(xaxis_tickangle=-30),
        use_container_width=True
    )

    topN = 10
    cols = st.columns(5)
    for i, sev in enumerate(SEVERITY_LABELS):
        srt = eth_ct[sev].sort_values(ascending=False).head(topN).reset_index()
        fig = px.bar(
            srt, x="ethnicity", y=sev,
            title=f"Top {topN} Etnis • {sev}",
            labels={sev:"Persentase", "ethnicity":"Etnis"}
        )
        fig.update_layout(xaxis_tickangle=-30)
        cols[i].plotly_chart(fig, use_container_width=True)
else:
    st.info("Kolom `ethnicity`/`insomnia_severity` tidak tersedia.")

st.divider()

# ============ 4) PENYAKIT × 5 LEVEL (5 tabel + 5 plot) =========
st.subheader("4) Penyakit per Kategori Probabilitas (5 Tabel & 5 Plot)")

candidates = [
    "afib_or_flutter","asthma","obesity","cancer","hypertension",
    "peripheral_vascular_disease","osteoporosis","gastrointestinal_disorder",
    "renal_failure","coronary_artery_disease","depression","anxiety","copd","stroke","diabetes",
    "cerebrovascular_disease","ckd_or_esrd","congestive_heart_failure",
    "psychiatric_disorder","lipid_metabolism_disorder"
]
disease_cols = [c for c in candidates if c in df.columns]

def proporsi_penyakit_per_kategori(dframe: pd.DataFrame, kategori: str) -> pd.Series:
    sub = dframe[dframe["insomnia_severity"] == kategori]
    total = len(sub)
    if total == 0 or not disease_cols:
        return pd.Series(dtype=float)
    return pd.Series({c: (sub[c]==1).mean()*100 for c in disease_cols}) \
             .sort_values(ascending=False)

for sev in SEVERITY_LABELS:
    st.markdown(f"**Kategori: {sev}**")
    ser = proporsi_penyakit_per_kategori(df, sev)
    if ser.empty:
        st.info("Tidak ada data pada kategori ini.")
        continue
    top10 = ser.head(10).round(3).to_frame("Persentase (%)")
    st.dataframe(top10, use_container_width=True, height=300)
    st.plotly_chart(
        px.bar(
            top10.reset_index(), x="Persentase (%)", y="index",
            orientation="h", title=f"Top-10 Penyakit • {sev}",
            labels={"index":"Penyakit"}
        ),
        use_container_width=True
    )

if disease_cols:
    heat_df = pd.DataFrame(
        {sev: proporsi_penyakit_per_kategori(df, sev) for sev in SEVERITY_LABELS}
    ).fillna(0)
    st.plotly_chart(
        px.imshow(
            heat_df, aspect="auto", color_continuous_scale="YlGnBu",
            title="Heatmap Proporsi Penyakit (%) per Kategori"
        ),
        use_container_width=True
    )

st.divider()

# ===== 5–6) STATISTIK USIA PER KATEGORI =====
st.subheader("5–6) Statistik Usia per Kategori (Mean • Median • Modus • Var • Std)")

def stats_usia_per_kategori(dframe: pd.DataFrame, kategori: str) -> dict:
    if "age" not in dframe.columns:
        return dict(Mean=np.nan, Median=np.nan, Modus=np.nan, Varian=np.nan, Std=np.nan, Jumlah=0)
    s = dframe.loc[dframe["insomnia_severity"]==kategori, "age"].dropna().astype(float)
    if len(s) == 0:
        return dict(Mean=np.nan, Median=np.nan, Modus=np.nan, Varian=np.nan, Std=np.nan, Jumlah=0)
    mode_v = s.round().mode()
    return dict(
        Mean=s.mean(), Median=s.median(), Modus=(mode_v.iloc[0] if not mode_v.empty else np.nan),
        Varian=s.var(ddof=1), Std=s.std(ddof=1), Jumlah=len(s)
    )

rows = []
for sev in SEVERITY_LABELS:
    rows.append({"Kategori": sev, **stats_usia_per_kategori(df, sev)})
stat_df = pd.DataFrame(rows).round(3)

st.dataframe(stat_df, use_container_width=True)

melt1 = stat_df.melt(
    id_vars="Kategori", value_vars=["Mean","Median","Modus"],
    var_name="Stat", value_name="Nilai"
)
st.plotly_chart(
    px.bar(
        melt1, x="Kategori", y="Nilai", color="Stat",
        barmode="group", title="Mean • Median • Modus Usia per Kategori"
    ),
    use_container_width=True
)

melt2 = stat_df.melt(
    id_vars="Kategori", value_vars=["Varian","Std"],
    var_name="Stat", value_name="Nilai"
)
st.plotly_chart(
    px.bar(
        melt2, x="Kategori", y="Nilai", color="Stat",
        barmode="group", title="Varian • Standar Deviasi Usia per Kategori"
    ),
    use_container_width=True
)

st.divider()

# ======================= 7) SKEWNESS ==================
st.subheader("7) Skewness (kolom numerik)")
if not num_df.empty:
    sk_ser = num_df.apply(lambda s: skew(s.dropna())).sort_values(ascending=False)
    cA, cB = st.columns(2)
    cA.dataframe(
        sk_ser.head(10).to_frame("skewness").style.format("{:.3f}"),
        use_container_width=True, height=280
    )
    cB.dataframe(
        sk_ser.tail(10).to_frame("skewness").style.format("{:.3f}"),
        use_container_width=True, height=280
    )
    st.plotly_chart(
        px.bar(
            sk_ser.head(12),
            labels={"index":"Fitur","value":"Skewness"},
            title="Top 12 Fitur Paling Right-skew"
        ),
        use_container_width=True
    )
else:
    st.info("Tidak ada kolom numerik untuk dihitung skewness.")
