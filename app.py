"""
Aplikasi Streamlit - Prediksi Segmen Pelanggan (K-Means Clustering)
Tugas Mandiri: Implementasi Clustering dengan Metodologi CRISP-DM
Firza Aliyah - 10123455 - Universitas Gunadarma
"""

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Prediksi Segmen Pelanggan",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "Income",
    "Recency",
    "TotalMnt",
    "TotalPurchases",
    "NumDealsPurchases",
    "NumWebVisitsMonth",
]

FEATURE_LABELS = {
    "Income": "Pendapatan",
    "Recency": "Recency (hari)",
    "TotalMnt": "Total Pengeluaran",
    "TotalPurchases": "Total Transaksi",
    "NumDealsPurchases": "Pembelian via Diskon",
    "NumWebVisitsMonth": "Kunjungan Web/Bulan",
}

PERSONA_STYLE = [
    {"label": "Pelanggan Hemat", "emoji": "🌱", "color": "#22c55e"},
    {"label": "Pelanggan Menengah", "emoji": "🛒", "color": "#3b82f6"},
    {"label": "Pelanggan Loyal", "emoji": "⭐", "color": "#a855f7"},
    {"label": "Pelanggan Bernilai Tinggi", "emoji": "💎", "color": "#f59e0b"},
]

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    }
    .hero {
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #6366f1 0%, #ec4899 100%);
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.25);
    }
    .hero h1 {
        color: white;
        font-size: 2.1rem;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin: 0;
    }
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .persona-card {
        border-radius: 18px;
        padding: 1.6rem;
        text-align: center;
        color: white;
        margin-top: 0.5rem;
    }
    .persona-card .emoji {
        font-size: 3rem;
    }
    .persona-card .title {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .footer-note {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] {
        background: #0b1120;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD DATA & MODEL
# =========================================================
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler_pelanggan.joblib")
    model = joblib.load("kmeans_pelanggan.joblib")
    return scaler, model


@st.cache_data
def load_cluster_profile():
    try:
        return pd.read_csv("cluster_profile.csv")
    except FileNotFoundError:
        return None


def build_persona_map(profile: pd.DataFrame):
    """Urutkan cluster berdasarkan skor gabungan Income + TotalMnt, lalu beri persona otomatis."""
    score = profile["Income"].rank() + profile["TotalMnt"].rank()
    order = score.sort_values().index.tolist()
    n = len(order)
    persona_map = {}
    for rank, cluster_id in enumerate(order):
        style_idx = min(int(rank * len(PERSONA_STYLE) / n), len(PERSONA_STYLE) - 1)
        persona_map[profile.loc[cluster_id, "KMeans_Cluster"]] = PERSONA_STYLE[style_idx]
    return persona_map


def predict_cluster(scaler, model, values: list):
    input_data = pd.DataFrame([values], columns=FEATURES)
    scaled_input = scaler.transform(input_data)
    return int(model.predict(scaled_input)[0])


scaler, model = load_artifacts()
profile = load_cluster_profile()
persona_map = build_persona_map(profile) if profile is not None else {}

# =========================================================
# HERO HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <h1>🛍️ Prediksi Segmen Pelanggan</h1>
        <p>Demo deployment model <b>K-Means Clustering</b> — proyek CRISP-DM untuk menemukan
        pola tersembunyi pada data pelanggan Marketing Campaign.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR - INPUT
# =========================================================
with st.sidebar:
    st.header("🧮 Input Data Pelanggan")
    st.caption("Isi karakteristik pelanggan untuk memprediksi segmennya.")

    income = st.slider("💰 Income (pendapatan tahunan)", 0, 150000, 60000, step=1000)
    recency = st.slider("⏱️ Recency (hari sejak transaksi terakhir)", 0, 100, 20, step=1)
    total_mnt = st.slider("🛒 TotalMnt (total pengeluaran)", 0, 2600, 800, step=10)
    total_purchases = st.slider("🧾 TotalPurchases (total transaksi)", 0, 40, 15, step=1)
    num_deals = st.slider("🏷️ NumDealsPurchases (pembelian via diskon)", 0, 15, 2, step=1)
    num_web_visits = st.slider("🌐 NumWebVisitsMonth (kunjungan web/bulan)", 0, 20, 5, step=1)

    st.markdown("---")
    predict_clicked = st.button("🔮 Prediksi Segmen", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption(
        "Proyek CRISP-DM • Firza Aliyah — 10123455 — "
        "Universitas Gunadarma"
    )

# =========================================================
# TABS
# =========================================================
tab_predict, tab_explore = st.tabs(["🔮 Prediksi", "📊 Eksplorasi Cluster"])

input_values = [income, recency, total_mnt, total_purchases, num_deals, num_web_visits]

with tab_predict:
    if predict_clicked or "last_cluster" in st.session_state:
        cluster = predict_cluster(scaler, model, input_values)
        st.session_state["last_cluster"] = cluster

        style = persona_map.get(cluster, PERSONA_STYLE[0])

        col_persona, col_metrics = st.columns([1, 2], gap="large")

        with col_persona:
            st.markdown(
                f"""
                <div class="persona-card" style="background: linear-gradient(135deg, {style['color']}cc, {style['color']}55);">
                    <div class="emoji">{style['emoji']}</div>
                    <div class="title">{style['label']}</div>
                    <div>Cluster {cluster}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_metrics:
            st.markdown("##### Data yang Kamu Masukkan")
            metric_cols = st.columns(3)
            for i, (feat, val) in enumerate(zip(FEATURES, input_values)):
                with metric_cols[i % 3]:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="value">{val:,}</div>
                            <div class="label">{FEATURE_LABELS[feat]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        if profile is not None:
            st.markdown("##### 📌 Perbandingan dengan Rata-rata Cluster")
            row = profile[profile["KMeans_Cluster"] == cluster].iloc[0]

            radar_categories = ["Income", "TotalMnt", "TotalPurchases", "NumDealsPurchases", "NumWebVisitsMonth"]
            user_vals = [income, total_mnt, total_purchases, num_deals, num_web_visits]
            cluster_vals = [row[c] for c in radar_categories]

            # Normalisasi 0-1 antar kedua nilai per kategori agar bentuk radar proporsional
            max_vals = [max(u, c, 1) for u, c in zip(user_vals, cluster_vals)]
            user_norm = [u / m for u, m in zip(user_vals, max_vals)]
            cluster_norm = [c / m for c, m in zip(cluster_vals, max_vals)]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=user_norm + [user_norm[0]],
                theta=[FEATURE_LABELS[c] for c in radar_categories] + [FEATURE_LABELS[radar_categories[0]]],
                fill='toself',
                name='Input Kamu',
                line_color='#ec4899',
            ))
            fig.add_trace(go.Scatterpolar(
                r=cluster_norm + [cluster_norm[0]],
                theta=[FEATURE_LABELS[c] for c in radar_categories] + [FEATURE_LABELS[radar_categories[0]]],
                fill='toself',
                name=f'Rata-rata Cluster {cluster}',
                line_color='#6366f1',
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, showticklabels=False, range=[0, 1])),
                showlegend=True,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=420,
                margin=dict(t=30, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Isi data pelanggan di sidebar, lalu klik **Prediksi Segmen** untuk melihat hasilnya di sini.")

with tab_explore:
    if profile is not None:
        st.markdown("##### Ringkasan Seluruh Cluster")

        display_profile = profile.copy()
        display_profile["Persona"] = display_profile["KMeans_Cluster"].map(
            lambda c: f"{persona_map[c]['emoji']} {persona_map[c]['label']}"
        )
        cols_order = ["KMeans_Cluster", "Persona"] + FEATURES + ["Jumlah_Pelanggan"]
        st.dataframe(
            display_profile[cols_order].rename(columns={"KMeans_Cluster": "Cluster", **FEATURE_LABELS}),
            use_container_width=True,
            hide_index=True,
        )

        col_a, col_b = st.columns(2)

        with col_a:
            fig_bar = px.bar(
                profile,
                x="KMeans_Cluster",
                y="Jumlah_Pelanggan",
                color="KMeans_Cluster",
                text="Jumlah_Pelanggan",
                labels={"KMeans_Cluster": "Cluster", "Jumlah_Pelanggan": "Jumlah Pelanggan"},
                title="Jumlah Pelanggan per Cluster",
                color_continuous_scale=["#6366f1", "#ec4899"],
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, height=380)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            fig_scatter = px.scatter(
                profile,
                x="Income",
                y="TotalMnt",
                size="Jumlah_Pelanggan",
                color="KMeans_Cluster",
                labels={"Income": "Rata-rata Income", "TotalMnt": "Rata-rata Total Pengeluaran"},
                title="Posisi Rata-rata Tiap Cluster",
                color_continuous_scale=["#6366f1", "#ec4899"],
            )
            fig_scatter.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, height=380)
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("##### Perbandingan Fitur Antar Cluster")
        feature_pick = st.selectbox("Pilih fitur untuk dibandingkan", FEATURES, format_func=lambda f: FEATURE_LABELS[f])
        fig_feat = px.bar(
            profile,
            x="KMeans_Cluster",
            y=feature_pick,
            color="KMeans_Cluster",
            text=feature_pick,
            labels={"KMeans_Cluster": "Cluster", feature_pick: FEATURE_LABELS[feature_pick]},
            color_continuous_scale=["#6366f1", "#ec4899"],
        )
        fig_feat.update_traces(textposition="outside")
        fig_feat.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, height=360)
        st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.warning("File `cluster_profile.csv` tidak ditemukan di folder aplikasi.")

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer-note">
        Model dilatih dengan algoritma K-Means pada dataset Marketing Campaign
        (fitur: Income, Recency, TotalMnt, TotalPurchases, NumDealsPurchases, NumWebVisitsMonth) •
        Proyek CRISP-DM: Business Understanding → Deployment
    </div>
    """,
    unsafe_allow_html=True,
)
