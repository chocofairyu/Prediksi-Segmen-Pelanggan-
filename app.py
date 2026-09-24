import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Prediksi Segmen Pelanggan",
    page_icon="🛍️",
    layout="centered",
)

FEATURES = [
    "Income",
    "Recency",
    "TotalMnt",
    "TotalPurchases",
    "NumDealsPurchases",
    "NumWebVisitsMonth",
]


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


def predict_cluster(scaler, model, income, recency, total_mnt, total_purchases, num_deals, num_web_visits):
    input_data = pd.DataFrame(
        [[income, recency, total_mnt, total_purchases, num_deals, num_web_visits]],
        columns=FEATURES,
    )
    scaled_input = scaler.transform(input_data)
    return int(model.predict(scaled_input)[0])


scaler, model = load_artifacts()
cluster_profile = load_cluster_profile()

st.title("🛍️ Prediksi Segmen Pelanggan")
st.markdown(
    """
Aplikasi ini memprediksi **segmen (cluster) pelanggan** menggunakan model
**K-Means Clustering** yang dilatih pada dataset *Marketing Campaign*, sebagai bagian dari
proyek CRISP-DM: Business Understanding → Data Understanding → Data Preparation →
Modeling → Evaluation → **Deployment**.

Masukkan data pelanggan pada form di bawah ini untuk melihat pelanggan tersebut
termasuk ke dalam segmen yang mana.
"""
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input("Income (pendapatan tahunan)", min_value=0, value=60000, step=1000)
        recency = st.number_input("Recency (hari sejak transaksi terakhir)", min_value=0, value=20, step=1)
        total_mnt = st.number_input("TotalMnt (total pengeluaran)", min_value=0, value=800, step=10)

    with col2:
        total_purchases = st.number_input("TotalPurchases (total transaksi)", min_value=0, value=15, step=1)
        num_deals = st.number_input("NumDealsPurchases (pembelian via diskon)", min_value=0, value=2, step=1)
        num_web_visits = st.number_input("NumWebVisitsMonth (kunjungan web/bulan)", min_value=0, value=5, step=1)

    submitted = st.form_submit_button("Prediksi Segmen", use_container_width=True)

if submitted:
    cluster = predict_cluster(
        scaler, model, income, recency, total_mnt, total_purchases, num_deals, num_web_visits
    )
    st.success(f"Pelanggan ini termasuk ke dalam **Cluster {cluster}**")

    if cluster_profile is not None:
        row = cluster_profile[cluster_profile["KMeans_Cluster"] == cluster]
        if not row.empty:
            st.markdown("#### Karakteristik rata-rata Cluster ini (dari data pelatihan)")
            st.dataframe(row.drop(columns=["KMeans_Cluster"]), use_container_width=True, hide_index=True)

st.divider()

if cluster_profile is not None:
    st.markdown("### Ringkasan Seluruh Cluster")
    st.dataframe(cluster_profile, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        cluster_profile["KMeans_Cluster"].astype(str),
        cluster_profile["Jumlah_Pelanggan"],
        color="#4C72B0",
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Jumlah Pelanggan")
    ax.set_title("Jumlah Pelanggan per Cluster")
    st.pyplot(fig)

st.caption(
    "Proyek CRISP-DM • Implementasi Clustering untuk Menemukan Pola pada Data • "
    "Firza Aliyah — 10123455 — Universitas Gunadarma"
)
