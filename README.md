# Prediksi Segmen Pelanggan — Deployment (Streamlit)

Aplikasi ini adalah tahap **Deployment** dari proyek CRISP-DM *Implementasi Clustering
untuk Menemukan Pola pada Data* (dataset `marketing_campaign.csv`), menggunakan model
**K-Means Clustering** yang sudah dilatih.

Versi ini menggunakan tampilan yang lebih modern: kartu metrik, grafik interaktif
(Plotly), radar chart perbandingan input vs rata-rata cluster, label persona otomatis
per cluster, serta layout dua tab (Prediksi & Eksplorasi Cluster).

## Isi folder
- `app.py` — aplikasi Streamlit (sidebar input → prediksi cluster + visualisasi)
- `scaler_pelanggan.joblib` — StandardScaler hasil training
- `kmeans_pelanggan.joblib` — model K-Means hasil training
- `cluster_profile.csv` — ringkasan karakteristik rata-rata tiap cluster
- `requirements.txt` — daftar dependency (termasuk `plotly`)
- `train_artifacts.py` — skrip untuk melatih ulang model dari `marketing_campaign.csv` (opsional)

## 1. Menjalankan secara lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```
Buka `http://localhost:8501` di browser.

> **Catatan versi Python:** gunakan Python 3.11 atau 3.12. Python versi sangat baru
> (mis. 3.14) kadang belum punya wheel stabil untuk `scikit-learn`/`numpy` dan bisa
> menyebabkan proses loading model menjadi lambat atau tersangkut.

## 2. Push ke GitHub
Kalau kamu sudah punya repo dari sebelumnya, cukup timpa (replace) file `app.py` dan
`requirements.txt` yang lama dengan versi baru ini, lalu:

```bash
git add .
git commit -m "Redesign UI: tampilan modern dengan Plotly, tab, dan persona cluster"
git push
```

Kalau repo baru, ikuti langkah lengkap dari awal:
```bash
git init
git add .
git commit -m "Deploy: Streamlit app prediksi segmen pelanggan (CRISP-DM)"
git branch -M main
git remote add origin https://github.com/<username>/<nama-repo>.git
git push -u origin main
```

## 3. Deploy / Redeploy ke Streamlit Community Cloud
Kalau aplikasi sudah pernah dideploy sebelumnya, Streamlit Cloud akan **otomatis
rebuild** begitu kamu push perubahan ke GitHub — tidak perlu deploy ulang dari nol,
cukup tunggu 1-2 menit lalu refresh halaman aplikasinya.

Kalau belum pernah deploy:
1. Buka **https://share.streamlit.io** dan login dengan akun GitHub kamu.
2. Klik **New app** (atau **Create app**).
3. Pilih repository, branch (`main`), dan file utama: `app.py`.
4. Klik **Deploy**.

## Catatan
- File `.joblib` (model & scaler) sudah disertakan langsung di repo agar aplikasi tidak
  perlu melatih ulang model saat deploy.
- Warna dan persona cluster (mis. "Pelanggan Bernilai Tinggi", "Pelanggan Hemat")
  dihitung otomatis dari `cluster_profile.csv` berdasarkan kombinasi `Income` dan
  `TotalMnt`, jadi tetap sesuai walau jumlah/urutan cluster berubah saat retrain.
