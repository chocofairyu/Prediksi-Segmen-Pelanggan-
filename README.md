# Prediksi Segmen Pelanggan — Deployment (Streamlit)

Aplikasi ini adalah tahap **Deployment** dari proyek CRISP-DM *Implementasi Clustering
untuk Menemukan Pola pada Data* (dataset `marketing_campaign.csv`), menggunakan model
**K-Means Clustering** yang sudah dilatih.

## Isi folder
- `app.py` — aplikasi Streamlit (form input → prediksi cluster + ringkasan tiap cluster)
- `scaler_pelanggan.joblib` — StandardScaler hasil training
- `kmeans_pelanggan.joblib` — model K-Means hasil training
- `cluster_profile.csv` — ringkasan karakteristik rata-rata tiap cluster
- `requirements.txt` — daftar dependency
- `train_artifacts.py` — skrip untuk melatih ulang model dari `marketing_campaign.csv` (opsional)

## 1. Menjalankan secara lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```
Buka `http://localhost:8501` di browser.

## 2. Push ke GitHub
Buat repository baru terlebih dahulu di GitHub (tanpa README/gitignore bawaan agar tidak
bentrok), lalu jalankan dari dalam folder ini:

```bash
git init
git add .
git commit -m "Deploy: Streamlit app untuk prediksi segmen pelanggan (CRISP-DM)"
git branch -M main
git remote add origin https://github.com/<username>/<nama-repo>.git
git push -u origin main
```
Ganti `<username>` dan `<nama-repo>` sesuai akun dan repository GitHub kamu.

> Jika repository sudah ada isinya (mis. sudah dibuat lewat web GitHub dengan README),
> ganti `git push -u origin main` dengan:
> ```bash
> git pull origin main --allow-unrelated-histories
> git push -u origin main
> ```

## 3. Deploy ke Streamlit Community Cloud
1. Buka **https://share.streamlit.io** dan login dengan akun GitHub kamu.
2. Klik **New app** (atau **Create app**).
3. Pilih repository, branch (`main`), dan file utama: `app.py`.
4. Klik **Deploy**. Streamlit Cloud akan otomatis membaca `requirements.txt` dan
   menginstal seluruh dependency, lalu menjalankan aplikasinya.
5. Setelah build selesai (biasanya 1–3 menit), aplikasi akan tersedia di URL publik
   berbentuk `https://<nama-app>-<random>.streamlit.app`.

## Catatan
- File `.joblib` (model & scaler) sudah disertakan langsung di repo agar aplikasi tidak
  perlu melatih ulang model saat deploy. Jika ingin melatih ulang dari data terbaru,
  jalankan `python train_artifacts.py` (membutuhkan `marketing_campaign.csv` di folder
  yang sama), lalu commit ulang file `.joblib` yang baru.
- Pastikan versi Python di pengaturan Streamlit Cloud (Advanced settings) kompatibel
  dengan `requirements.txt` (disarankan Python 3.10–3.12).
