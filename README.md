# 🧬 SegmentIQ — Customer Segmentation Dashboard

> **Machine Learning Final Project — Kelompok 6**  
> Customer Segmentation in Digital Marketing menggunakan pendekatan Hybrid ML (Unsupervised + Supervised Learning)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20Online%20Retail-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Deskripsi Proyek

**SegmentIQ** adalah aplikasi web interaktif berbasis [Streamlit](https://streamlit.io) yang mengimplementasikan pipeline Machine Learning end-to-end untuk melakukan **segmentasi pelanggan (customer segmentation)** pada data transaksi e-commerce. Proyek ini dirancang sebagai tugas akhir mata kuliah Machine Learning.

Pipeline utama menggabungkan dua pendekatan ML:
1. **Unsupervised Learning** — DBSCAN untuk deteksi outlier, lalu K-Means / Hierarchical / GMM untuk clustering
2. **Supervised Learning** — Decision Tree, Random Forest, dan SVM untuk klasifikasi segmen secara real-time

---

## 🎯 Fitur Utama

| Halaman | Deskripsi |
|---------|-----------|
| 🏠 **Home** | Dashboard ringkasan: KPI metrik, pipeline overview, data preview, anomaly summary |
| 📊 **EDA** | Exploratory Data Analysis interaktif: distribusi RFM, extended features, heatmap korelasi, box plot |
| 🔵 **Clustering** | Perbandingan 4 algoritma clustering (K-Means, Hierarchical, GMM, DBSCAN) dengan visualisasi 3D/2D, radar chart |
| 🌲 **Classification** | Perbandingan classifier (Decision Tree, Random Forest, SVM) dengan feature importance dan metrik evaluasi |
| 🎯 **Predict** | Prediksi segmen pelanggan secara real-time + batch prediction via upload CSV |

---

## 🏗️ Struktur Proyek

```
segmentation-web/
│
├── app.py                         # 🏠 Halaman utama (Home Dashboard)
│
├── pages/
│   ├── 01_📊_EDA.py               # Exploratory Data Analysis
│   ├── 02_🔵_Clustering.py        # Unsupervised Learning
│   ├── 03_🌲_Classification.py    # Supervised Learning
│   └── 04_🎯_Predict.py           # Real-time Prediction
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py             # Data loading & caching utilities
│   ├── mock_models.py             # ML model implementations (clustering & classification)
│   └── visualizer.py             # Plotly chart generators
│
├── data/
│   └── processed/
│       ├── clean_customer_features.csv    # ← Output Anggota 1 (wajib ada)
│       └── anomalous_customers.csv        # ← Output DBSCAN noise detection
│
├── models/                        # Direktori untuk file .pkl model terlatih
│   └── .gitkeep
│
├── .streamlit/
│   └── config.toml               # Konfigurasi tema Streamlit
│
└── requirements.txt              # Python dependencies
```

---

## ⚙️ Instalasi & Menjalankan Aplikasi

### Prasyarat

- Python **3.10+**
- pip atau conda

### 1. Clone Repository

```bash
git clone <repository-url>
cd segmentation-web
```

### 2. Buat Virtual Environment (Opsional tapi Direkomendasikan)

```bash
# Menggunakan venv
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Atau menggunakan conda
conda create -n segmentiq python=3.10
conda activate segmentiq
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Siapkan Data

Letakkan file CSV hasil preprocessing ke dalam folder `data/processed/`:

```
data/processed/
├── clean_customer_features.csv     # Wajib
└── anomalous_customers.csv         # Opsional
```

**Format `clean_customer_features.csv`** (kolom yang diperlukan):

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| `CustomerID` | int | ID unik pelanggan |
| `Recency` | float | Hari sejak pembelian terakhir |
| `Frequency` | float | Jumlah transaksi |
| `Monetary` | float | Total pengeluaran (£) |
| `AvgSpending` | float | Rata-rata pengeluaran per item (£) |
| `UniqueProducts` | float | Jumlah produk unik yang dibeli |
| `CancelFrequency` | float | Jumlah pembatalan transaksi |
| `AvgMonthlySpending` | float | Rata-rata pengeluaran per bulan (£) |

### 5. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser: **http://localhost:8501**

---

## 📦 Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
streamlit-option-menu>=0.3.6
scipy>=1.11.0
```

---

## 🔬 Pipeline Machine Learning

```
Raw Data (UCI Online Retail)
        │
        ▼
┌─────────────────────────┐
│  Data Cleaning           │  → Remove nulls, negatives, cancelled invoices
│  (Anggota 1)            │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Feature Engineering    │  → RFM + AvgSpending + UniqueProducts +
│  (Anggota 1)            │    CancelFrequency + AvgMonthlySpending
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  DBSCAN Noise Filter    │  → Isolasi outlier ekstrem → anomalous_customers.csv
│  (Anggota 2)            │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Clustering             │  → K-Means ✦ Hierarchical ✦ GMM
│  (Anggota 2)            │  → Evaluasi: Silhouette Score, Inertia, BIC/AIC
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Classification         │  → Decision Tree ✦ Random Forest ✦ SVM
│  (Anggota 3)            │  → Label: hasil K-Means (k=4)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Real-time Prediction   │  → Input → StandardScaler → RandomForest → Segmen
│  (Halaman Predict)      │
└─────────────────────────┘
```

---

## 🏷️ Segmen Pelanggan

| Segmen | Deskripsi | Strategi |
|--------|-----------|----------|
| 🏆 **Champion** | Pelanggan paling aktif, belanja besar & sering | Reward eksklusif, early access produk baru |
| 💎 **Loyal** | Sering belanja, nilainya stabil tinggi | Upsell premium bundle & langganan |
| ⚠️ **At-Risk** | Mulai jarang belanja, butuh perhatian | Win-back campaign dengan diskon personal |
| 🌱 **New/Inactive** | Baru bergabung atau sudah lama tidak aktif | Welcome series, starter bundle |

---

## 👥 Tim Pengembang — Kelompok 6

| Anggota | Peran | Tanggung Jawab |
|---------|-------|----------------|
| **Anggota 1** | Data Engineer | Data cleaning, feature engineering (RFM), preprocessing |
| **Anggota 2** | Unsupervised Learning Specialist | DBSCAN noise filter, K-Means, Hierarchical, GMM clustering |
| **Anggota 3** | Supervised Learning Specialist | Decision Tree, Random Forest, SVM classification |

---

## 📊 Dataset

- **Sumber**: [UCI Machine Learning Repository — Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail)
- **Cakupan**: Transaksi e-commerce dari perusahaan berbasis di UK
- **Periode**: 01 Desember 2010 — 09 Desember 2011
- **Ukuran awal**: ~541.909 baris transaksi
- **Mata uang**: GBP (£ Pound Sterling)

---

## 🛠️ Menambahkan Model .pkl Terlatih

Ketika model dari Anggota 2 & 3 sudah tersedia dalam format `.pkl`, letakkan di folder `models/` dan perbarui fungsi `load_model_if_exists()` di `utils/mock_models.py`:

```python
# Contoh penggunaan
from utils.mock_models import load_model_if_exists

kmeans_model = load_model_if_exists("kmeans_k4.pkl")
rf_model     = load_model_if_exists("random_forest.pkl")
```

---

## 📝 Catatan Pengembangan

- Aplikasi saat ini menggunakan **implementasi langsung scikit-learn** (bukan file `.pkl` yang telah disimpan) karena model dari Anggota 2 & 3 masih dalam tahap pengembangan.
- Seluruh komputasi model menggunakan `@st.cache_data` untuk performa optimal.
- Tema glassmorphism dark mode menggunakan custom CSS dengan font Inter dari Google Fonts.

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademis. Silakan gunakan dan modifikasi sesuai kebutuhan.

---

<div align="center">
  <b>Kelompok 6 · ML Final Project · Customer Segmentation in Digital Marketing</b>
</div>
