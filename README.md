# SegmentIQ - Customer Segmentation Dashboard

> **Machine Learning Final Project - Kelompok 6**  
> Customer Segmentation in Digital Marketing menggunakan pendekatan Hybrid ML (Unsupervised + Supervised Learning)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20Online%20Retail-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Deskripsi Proyek

SegmentIQ adalah aplikasi web interaktif berbasis Streamlit yang mengimplementasikan pipeline Machine Learning end-to-end untuk melakukan segmentasi pelanggan (customer segmentation) pada data transaksi e-commerce. Proyek ini dirancang sebagai tugas akhir mata kuliah Machine Learning.

Pipeline utama menggabungkan dua pendekatan ML:
1. **Unsupervised Learning**: Perbandingan 5 algoritma clustering (K-Means Standard, K-Means + DE, K-Means + PSO, K-Means + EOA, K-Means QLDE) dengan jumlah cluster K=6.
2. **Supervised Learning**: Decision Tree dan SVM untuk klasifikasi segmen pelanggan.

---

## Fitur Utama

| Halaman | Deskripsi |
|---------|-----------|
| Home | Dashboard ringkasan: KPI metrik, pipeline overview, dan data preview |
| EDA | Exploratory Data Analysis interaktif: distribusi RFM, extended features, heatmap korelasi, dan box plot |
| Preprocessing | Langkah pemrosesan data: standardisasi fitur, analisis varians PCA, loading matriks, dan visualisasi komponen utama |
| Unsupervised | Perbandingan 5 algoritma clustering dengan visualisasi scatter 3D/2D, donut chart, radar chart, kurva konvergensi, dan perbandingan metrik evaluasi |
| Supervised | Pelatihan klasifikasi dinamis (Decision Tree dan SVM) dengan feature importance, visualisasi rules, dan tabel perbandingan performa |
| About | Informasi tim pengembang, metadata proyek, dan referensi paper |

---

## Struktur Proyek

```
segmentation-web/
│
├── app.py                         # Halaman utama (Home Dashboard)
│
├── views/
│   ├── eda.py                     # Exploratory Data Analysis
│   ├── preprocessing.py           # Preprocessing & PCA Analysis
│   ├── unsupervised_learning.py   # Unsupervised Learning (Clustering)
│   ├── supervised_learning.py     # Supervised Learning (Classification)
│   └── about.py                   # About & Team Metadata
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py             # Data loading & caching utilities
│   ├── mock_models.py             # ML model training and evaluation
│   ├── visualizer.py             # Plotly chart generators
│   └── algorithms.py             # Evolutionary clustering algorithms
│
├── data/
│   ├── processed/
│   │   ├── customer_features_raw.csv
│   │   ├── customer_features_scaled.csv
│   │   └── customer_features_pca.csv
│   └── Labeled/
│       └── hasildata_*.csv
│
└── requirements.txt              # Python dependencies
```

---

## Instalasi dan Menjalankan Aplikasi

### Prasyarat

- Python 3.10+
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

Letakkan file CSV data ke dalam folder `data/processed/` dan `data/Labeled/`:

```
data/processed/
├── customer_features_raw.csv
├── customer_features_scaled.csv
└── customer_features_pca.csv
```

### 5. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser: **http://localhost:8501**

---

## Dependencies

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

## Pipeline Machine Learning

```
Raw Data (UCI Online Retail)
        │
        ▼
┌─────────────────────────┐
│  Data Cleaning          │  -> Menghapus nulls, nilai negatif, dan transaksi dibatalkan
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Feature Engineering    │  -> RFM + AvgSpending + UniqueProducts +
└────────────┬────────────┘     CancelFrequency + AvgMonthlySpending
             │
             ▼
┌─────────────────────────┐
│  Standardization & PCA  │  -> Z-score scaling + reduksi dimensi (6 komponen utama)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Clustering (K=6)       │  -> Centroid optimasi (K-Means, DE, PSO, EOA, QLDE)
└────────────┬────────────┘  -> Metrik: SSE, Silhouette, Davies-Bouldin, Calinski-Harabasz
             │
             ▼
┌─────────────────────────┐
│  Classification         │  -> Klasifikasi segmen pelanggan (Decision Tree & SVM)
└─────────────────────────┘  -> Pelatihan dinamis menggunakan target label clustering
```

---

## Segmen Pelanggan

| ID | Segmen | Karakteristik | Strategi |
|----|--------|---------------|----------|
| 0 | High-Value (VIP) | Recency rendah, Frequency tinggi, Monetary tinggi | Program loyalitas eksklusif dan early access produk baru |
| 1 | Price-Sensitive | Volume pembelian tinggi, frequency & spending rendah | Promo diskon dan bundle harga terjangkau |
| 2 | High-Expectation (UK) | Mayoritas dari UK, cancellation sedang | Peningkatan kualitas layanan dan informasi produk |
| 3 | Uncertain Buyer | Cancellation tinggi, berasal dari non-UK | Optimasi checkout flow dan keamanan transaksi |
| 4 | Cautious Consumer | Recency tinggi, frekuensi rendah | Re-engagement campaign dan diskon musiman |
| 5 | Balanced Customer | Kinerja seimbang pada seluruh fitur | Kampanye pemasaran terdiversifikasi |

---

## Tim Pengembang - Kelompok 6

- **Ibnu Dwiki Hermawan** (EDA & Preprocessing)
- **Naufal Rifqi Rahman** (Unsupervised Learning)
- **Muhammad Farel Alkayis** (Supervised Learning)
- **Mochammad Azriel Albian Putra** (Website Implementation)

---

## Dataset

- **Sumber**: [UCI Machine Learning Repository - Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail)
- **Cakupan**: Transaksi e-commerce dari perusahaan berbasis di UK
- **Periode**: 01 Desember 2010 - 09 Desember 2011
- **Ukuran awal**: ~541.909 baris transaksi
- **Mata uang**: GBP (Pound Sterling)

---

## Lisensi

Proyek ini dibuat untuk keperluan akademis. Silakan gunakan dan modifikasi sesuai kebutuhan.

---

<div align="center">
  <b>Kelompok 6 · ML Final Project · Customer Segmentation in Digital Marketing</b>
</div>
