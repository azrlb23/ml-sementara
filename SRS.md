# Software Requirements Specification (SRS)

## SegmentIQ — Customer Segmentation Dashboard

---

| Atribut | Detail |
|---------|--------|
| **Nama Proyek** | SegmentIQ — Customer Segmentation in Digital Marketing |
| **Versi Dokumen** | 1.0 |
| **Tanggal** | Juni 2026 |
| **Tim** | Kelompok 6 — ML Final Project |
| **Status** | Draft |

---

## Daftar Isi

1. [Pendahuluan](#1-pendahuluan)
2. [Deskripsi Umum](#2-deskripsi-umum)
3. [Kebutuhan Fungsional](#3-kebutuhan-fungsional)
4. [Kebutuhan Non-Fungsional](#4-kebutuhan-non-fungsional)
5. [Arsitektur Sistem](#5-arsitektur-sistem)
6. [Model Machine Learning](#6-model-machine-learning)
7. [Kebutuhan Data](#7-kebutuhan-data)
8. [Antarmuka Sistem](#8-antarmuka-sistem)
9. [Batasan & Asumsi](#9-batasan--asumsi)

---

## 1. Pendahuluan

### 1.1 Tujuan Dokumen

Dokumen ini mendefinisikan kebutuhan perangkat lunak untuk aplikasi web **SegmentIQ**, sebuah dashboard interaktif yang mengimplementasikan pipeline Machine Learning end-to-end untuk segmentasi pelanggan di bidang pemasaran digital.

### 1.2 Ruang Lingkup

Sistem ini mencakup:
- Visualisasi interaktif hasil Exploratory Data Analysis (EDA)
- Eksekusi dan perbandingan algoritma clustering (unsupervised learning)
- Eksekusi dan perbandingan algoritma klasifikasi (supervised learning)
- Prediksi segmen pelanggan secara real-time (single & batch)

### 1.3 Definisi & Singkatan

| Singkatan | Kepanjangan |
|-----------|-------------|
| RFM | Recency, Frequency, Monetary |
| EDA | Exploratory Data Analysis |
| DBSCAN | Density-Based Spatial Clustering of Applications with Noise |
| GMM | Gaussian Mixture Model |
| SVM | Support Vector Machine |
| KPI | Key Performance Indicator |
| LTV | Lifetime Value |
| UCI | UC Irvine Machine Learning Repository |

### 1.4 Referensi

- UCI Online Retail Dataset: https://archive.ics.uci.edu/ml/datasets/Online+Retail
- Streamlit Documentation: https://docs.streamlit.io
- scikit-learn Documentation: https://scikit-learn.org/stable/

---

## 2. Deskripsi Umum

### 2.1 Perspektif Produk

SegmentIQ adalah aplikasi web berbasis **Streamlit** yang berdiri sendiri (standalone). Aplikasi ini mengolah data pelanggan yang telah diproses oleh tim Data Engineer, kemudian menampilkan hasil clustering dan klasifikasi dalam bentuk visualisasi interaktif.

```
[Data Engineer Output] → [SegmentIQ Web App] → [Business Insights]
    (CSV Files)              (Streamlit)          (Segment Strategy)
```

### 2.2 Fungsi Utama Produk

1. **Visualisasi EDA** — Menampilkan distribusi fitur RFM dan extended features secara interaktif
2. **Clustering Interaktif** — Menjalankan dan membandingkan 4 algoritma clustering
3. **Klasifikasi Interaktif** — Menjalankan dan membandingkan 3 algoritma klasifikasi
4. **Prediksi Real-time** — Memprediksi segmen pelanggan baru berdasarkan input manual
5. **Batch Prediction** — Memprediksi segmen untuk banyak pelanggan sekaligus via upload CSV

### 2.3 Karakteristik Pengguna

| Tipe Pengguna | Deskripsi | Kebutuhan Utama |
|---------------|-----------|-----------------|
| **Mahasiswa / Researcher** | Menganalisis hasil ML | Visualisasi interaktif, metrik evaluasi |
| **Data Scientist** | Mengevaluasi performa model | Perbandingan algoritma, feature importance |
| **Business Analyst** | Memahami segmen pelanggan | Segmen deskripsi, rekomendasi strategi |
| **Dosen / Evaluator** | Menilai proyek akhir | Kelengkapan pipeline, kualitas visualisasi |

### 2.4 Batasan Umum

- Aplikasi berjalan secara lokal melalui browser
- Membutuhkan file CSV input dari tim preprocessing
- Tidak memiliki sistem autentikasi atau database eksternal

---

## 3. Kebutuhan Fungsional

### 3.1 Modul Home Dashboard (app.py)

| ID | Kebutuhan | Prioritas |
|----|-----------|-----------|
| F-01 | Sistem menampilkan 5 KPI cards: Total Customers, Avg Recency, Avg Frequency, Avg Monetary, Cancel Rate | Tinggi |
| F-02 | Sistem menampilkan 4-tahap pipeline overview (Data Cleaning, Feature Engineering, Clustering, Classification) dengan status | Tinggi |
| F-03 | Sistem menampilkan preview 10 baris data pelanggan dengan format mata uang GBP | Sedang |
| F-04 | Sistem menampilkan ringkasan anomaly customers yang diisolasi oleh DBSCAN | Sedang |
| F-05 | Sidebar menampilkan navigasi ke seluruh halaman | Tinggi |

### 3.2 Modul EDA (pages/01_📊_EDA.py)

| ID | Kebutuhan | Prioritas |
|----|-----------|-----------|
| F-10 | Sistem menampilkan statistik deskriptif (mean, std, min, max, quartile) untuk dataset bersih dan dataset anomali | Tinggi |
| F-11 | Sistem menampilkan histogram distribusi fitur RFM (Recency, Frequency, Monetary) | Tinggi |
| F-12 | Sistem menampilkan distribusi 4 extended features (AvgSpending, UniqueProducts, CancelFrequency, AvgMonthlySpending) | Tinggi |
| F-13 | Sistem menampilkan heatmap korelasi antar seluruh fitur | Sedang |
| F-14 | Sistem menampilkan box plot untuk fitur yang dipilih pengguna | Rendah |
| F-15 | Pengguna dapat mengaktifkan/menonaktifkan setiap seksi via sidebar checkbox | Sedang |
| F-16 | Sistem menampilkan 3 metrik ringkasan: Median Recency, Median Frequency, Median Monetary | Sedang |
| F-17 | Sistem menampilkan tabel dan kartu untuk top 5 pelanggan anomali berdasarkan Monetary | Sedang |

### 3.3 Modul Clustering (pages/02_🔵_Clustering.py)

| ID | Kebutuhan | Prioritas |
|----|-----------|-----------|
| F-20 | Pengguna dapat memilih algoritma clustering: K-Means, Hierarchical, GMM, atau DBSCAN | Tinggi |
| F-21 | Pengguna dapat mengatur jumlah cluster (k) via slider untuk K-Means, Hierarchical, dan GMM | Tinggi |
| F-22 | Pengguna dapat mengatur parameter DBSCAN (eps, min_samples) via slider | Tinggi |
| F-23 | Sistem menampilkan scatter plot 3D menggunakan ruang RFM (Recency-Frequency-Monetary) | Tinggi |
| F-24 | Sistem menampilkan scatter plot 2D dengan pilihan sumbu X dan Y yang dapat dikustomisasi | Sedang |
| F-25 | Sistem menampilkan bar chart ukuran tiap cluster | Sedang |
| F-26 | Sistem menampilkan radar chart profil cluster (setelah normalisasi MinMax) | Sedang |
| F-27 | Sistem menampilkan deskripsi dan statistik untuk setiap cluster yang dihasilkan | Tinggi |
| F-28 | Sistem menampilkan perbandingan Silhouette Score keempat algoritma secara bersamaan | Tinggi |
| F-29 | Sistem menampilkan tabel ringkasan perbandingan algoritma (Silhouette, jumlah cluster, noise points) | Sedang |
| F-30 | Sistem menampilkan metrik evaluasi: Silhouette Score, Inertia (K-Means), BIC/AIC (GMM) | Tinggi |

### 3.4 Modul Classification (pages/03_🌲_Classification.py)

| ID | Kebutuhan | Prioritas |
|----|-----------|-----------|
| F-40 | Pengguna dapat memilih algoritma klasifikasi: Decision Tree, Random Forest, atau SVM | Tinggi |
| F-41 | Sistem melatih classifier menggunakan label hasil K-Means (k=4) sebagai target | Tinggi |
| F-42 | Sistem menampilkan 4 metrik evaluasi: Accuracy, F1 Score (macro), Precision, Recall | Tinggi |
| F-43 | Sistem menampilkan bar chart feature importance untuk Decision Tree dan Random Forest | Sedang |
| F-44 | Sistem menampilkan informasi bahwa SVM tidak mendukung feature importance langsung | Rendah |
| F-45 | Sistem menampilkan perbandingan keempat metrik untuk ketiga classifier secara bersamaan | Tinggi |
| F-46 | Sistem menampilkan tabel ringkasan perbandingan classifier dengan highlight nilai terbaik | Sedang |
| F-47 | Sistem menampilkan notifikasi classifier terbaik berdasarkan accuracy | Sedang |
| F-48 | Sistem menampilkan jumlah sampel training per segmen | Rendah |

### 3.5 Modul Predict (pages/04_🎯_Predict.py)

| ID | Kebutuhan | Prioritas |
|----|-----------|-----------|
| F-50 | Pengguna dapat memasukkan 7 fitur pelanggan melalui form input (slider & number input) | Tinggi |
| F-51 | Sistem memprediksi segmen pelanggan menggunakan Random Forest yang di-cache | Tinggi |
| F-52 | Sistem menampilkan nama segmen dan confidence score hasil prediksi | Tinggi |
| F-53 | Sistem menampilkan probability breakdown untuk semua segmen | Sedang |
| F-54 | Sistem menampilkan progress bar probability untuk setiap segmen | Sedang |
| F-55 | Sistem menampilkan rekomendasi strategi bisnis berdasarkan segmen yang diprediksi | Tinggi |
| F-56 | Pengguna dapat mengupload file CSV untuk batch prediction | Sedang |
| F-57 | Sistem memvalidasi kolom CSV yang diupload dan menampilkan error jika ada kolom yang hilang | Sedang |
| F-58 | Sistem menampilkan hasil batch prediction dalam tabel dengan kolom PredictedCluster dan SegmentName | Sedang |
| F-59 | Pengguna dapat mengunduh hasil batch prediction dalam format CSV | Rendah |

---

## 4. Kebutuhan Non-Fungsional

### 4.1 Performa

| ID | Kebutuhan |
|----|-----------|
| NF-01 | Halaman pertama harus termuat dalam < 10 detik pada koneksi lokal |
| NF-02 | Hasil clustering harus tampil dalam < 5 detik setelah pemilihan algoritma (menggunakan `@st.cache_data`) |
| NF-03 | Prediksi single customer harus menghasilkan output dalam < 2 detik |
| NF-04 | Batch prediction untuk 1.000 baris data harus selesai dalam < 10 detik |

### 4.2 Keandalan (Reliability)

| ID | Kebutuhan |
|----|-----------|
| NF-10 | Sistem harus menampilkan pesan error yang jelas jika file CSV tidak ditemukan |
| NF-11 | Sistem harus tetap berjalan meskipun `anomalous_customers.csv` tidak tersedia |
| NF-12 | Sistem harus memvalidasi format kolom CSV yang diupload sebelum pemrosesan |

### 4.3 Usabilitas

| ID | Kebutuhan |
|----|-----------|
| NF-20 | Antarmuka menggunakan dark mode dengan desain glassmorphism yang konsisten |
| NF-21 | Seluruh grafik bersifat interaktif (zoom, hover tooltip, legend toggle) menggunakan Plotly |
| NF-22 | Navigasi antar halaman tersedia di sidebar pada setiap halaman |
| NF-23 | Setiap metrik dan grafik harus disertai label atau caption yang deskriptif |
| NF-24 | Warna segmen harus konsisten di seluruh halaman |

### 4.4 Maintainability

| ID | Kebutuhan |
|----|-----------|
| NF-30 | Setiap modul Python harus memiliki docstring yang menjelaskan fungsinya |
| NF-31 | Fungsi data loading dan model harus terpisah dari logika tampilan (UI) |
| NF-32 | Sistem harus mendukung penggantian model mock dengan file `.pkl` tanpa mengubah logika halaman |
| NF-33 | Konfigurasi path data harus terpusat di `utils/data_loader.py` |

### 4.5 Portabilitas

| ID | Kebutuhan |
|----|-----------|
| NF-40 | Aplikasi harus dapat dijalankan di Windows, Linux, dan macOS |
| NF-41 | Seluruh dependency harus terdaftar dalam `requirements.txt` |
| NF-42 | Aplikasi tidak bergantung pada layanan cloud atau API eksternal |

---

## 5. Arsitektur Sistem

### 5.1 Diagram Komponen

```
┌───────────────────────────────────────────────────┐
│                  Browser (Client)                  │
│              http://localhost:8501                 │
└─────────────────────┬─────────────────────────────┘
                      │ HTTP
┌─────────────────────▼─────────────────────────────┐
│              Streamlit Server (Python)             │
│                                                   │
│  ┌──────────┐  ┌───────────┐  ┌───────────────┐  │
│  │  app.py  │  │  pages/   │  │    utils/     │  │
│  │  (Home)  │  │ 01 EDA    │  │ data_loader   │  │
│  └──────────┘  │ 02 Clust  │  │ mock_models   │  │
│                │ 03 Class  │  │ visualizer    │  │
│                │ 04 Pred   │  └───────┬───────┘  │
│                └───────────┘          │           │
└───────────────────────────────────────┼───────────┘
                                        │ read/write
┌───────────────────────────────────────▼───────────┐
│                    File System                    │
│  data/processed/clean_customer_features.csv       │
│  data/processed/anomalous_customers.csv           │
│  models/*.pkl  (opsional)                         │
└───────────────────────────────────────────────────┘
```

### 5.2 Alur Data

```
CSV Input
  │
  ▼
data_loader.py
  ├─ load_clean_data()      → DataFrame (clean)
  └─ load_anomalous_data()  → DataFrame (anomali)
        │
        ▼
mock_models.py
  ├─ run_kmeans()         ─┐
  ├─ run_hierarchical()   ├─→ (df_clustered, metrics)
  ├─ run_gmm()            │
  ├─ run_dbscan()        ─┘
  │
  ├─ run_decision_tree()  ─┐
  ├─ run_random_forest()   ├─→ (metrics, feature_importance)
  └─ run_svm()            ─┘
        │
        ▼
visualizer.py
  └─ plot_*()  →  Plotly Figure  →  st.plotly_chart()
```

---

## 6. Model Machine Learning

### 6.1 Algoritma Clustering

| Algoritma | Library | Parameter Utama | Metrik Evaluasi |
|-----------|---------|-----------------|-----------------|
| **K-Means** | `sklearn.cluster.KMeans` | `n_clusters`, `random_state=42`, `n_init=10` | Silhouette Score, Inertia |
| **Hierarchical** | `sklearn.cluster.AgglomerativeClustering` | `n_clusters`, `linkage='ward'` | Silhouette Score |
| **GMM** | `sklearn.mixture.GaussianMixture` | `n_components`, `covariance_type='full'` | Silhouette Score, BIC, AIC |
| **DBSCAN** | `sklearn.cluster.DBSCAN` | `eps`, `min_samples` | Silhouette Score, n_clusters, n_noise |

### 6.2 Algoritma Klasifikasi

| Algoritma | Library | Parameter Utama | Feature Importance |
|-----------|---------|-----------------|-------------------|
| **Decision Tree** | `sklearn.tree.DecisionTreeClassifier` | `max_depth=6`, `random_state=42` | ✅ `feature_importances_` |
| **Random Forest** | `sklearn.ensemble.RandomForestClassifier` | `n_estimators=100`, `max_depth=8`, `n_jobs=-1` | ✅ `feature_importances_` |
| **SVM** | `sklearn.svm.SVC` | `kernel='rbf'`, `C=1.0` | ❌ Tidak tersedia langsung |

### 6.3 Preprocessing Fitur

- **Scaler**: `sklearn.preprocessing.StandardScaler` (Z-score normalization)
- **Train-Test Split**: 80/20, `stratify=y`, `random_state=42`
- **Label Source**: Hasil K-Means (k=4) digunakan sebagai label target untuk klasifikasi

### 6.4 Segmen Pelanggan (Output)

| Cluster ID | Nama Segmen | Karakteristik |
|------------|-------------|---------------|
| 0 | 🏆 Champion | Recency rendah, Frequency tinggi, Monetary tinggi |
| 1 | 💎 Loyal | Frequency sedang-tinggi, Monetary sedang |
| 2 | ⚠️ At-Risk | Recency tinggi (lama tidak beli), Frequency menurun |
| 3 | 🌱 New/Inactive | Frequency sangat rendah, Monetary kecil |

---

## 7. Kebutuhan Data

### 7.1 File Input

#### `clean_customer_features.csv`
- **Lokasi**: `data/processed/clean_customer_features.csv`
- **Status**: **Wajib**
- **Sumber**: Output preprocessing Anggota 1

| Kolom | Tipe Data | Constraint |
|-------|-----------|------------|
| CustomerID | int | Unik, tidak boleh null |
| Recency | float | ≥ 0 |
| Frequency | float | ≥ 1 |
| Monetary | float | ≥ 0 |
| AvgSpending | float | ≥ 0 |
| UniqueProducts | float | ≥ 1 |
| CancelFrequency | float | ≥ 0 |
| AvgMonthlySpending | float | ≥ 0 |

#### `anomalous_customers.csv`
- **Lokasi**: `data/processed/anomalous_customers.csv`
- **Status**: Opsional (aplikasi tetap berjalan tanpa file ini)
- **Sumber**: Output DBSCAN noise detection Anggota 2
- **Kolom**: Sama dengan `clean_customer_features.csv`

### 7.2 File Output Batch Prediction

| Kolom | Tipe Data | Deskripsi |
|-------|-----------|-----------|
| *(semua kolom input)* | — | Dipertahankan |
| PredictedCluster | int | ID cluster (0–3) |
| SegmentName | str | Nama segmen (Champion, Loyal, dll.) |

---

## 8. Antarmuka Sistem

### 8.1 Antarmuka Pengguna (UI)

- **Framework**: Streamlit ≥ 1.32.0
- **Tema**: Dark mode glassmorphism
- **Tipografi**: Inter (Google Fonts), diimpor via CSS
- **Warna Utama**:

| Warna | Hex | Penggunaan |
|-------|-----|------------|
| Violet | `#7C3AED` | Warna primer, Home & Clustering |
| Cyan | `#06B6D4` | EDA highlights |
| Emerald | `#10B981` | Classification, status positif |
| Amber | `#F59E0B` | Predict, peringatan |
| Slate | `#94A3B8` | Teks sekunder |

### 8.2 Komponen Visualisasi

| Komponen | Library | Halaman |
|----------|---------|---------|
| Histogram distribusi | Plotly | EDA |
| Heatmap korelasi | Plotly | EDA |
| Box plot | Plotly | EDA |
| Scatter 3D (RFM space) | Plotly | Clustering |
| Scatter 2D | Plotly | Clustering |
| Bar chart cluster size | Plotly | Clustering |
| Radar chart | Plotly | Clustering |
| Bar chart perbandingan algoritma | Plotly | Clustering, Classification |
| Bar chart feature importance | Plotly | Classification |
| Progress bar probabilitas | HTML/CSS | Predict |

### 8.3 Antarmuka File

| Tipe | Detail |
|------|--------|
| **Input** | File CSV via `st.file_uploader` (halaman Predict — Batch) |
| **Output** | File CSV via `st.download_button` (halaman Predict — Batch) |

---

## 9. Batasan & Asumsi

### 9.1 Batasan Sistem

1. Aplikasi hanya mendukung dataset dalam format **CSV** dengan pemisah koma (`,`)
2. Semua fitur numerik diharapkan sudah dalam skala wajar (tidak perlu transformasi log)
3. Model tidak disimpan ke disk secara otomatis; model di-cache hanya selama sesi Streamlit aktif
4. Aplikasi tidak mendukung multi-user secara bersamaan pada mode lokal
5. Implementasi saat ini menggunakan scikit-learn secara langsung (bukan file `.pkl` yang dipersist)

### 9.2 Asumsi

1. Data input sudah melalui proses cleaning oleh Anggota 1 (tidak ada nilai null atau negatif)
2. CustomerID bersifat unik dalam dataset
3. Jumlah cluster optimal adalah **4** (Champion, Loyal, At-Risk, New/Inactive)
4. Browser pengguna mendukung JavaScript untuk rendering Plotly
5. Koneksi internet tersedia untuk memuat Google Fonts (Inter)

### 9.3 Dependensi Eksternal

| Dependensi | Versi | Kritis |
|------------|-------|--------|
| Python | ≥ 3.10 | ✅ Ya |
| Streamlit | ≥ 1.32.0 | ✅ Ya |
| scikit-learn | ≥ 1.3.0 | ✅ Ya |
| pandas | ≥ 2.0.0 | ✅ Ya |
| plotly | ≥ 5.18.0 | ✅ Ya |
| Google Fonts (CDN) | — | ❌ Tidak (hanya kosmetik) |

---

## Riwayat Revisi

| Versi | Tanggal | Deskripsi | Penulis |
|-------|---------|-----------|---------|
| 1.0 | Juni 2026 | Draft awal — mencakup seluruh 5 modul | Kelompok 6 |

---

*Dokumen ini merupakan bagian dari tugas akhir mata kuliah Machine Learning — Kelompok 6.*
