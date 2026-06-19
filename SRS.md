# Software Requirements Specification (SRS)

## SegmentIQ - Customer Segmentation Dashboard

---

| Atribut | Detail |
|---------|--------|
| **Nama Proyek** | SegmentIQ - Customer Segmentation in Digital Marketing |
| **Versi Dokumen** | 1.0 |
| **Tanggal** | Juni 2026 |
| **Tim** | Kelompok 6 - ML Final Project |
| **Status** | Final |

---

## Daftar Isi

1. Pendahuluan
2. Deskripsi Umum
3. Kebutuhan Fungsional
4. Kebutuhan Non-Fungsional
5. Arsitektur Sistem
6. Model Machine Learning
7. Kebutuhan Data
8. Antarmuka Sistem
9. Batasan dan Asumsi

---

## 1. Pendahuluan

### 1.1 Tujuan Dokumen

Dokumen ini mendefinisikan kebutuhan perangkat lunak untuk aplikasi web SegmentIQ, sebuah dashboard interaktif yang mengimplementasikan pipeline Machine Learning end-to-end untuk segmentasi pelanggan di bidang pemasaran digital.

### 1.2 Ruang Lingkup

Sistem ini mencakup:
- Visualisasi interaktif hasil Exploratory Data Analysis (EDA)
- Penjelasan langkah preprocessing data termasuk Standard Scaling dan Principal Component Analysis (PCA)
- Perbandingan 5 algoritma clustering (unsupervised learning) dengan jumlah cluster K=6
- Pelatihan dan perbandingan 2 algoritma klasifikasi (supervised learning) menggunakan data dari 5 sumber label hasil clustering

### 1.3 Definisi dan Singkatan

| Singkatan | Kepanjangan |
|-----------|-------------|
| RFM | Recency, Frequency, Monetary |
| EDA | Exploratory Data Analysis |
| PCA | Principal Component Analysis |
| SVM | Support Vector Machine |
| DT | Decision Tree |
| DE | Differential Evolution |
| PSO | Particle Swarm Optimization |
| EOA | Evolutionary Optimization Algorithm |
| QLDE | Q-Learning Differential Evolution |
| KPI | Key Performance Indicator |
| UCI | UC Irvine Machine Learning Repository |

### 1.4 Referensi

- UCI Online Retail Dataset: https://archive.ics.uci.edu/ml/datasets/Online+Retail
- Streamlit Documentation: https://docs.streamlit.io
- scikit-learn Documentation: https://scikit-learn.org/stable/

---

## 2. Deskripsi Umum

### 2.1 Perspektif Produk

SegmentIQ adalah aplikasi web berbasis Streamlit yang berdiri sendiri (standalone). Aplikasi ini mengolah data pelanggan yang telah diproses oleh tim Data Engineer, kemudian menampilkan hasil clustering dan klasifikasi dalam bentuk visualisasi interaktif.

```
[Data Preprocessing Output] -> [SegmentIQ Web App] -> [Business Insights]
      (CSV Files)                 (Streamlit)          (Segment Strategy)
```

### 2.2 Fungsi Utama Produk

1. **Visualisasi EDA**: Menampilkan distribusi fitur RFM dan extended features secara interaktif.
2. **Visualisasi Preprocessing**: Menampilkan langkah standardisasi fitur, analisis rasio varians PCA, matriks loading komponen, dan proyeksi komponen utama.
3. **Clustering Interaktif**: Menjalankan dan membandingkan 5 algoritma clustering dengan jumlah cluster K=6 yang terkunci.
4. **Klasifikasi Interaktif**: Menjalankan dan membandingkan 2 algoritma klasifikasi dengan pilihan 5 target label sumber.

### 2.3 Karakteristik Pengguna

| Tipe Pengguna | Deskripsi | Kebutuhan Utama |
|---------------|-----------|-----------------|
| **Mahasiswa / Researcher** | Menganalisis hasil ML | Visualisasi interaktif, metrik evaluasi |
| **Data Scientist** | Mengevaluasi performa model | Perbandingan algoritma, feature importance |
| **Business Analyst** | Memahami segmen pelanggan | Segmen deskripsi, rekomendasi strategi |
| **Dosen / Evaluator** | Menilai proyek akhir | Kelengkapan pipeline, kualitas visualisasi |

### 2.4 Batasan Umum

- Aplikasi berjalan secara lokal melalui browser.
- Membutuhkan file CSV input dari tim preprocessing.
- Tidak memiliki sistem autentikasi atau database eksternal.

---

## 3. Kebutuhan Fungsional

### 3.1 Modul Home Dashboard (app.py)

| ID | Kebutuhan | Prioritas |
|----|-----------|-----------|
| F-01 | Sistem menampilkan 5 KPI cards: Total Customers, Avg Recency, Avg Frequency, Avg Monetary, Cancel Rate | Tinggi |
| F-02 | Sistem menampilkan 4-tahap pipeline overview (EDA, Preprocessing, Unsupervised, Supervised) dengan status | Tinggi |
| F-03 | Sistem menampilkan preview 15 baris data pelanggan dengan format mata uang GBP | Sedang |
| F-04 | Sidebar menampilkan navigasi ke seluruh halaman | Tinggi |

### 3.2 Modul EDA (views/eda.py)

| ID | Kebutuhan | Prioritas |
|----|-----------|-----------|
| F-10 | Sistem menampilkan statistik deskriptif untuk dataset bersih pelanggan (4,335 records) | Tinggi |
| F-11 | Sistem menampilkan histogram distribusi fitur RFM (Recency, Frequency, Monetary) | Tinggi |
| F-12 | Sistem menampilkan distribusi extended features (AvgSpending, UniqueProducts, CancelFrequency, AvgMonthlySpending) | Tinggi |
| F-13 | Sistem menampilkan heatmap korelasi antar seluruh fitur | Sedang |
| F-14 | Sistem menampilkan box plot untuk fitur yang dipilih pengguna | Rendah |
| F-15 | Pengguna dapat mengaktifkan/menonaktifkan setiap seksi via sidebar checkbox | Sedang |

### 3.3 Modul Preprocessing (views/preprocessing.py)

| ID | Kebutuhan | Prioritas |
|----|-----------|-----------|
| F-20 | Sistem menampilkan deskripsi formula standardisasi fitur (Z-score normalization) | Tinggi |
| F-21 | Sistem menampilkan visualisasi rasio varians kumulatif PCA untuk menentukan jumlah komponen utama optimal | Tinggi |
| F-22 | Sistem menampilkan heatmap matriks loading PCA untuk melihat kontribusi fitur terhadap masing-masing komponen utama | Tinggi |
| F-23 | Sistem menampilkan visualisasi scatter plot proyeksi 2 dimensi komponen utama pertama (PC1) dan kedua (PC2) | Tinggi |

### 3.4 Modul Clustering (views/unsupervised_learning.py)

| ID | Kebutuhan | Prioritas |
|----|-----------|-----------|
| F-30 | Pengguna dapat memilih algoritma clustering: K-Means Standard, K-Means + DE, K-Means + PSO, K-Means + EOA, atau K-Means QLDE | Tinggi |
| F-31 | Sistem mengunci jumlah cluster K=6 berdasarkan analisis optimalitas elbow dan silhouette | Tinggi |
| F-32 | Sistem menampilkan scatter plot 3D menggunakan ruang RFM (Recency-Frequency-Monetary) | Tinggi |
| F-33 | Sistem menampilkan scatter plot 2D dengan pilihan sumbu X dan Y yang dapat dikustomisasi | Sedang |
| F-34 | Sistem menampilkan donut chart proporsi ukuran tiap cluster | Sedang |
| F-35 | Sistem menampilkan radar chart profil cluster (setelah normalisasi MinMax) | Sedang |
| F-36 | Sistem menampilkan tabel profil dan deskripsi detail untuk setiap cluster yang dihasilkan | Tinggi |
| F-37 | Sistem menampilkan visualisasi kurva konvergensi nilai SSE per iterasi optimasi centroid | Tinggi |
| F-38 | Sistem menampilkan visualisasi perbandingan performa semua algoritma (SSE, Silhouette, Davies-Bouldin, Calinski-Harabasz) | Tinggi |

### 3.5 Modul Classification (views/supervised_learning.py)

| ID | Kebutuhan | Prioritas |
|----|-----------|-----------|
| F-40 | Pengguna dapat melatih algoritma klasifikasi secara dinamis: Decision Tree atau SVM | Tinggi |
| F-41 | Pengguna dapat memilih target label sumber yang digunakan sebagai target klasifikasi (QLDE, STANDARD, DE, PSO, EOA) | Tinggi |
| F-42 | Sistem melatih model secara real-time dan menampilkan metrik evaluasi: Accuracy, Precision, Recall, F1 Score | Tinggi |
| F-43 | Sistem menampilkan bar chart feature importance dan visualisasi rules logika bisnis untuk model Decision Tree | Tinggi |
| F-44 | Sistem menampilkan perbandingan performa (akurasi dan waktu latih) antara Decision Tree dan SVM dalam bentuk grafik dan tabel komparatif | Tinggi |

---

## 4. Kebutuhan Non-Fungsional

### 4.1 Performa

| ID | Kebutuhan |
|----|-----------|
| NF-01 | Halaman dashboard harus termuat dalam < 5 detik pada koneksi lokal |
| NF-02 | Eksekusi clustering harus selesai dengan cepat dibantu caching decorator (@st.cache_data) |
| NF-03 | Pelatihan model klasifikasi secara dinamis harus selesai dalam < 3 detik |

### 4.2 Keandalan (Reliability)

| ID | Kebutuhan |
|----|-----------|
| NF-10 | Sistem harus menampilkan pesan error jika file CSV data utama tidak ditemukan |
| NF-11 | Sistem harus memvalidasi data target klasifikasi agar sesuai dengan jumlah sampel training |

### 4.3 Usabilitas

| ID | Kebutuhan |
|----|-----------|
| NF-20 | Antarmuka menggunakan dark mode dengan desain glassmorphism yang konsisten |
| NF-21 | Seluruh grafik bersifat interaktif (zoom, hover tooltip, legend toggle) menggunakan Plotly |
| NF-22 | Navigasi antar halaman tersedia di sidebar pada setiap halaman |
| NF-23 | Setiap metrik dan grafik harus disertai label atau caption yang deskriptif |

### 4.4 Maintainability

| ID | Kebutuhan |
|----|-----------|
| NF-30 | Setiap modul Python harus memiliki docstring yang menjelaskan fungsinya |
| NF-31 | Fungsi data loading dan model harus terpisah dari logika tampilan (UI) |
| NF-32 | Konfigurasi path data harus terpusat di utils/data_loader.py |

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
│  │  app.py  │  │   views/  │  │    utils/     │  │
│  │  (Home)  │  │ eda.py    │  │ data_loader   │  │
│  └──────────┘  │ preproc.py│  │ mock_models   │  │
│                │ cluster.py│  │ visualizer    │  │
│                │ superv.py │  └───────┬───────┘  │
│                └───────────┘          │           │
└───────────────────────────────────────┼───────────┘
                                        │ read/write
┌───────────────────────────────────────▼───────────┐
│                    File System                    │
│  data/processed/customer_features_raw.csv         │
│  data/processed/customer_features_scaled.csv      │
│  data/processed/customer_features_pca.csv         │
│  data/Labeled/hasildata_*.csv                     │
└───────────────────────────────────────────────────┘
```

---

## 6. Model Machine Learning

### 6.1 Algoritma Clustering

| Algoritma | Library | Parameter Utama | Metrik Evaluasi |
|-----------|---------|-----------------|-----------------|
| **K-Means Standard** | sklearn.cluster.KMeans | n_clusters=6, random_state=42, n_init=10 | Silhouette, SSE, DB, CH |
| **K-Means + DE** | utils.algorithms.KMeansDE | n_clusters=6, pop_size=30, max_iter=100, F=0.5, Cr=0.9 | Silhouette, SSE, DB, CH |
| **K-Means + PSO** | utils.algorithms.KMeansPSO | n_clusters=6, pop_size=30, max_iter=100 | Silhouette, SSE, DB, CH |
| **K-Means + EOA** | utils.algorithms.KMeansEOA | n_clusters=6, pop_size=30, max_iter=100 | Silhouette, SSE, DB, CH |
| **K-Means QLDE** | utils.algorithms.QLDE | n_clusters=6, pop_size=30, max_iter=100, F_init=0.5, Cr=0.9 | Silhouette, SSE, DB, CH |

### 6.2 Algoritma Klasifikasi

| Algoritma | Library | Parameter Utama | Feature Importance |
|-----------|---------|-----------------|-------------------|
| **Decision Tree** | sklearn.tree.DecisionTreeClassifier | max_depth=4, random_state=42 | Tersedia langsung |
| **SVM** | sklearn.svm.SVC | kernel='rbf', class_weight='balanced', random_state=42 | Tidak tersedia langsung |

---

## 7. Kebutuhan Data

### 7.1 File Input

#### customer_features_raw.csv
- **Lokasi**: data/processed/customer_features_raw.csv
- **Status**: Wajib
- **Jumlah Baris**: 4,335 records

#### customer_features_scaled.csv
- **Lokasi**: data/processed/customer_features_scaled.csv
- **Status**: Wajib
- **Jumlah Baris**: 4,335 records

#### customer_features_pca.csv
- **Lokasi**: data/processed/customer_features_pca.csv
- **Status**: Wajib
- **Jumlah Baris**: 4,335 records

#### Labeled Data CSVs
- **Lokasi**: data/Labeled/hasildata_*.csv
- **Status**: Wajib untuk supervised learning training labels

---

## 8. Antarmuka Sistem

### 8.1 Antarmuka Pengguna (UI)

- **Framework**: Streamlit >= 1.32.0
- **Tema**: Dark mode glassmorphism
- **Tipografi**: Inter (Google Fonts), diimpor via CSS
- **Skema Warna**: Violet (primer), Steel (sekunder), Graphite (batasan), Ash (teks sekunder).

---

## 9. Batasan dan Asumsi

### 9.1 Batasan Sistem

1. Aplikasi hanya mendukung dataset dalam format CSV dengan pemisah koma (,).
2. Model klasifikasi dilatih secara real-time berdasarkan data training berukuran maksimum 2,000 sampel untuk mengoptimalkan kinerja SVM.
3. Aplikasi tidak mendukung multi-user secara bersamaan pada mode lokal.

### 9.2 Asumsi

1. Data input sudah melalui proses cleaning (tidak ada nilai null atau negatif).
2. CustomerID bersifat unik dalam dataset.
3. Jumlah cluster optimal adalah 6 sesuai dengan hasil optimalitas evaluasi centroid.
4. Browser pengguna mendukung JavaScript untuk rendering Plotly.

---

## Riwayat Revisi

| Versi | Tanggal | Deskripsi | Penulis |
|-------|---------|-----------|---------|
| 1.0 | Juni 2026 | Draft awal - mencakup seluruh 5 modul | Kelompok 6 |
| 1.1 | Juni 2026 | Final alignment - menghapus DBSCAN, membatasi klasifikasi ke DT dan SVM, mengunci K=6 | Kelompok 6 |

---

*Dokumen ini merupakan bagian dari tugas akhir mata kuliah Machine Learning - Kelompok 6.*
