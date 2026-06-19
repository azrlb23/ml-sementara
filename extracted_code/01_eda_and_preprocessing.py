# === Cell 1 ===# ============================================================
# Import library esensial
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi visualisasi global
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size']  = 11

print("Library berhasil diimport.")


# === Cell 2 ===# ============================================================
# Memuat dataset
# Pastikan path disesuaikan dengan struktur repositori
# Dataset: Kaggle E-Commerce Customer Segmentation
# Sumber: https://www.kaggle.com/code/fabiendaniel/customer-segmentation/notebook
# ============================================================
file_path = '../data/raw/data.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

print(f"Dimensi data awal : {df.shape}")
print(f"Kolom             : {list(df.columns)}")
print(f"\nMissing values :")
print(df.isnull().sum().to_string())
df.head()


# === Cell 3 ===# --- Step 1: Hapus CustomerID kosong ---
df_clean = df.dropna(subset=['CustomerID']).copy()
df_clean['InvoiceNo']   = df_clean['InvoiceNo'].astype(str)
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

n_before       = len(df)
n_after_cid    = len(df_clean)

# --- Step 2: Filter StockCode produk nyata (>= 5 digit angka) ---
df_clean       = df_clean[df_clean['StockCode'].str.contains(r'^\d{5}', na=False, regex=True)]
n_after_stock  = len(df_clean)

# --- Step 3: Buat TotalPrice ---
df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']

print("=== Ringkasan Data Cleaning ===")
print(f"  Baris awal                 : {n_before:>8,}")
print(f"  Setelah hapus null CustID : {n_after_cid:>8,}  (dihapus: {n_before - n_after_cid:,})")
print(f"  Setelah filter StockCode  : {n_after_stock:>8,}  (dihapus: {n_after_cid - n_after_stock:,})")
print(f"  Total baris dihapus       : {n_before - n_after_stock:>8,} ({(n_before - n_after_stock)/n_before*100:.1f}%)")


# === Cell 4 ===# ============================================================
# Ekstraksi Var 10 (CancelFrequency) dari df_clean
# Justifikasi: df_clean masih mengandung transaksi cancel (prefix 'C')
# yang merupakan perilaku behavioral pelanggan yang ingin ditangkap
# ============================================================
df_clean['IsCanceled'] = df_clean['InvoiceNo'].str.startswith('C')
cancellations = (
    df_clean[df_clean['IsCanceled']]
    .groupby('CustomerID')['InvoiceNo']
    .nunique()
    .reset_index()
    .rename(columns={'InvoiceNo': 'CancelFrequency'})
)

# --- Filter ke transaksi sukses untuk agregasi RFM utama ---
df_success    = df_clean[(~df_clean['IsCanceled']) & (df_clean['Quantity'] > 0)].copy()
snapshot_date = df_success['InvoiceDate'].max() + timedelta(days=1)

# --- Agregasi fitur per pelanggan ---
customer_df = df_success.groupby('CustomerID').agg(
    Var1   = ('InvoiceDate',  lambda x: (snapshot_date - x.max()).days),   # Recency
    Var2   = ('InvoiceNo',    'nunique'),                                   # Frequency
    Var3   = ('Quantity',     'sum'),                                       # Total products
    Var4   = ('TotalPrice',   'sum'),                                       # Monetary
    Var5   = ('TotalPrice',   'mean'),                                      # Avg transaction cost
    Var6   = ('StockCode',    'nunique'),                                   # Unique product types
    first  = ('InvoiceDate',  'min'),
    last   = ('InvoiceDate',  'max')
).reset_index()

# Var7: Rata-rata hari antar pembelian
# Dihitung dari selisih (last - first) / (frequency - 1), minimal 1 hari
customer_df['Var7'] = (
    (customer_df['last'] - customer_df['first']).dt.days
    / (customer_df['Var2'] - 1).clip(lower=1)
)

# Var8: Perkiraan hari pembelian berikutnya = Recency + Var7
customer_df['Var8'] = customer_df['Var1'] + customer_df['Var7']

# Var9: Asal negara UK (binary 0/1)
uk_customers = (
    df_success[df_success['Country'] == 'United Kingdom']
    .groupby('CustomerID')
    .size()
    .reset_index(name='uk_count')
)
customer_df = customer_df.merge(uk_customers[['CustomerID']], on='CustomerID', how='left', indicator=True)
customer_df['Var9'] = (customer_df['_merge'] == 'both').astype(int)
customer_df = customer_df.drop(columns=['_merge'])

# Var10: Frekuensi pembatalan transaksi
customer_df = customer_df.merge(cancellations, on='CustomerID', how='left')
customer_df['CancelFrequency'] = customer_df['CancelFrequency'].fillna(0)
customer_df = customer_df.rename(columns={'CancelFrequency': 'Var10'})

# Var11: Rata-rata pengeluaran bulanan
customer_df['CustomerAgeMonths'] = (
    (customer_df['last'] - customer_df['first']).dt.days / 30
).clip(lower=1)
customer_df['Var11'] = customer_df['Var4'] / customer_df['CustomerAgeMonths']

# --- Finalisasi: ambil hanya 11 fitur, set index ---
feature_cols = [f'Var{i}' for i in range(1, 12)]
customer_df  = customer_df.set_index('CustomerID')[feature_cols]

print(f"Jumlah pelanggan unik : {len(customer_df):,}")
print(f"Fitur yang diekstrak  : {list(customer_df.columns)}")
print(f"\nStatistik deskriptif (nilai mentah):")
customer_df.describe().round(2)


# === Cell 5 ===# ============================================================
# Z-Score Normalization — Persamaan (14) paper
# X = (Xo - mu) / sigma
# ============================================================
scaler     = StandardScaler()
scaled_arr = scaler.fit_transform(customer_df)
scaled_df  = pd.DataFrame(scaled_arr, columns=customer_df.columns, index=customer_df.index)

# Verifikasi hasil normalisasi
print("=== Verifikasi Z-Score Normalization ===")
print(f"\nMean tiap fitur (harus mendekati 0.0):")
print(scaled_df.mean().round(6).to_string())
print(f"\nStd tiap fitur (harus mendekati 1.0):")
print(scaled_df.std().round(6).to_string())
print(f"\nShape data ternormalisasi: {scaled_df.shape}")
scaled_df.describe().round(4)


# === Cell 6 ===# ============================================================
# Matriks Korelasi antar Fitur Pelanggan
# Sesuai Gambar 7 pada paper (Section 4.1)
# ============================================================
corr_matrix = scaled_df.corr().round(2)

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='RdYlBu_r',        # kuning = positif, biru = negatif (sesuai paper)
    fmt='.2f',
    linewidths=0.5,
    annot_kws={'size': 8},
    vmin=-1, vmax=1,
    ax=ax
)
ax.set_title('Correlation Analysis\nMatriks Korelasi Fitur Pelanggan (Var 1 — Var 11)',
             fontsize=13, pad=14)
ax.set_xticklabels([f'Var {i}' for i in range(1, 12)], rotation=45, ha='right')
ax.set_yticklabels([f'Var {i}' for i in range(1, 12)], rotation=0)
plt.tight_layout()
plt.show()

# Highlight pasangan dengan korelasi tinggi (|r| > 0.6)
high_corr = [
    (i, j, corr_matrix.loc[i, j])
    for i in corr_matrix.columns
    for j in corr_matrix.columns
    if i < j and abs(corr_matrix.loc[i, j]) > 0.6
]
if high_corr:
    print("\nPasangan fitur dengan korelasi tinggi (|r| > 0.6):")
    for a, b, r in sorted(high_corr, key=lambda x: -abs(x[2])):
        print(f"  {a:6s} <-> {b:6s}  r = {r:+.3f}")
else:
    print("\nTidak ada pasangan fitur dengan |r| > 0.6")


# === Cell 7 ===# ============================================================
# PCA — Fit pada semua komponen terlebih dahulu
# untuk menentukan jumlah komponen optimal
# Sesuai Section 3.1 dan Gambar 8a paper
# ============================================================
pca_full = PCA(random_state=42)
pca_full.fit(scaled_df)

explained_var   = pca_full.explained_variance_ratio_ * 100
cumulative_var  = np.cumsum(explained_var)
n_components_total = len(explained_var)

# --- Gambar 8a: Cumulative Variance Contribution of PCA ---
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    range(1, n_components_total + 1), cumulative_var,
    marker='o', markersize=6, linewidth=1.8,
    color='#2980b9', label='Cumulative Variance'
)

# Tandai setiap titik dengan nilai
for i, val in enumerate(cumulative_var):
    ax.annotate(
        f'{val:.1f}%',
        (i + 1, val),
        textcoords='offset points',
        xytext=(-8, 6),
        fontsize=8,
        color='#2c3e50'
    )

# Garis threshold 90%
ax.axhline(90, color='red', linestyle='--', linewidth=1.2, label='Threshold 90%')

# Temukan komponen ke berapa pertama kali ≥ 90%
n_comp_90 = int(np.argmax(cumulative_var >= 90)) + 1
ax.axvline(n_comp_90, color='orange', linestyle=':', linewidth=1.5,
           label=f'PC-{n_comp_90} ({cumulative_var[n_comp_90-1]:.1f}%)')

ax.set_title('Cumulative Variance Contribution of PCA\n(Gambar 8a — Paper Wang 2025)',
             fontsize=12)
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('Cumulative variance contribution (%)')
ax.set_xticks(range(1, n_components_total + 1))
ax.set_ylim(30, 105)
ax.legend(fontsize=9)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print(f"Jumlah komponen yang menjelaskan ≥ 90% variance: {n_comp_90} komponen")
print(f"\nVariance per komponen (%):")
for i, v in enumerate(explained_var):
    cum = cumulative_var[i]
    mark = " ← dipilih (kumulatif ≥ 90%)" if i + 1 == n_comp_90 else ""
    print(f"  PC-{i+1:2d}  : {v:6.2f}%   kumulatif: {cum:6.2f}%{mark}")


# === Cell 8 ===# ============================================================
# Terapkan PCA dengan jumlah komponen optimal (n_comp_90)
# Sesuai kriteria paper: komponen yang menjelaskan ≥ 90% variance kumulatif
# ============================================================
n_selected   = n_comp_90   # mengikuti hasil dari sel sebelumnya (biasanya = 6)
pca_selected = PCA(n_components=n_selected, random_state=42)
pca_data     = pca_selected.fit_transform(scaled_df)

# Buat DataFrame hasil PCA
pca_cols = [f'PCA {i}' for i in range(1, n_selected + 1)]
pca_df   = pd.DataFrame(pca_data, columns=pca_cols, index=scaled_df.index)

print(f"=== Hasil PCA ===")
print(f"  Dimensi asli          : {scaled_df.shape[1]} fitur")
print(f"  Komponen terpilih     : {n_selected} PC")
print(f"  Variance dijelaskan   : {pca_selected.explained_variance_ratio_.sum()*100:.2f}%")
print(f"  Shape output PCA      : {pca_df.shape}")
print(f"\nVariance tiap PC yang dipilih:")
for i, v in enumerate(pca_selected.explained_variance_ratio_):
    print(f"  {pca_cols[i]}: {v*100:.2f}%")
pca_df.head()


# === Cell 9 ===# ============================================================
# Gambar 8b: Kontribusi Variabel Asli pada PCA (Loading Matrix)
# Sesuai paper: "principal components effectively represent
# the various characteristics of the original customers"
# ============================================================
loadings_df = pd.DataFrame(
    pca_selected.components_.T,
    index=customer_df.columns,
    columns=[f'PCA {i}' for i in range(1, n_selected + 1)]
).round(2)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    loadings_df,
    annot=True,
    cmap='RdBu_r',
    fmt='.2f',
    linewidths=0.5,
    annot_kws={'size': 9},
    vmin=-1, vmax=1,
    ax=ax
)
ax.set_title(
    'Contribution of Original Variables to PCA (Loading Matrix)\n(Gambar 8b — Paper Wang 2025)',
    fontsize=12, pad=12
)
ax.set_xlabel('Principal Components')
ax.set_ylabel('Original Variables')
plt.tight_layout()
plt.show()

print("\nLoading matrix (kontribusi variabel asli ke setiap PC):")
print(loadings_df.to_string())


# === Cell 10 ===# ============================================================
# EDA: Distribusi 11 Fitur (sebelum normalisasi — nilai mentah)
# ============================================================
fig, axes = plt.subplots(4, 3, figsize=(16, 14))
axes = axes.flatten()

colors = plt.cm.tab10(np.linspace(0, 1, 11))

for i, col in enumerate(customer_df.columns):
    axes[i].hist(
        customer_df[col].clip(upper=customer_df[col].quantile(0.99)),
        bins=50, color=colors[i], edgecolor='white', alpha=0.85
    )
    axes[i].set_title(f'{col} — {["Recency","Frequency","Total Products","Monetary","Avg Trans. Cost","Unique Products","Avg Days to Purchase","Expected Purchase Days","From UK","Cancel Frequency","Avg Monthly Spending"][i]}',
                      fontsize=9)
    axes[i].set_xlabel('Nilai (clip p99)', fontsize=8)
    axes[i].set_ylabel('Frekuensi', fontsize=8)
    axes[i].tick_params(labelsize=7)
    skew_val = customer_df[col].skew()
    axes[i].annotate(f'skew={skew_val:.2f}', xy=(0.97, 0.95), xycoords='axes fraction',
                      ha='right', va='top', fontsize=7, color='#c0392b')

# Sembunyikan axes kosong (grid 4x3 = 12, fitur = 11)
axes[11].set_visible(False)

fig.suptitle('Distribusi 11 Fitur Pelanggan (Nilai Mentah, clip p99)',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.show()


# === Cell 11 ===# ============================================================
# EDA: Distribusi 11 Fitur SETELAH Z-Score Normalization
# Memverifikasi bahwa normalisasi berjalan benar (mean≈0, std≈1)
# ============================================================
fig, axes = plt.subplots(4, 3, figsize=(16, 14))
axes = axes.flatten()

for i, col in enumerate(scaled_df.columns):
    axes[i].hist(
        scaled_df[col],
        bins=50, color=colors[i], edgecolor='white', alpha=0.85
    )
    axes[i].set_title(f'{col} (Z-score normalized)', fontsize=9)
    axes[i].set_xlabel('Nilai Ternormalisasi', fontsize=8)
    axes[i].set_ylabel('Frekuensi', fontsize=8)
    axes[i].tick_params(labelsize=7)
    mean_v = scaled_df[col].mean()
    std_v  = scaled_df[col].std()
    axes[i].annotate(f'μ={mean_v:.2f}, σ={std_v:.2f}', xy=(0.97, 0.95),
                     xycoords='axes fraction', ha='right', va='top',
                     fontsize=7, color='#27ae60')

axes[11].set_visible(False)
fig.suptitle('Distribusi 11 Fitur Pelanggan — Setelah Z-Score Normalization',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.show()


# === Cell 12 ===# ============================================================
# EDA: Visualisasi 2D pada dua PC pertama
# Proyeksi data pelanggan ke ruang komponen utama
# ============================================================
fig, ax = plt.subplots(figsize=(9, 7))
scatter = ax.scatter(
    pca_df['PCA 1'], pca_df['PCA 2'],
    alpha=0.35, s=10, c='#2980b9', edgecolors='none'
)
ax.set_xlabel(f'PCA 1 ({pca_selected.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11)
ax.set_ylabel(f'PCA 2 ({pca_selected.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11)
ax.set_title('Proyeksi Data Pelanggan pada Dua Komponen Utama Pertama (PCA 1 vs PCA 2)',
             fontsize=12)
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

print(f"Total variance dijelaskan oleh PCA 1 + PCA 2: "
      f"{sum(pca_selected.explained_variance_ratio_[:2])*100:.2f}%")


# === Cell 13 ===# ============================================================
# Export hasil preprocessing untuk tahap clustering
# ============================================================
import os, joblib

os.makedirs('../data/processed', exist_ok=True)
os.makedirs('../models', exist_ok=True)

# 1. Fitur asli (mentah) per pelanggan — 11 fitur Tabel 1
customer_df.to_csv('../data/processed/customer_features_raw.csv')

# 2. Fitur ternormalisasi (Z-score) — 11 fitur
scaled_df.to_csv('../data/processed/customer_features_scaled.csv')

# 3. Fitur setelah PCA — n_comp_90 komponen
pca_df.to_csv('../data/processed/customer_features_pca.csv')

# 4. Simpan scaler dan PCA object untuk deployment
joblib.dump(scaler,       '../models/standard_scaler.pkl')
joblib.dump(pca_selected, '../models/pca_model.pkl')

print("=== Export Selesai ===")
print(f"  customer_features_raw.csv    : {customer_df.shape[0]:,} baris × {customer_df.shape[1]} fitur")
print(f"  customer_features_scaled.csv : {scaled_df.shape[0]:,} baris × {scaled_df.shape[1]} fitur")
print(f"  customer_features_pca.csv    : {pca_df.shape[0]:,} baris × {pca_df.shape[1]} komponen")
print(f"  standard_scaler.pkl          : tersimpan di ../models/")
print(f"  pca_model.pkl                : tersimpan di ../models/")
print("\nPreprocessing & EDA selesai. Data siap untuk K-means-QLDE Clustering.")


