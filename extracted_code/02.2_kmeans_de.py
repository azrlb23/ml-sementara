# === Cell 1 ===# ============================================================
# Import
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

RANDOM_SEED = 42
K_OPTIMAL   = 6
ALGO_NAME   = 'kmeans-de'
np.random.seed(RANDOM_SEED)

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 110
CLUSTER_COLORS = ['#E63946','#2A9D8F','#E9C46A','#264653','#F4A261','#A8DADC']
print(f'Algoritma: K-means-DE | K={K_OPTIMAL} | F=0.5 (konstan)')

# === Cell 2 ===# ============================================================
# Load Data
# ============================================================
df_pca    = pd.read_csv('../data/processed/customer_features_pca.csv', index_col='CustomerID')
df_scaled = pd.read_csv('../data/processed/customer_features_scaled.csv', index_col='CustomerID')
df_raw    = pd.read_csv('../data/processed/customer_features_raw.csv', index_col='CustomerID')

X_pca = df_pca.values
print(f'Data: {X_pca.shape} | {len(df_pca):,} pelanggan, 6 PC')

# === Cell 3 ===# ============================================================
# Kelas KMeansDE
# DE murni: F KONSTAN (tidak ada Q-learning adaptive)
# Semua bagian lain SAMA dengan QLDE:
#   - Inisialisasi: Logistic Chaotic
#   - Mutasi: cluster-guided (top 50%)
#   - Crossover: binomial
#   - Seleksi: greedy
# ============================================================

class KMeansDE:
    """
    K-means dengan Differential Evolution (DE/cluster-guided/1/bin).
    Scaling factor F KONSTAN — tanpa adaptive Q-learning.
    Memungkinkan isolasi kontribusi Q-learning dalam QLDE.
    """
    def __init__(self, n_clusters=6, pop_size=30, max_iter=100,
                 F=0.5, Cr=0.9, mu=3.9, random_state=None):
        self.n_clusters   = n_clusters
        self.pop_size     = pop_size
        self.max_iter     = max_iter
        self.F            = F       # KONSTAN
        self.Cr           = Cr
        self.mu           = mu
        self.random_state = random_state

    def _logistic_chaotic(self, n_pop, dim):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        phi_0  = np.random.rand(dim)
        pop    = np.zeros((n_pop, dim))
        pop[0] = phi_0
        for i in range(1, n_pop):
            phi    = pop[i-1]
            pop[i] = self.mu * phi * (1 - phi)
        return pop

    def _fitness(self, centers, X):
        cr = centers.reshape(self.n_clusters, -1)
        d  = np.linalg.norm(X[:,np.newaxis,:] - cr[np.newaxis,:,:], axis=2)
        lb = np.argmin(d, axis=1)
        sse = 0.0
        for l in range(self.n_clusters):
            mask = lb == l
            if mask.sum() > 0:
                sse += np.sum((X[mask] - X[mask].mean(axis=0))**2)
        return sse

    def fit(self, X):
        n, d   = X.shape
        dim    = self.n_clusters * d
        LB     = np.tile(X.min(axis=0), self.n_clusters)
        UB     = np.tile(X.max(axis=0), self.n_clusters)
        pop    = LB + self._logistic_chaotic(self.pop_size, dim) * (UB - LB)
        fit    = np.array([self._fitness(p, X) for p in pop])
        bi     = np.argmin(fit)
        best_s = pop[bi].copy()
        best_f = fit[bi]
        self.convergence_curve_ = [best_f]

        for k in range(1, self.max_iter + 1):
            top50 = np.argsort(fit)[:self.pop_size//2]
            for i in range(self.pop_size):
                Lk = np.random.choice(top50)
                cands = [j for j in range(self.pop_size) if j != i and j != Lk]
                i1, i2 = np.random.choice(cands, size=2, replace=False)
                # Mutasi — F KONSTAN
                v  = pop[Lk] + self.F * (pop[i1] - pop[i2])
                jr = np.random.randint(dim)
                r2 = np.random.rand(dim)
                u  = np.where((r2 <= self.Cr) | (np.arange(dim) == jr), v, pop[i])
                u  = np.clip(u, LB, UB)
                fu = self._fitness(u, X)
                if fu < fit[i]:
                    pop[i], fit[i] = u, fu
            bi = np.argmin(fit)
            if fit[bi] < best_f:
                best_f, best_s = fit[bi], pop[bi].copy()
            self.convergence_curve_.append(best_f)

        init_c = best_s.reshape(self.n_clusters, d)
        km = KMeans(n_clusters=self.n_clusters, init=init_c, n_init=1,
                    max_iter=300, random_state=self.random_state)
        km.fit(X)
        self.labels_          = km.labels_
        self.cluster_centers_ = km.cluster_centers_
        self.inertia_         = km.inertia_
        self.best_de_sse_     = best_f
        return self

print('Kelas KMeansDE berhasil didefinisikan.')

# === Cell 4 ===# ============================================================
# Jalankan K-means-DE
# ============================================================
t_start = time.time()

model = KMeansDE(
    n_clusters   = K_OPTIMAL,
    pop_size     = 30,
    max_iter     = 100,
    F            = 0.5,   # KONSTAN
    Cr           = 0.9,
    mu           = 3.9,
    random_state = RANDOM_SEED
)
model.fit(X_pca)

t_elapsed = time.time() - t_start
labels    = model.labels_

# Metrik
sse = model.inertia_
sil = silhouette_score(X_pca, labels)
db  = davies_bouldin_score(X_pca, labels)
ch  = calinski_harabasz_score(X_pca, labels)

print(f'✓ Selesai dalam {t_elapsed:.2f} detik')
print(f'  SSE DE (evolusi)     : {model.best_de_sse_:,.2f}')
print(f'  SSE K-means final    : {sse:,.2f}')
print(f'  Silhouette Score     : {sil:.4f}')
print(f'  Davies-Bouldin Index : {db:.4f}')
print(f'  Calinski-Harabasz    : {ch:.2f}')
print(f'\nDistribusi Cluster:')
for c in range(K_OPTIMAL):
    cnt = (labels == c).sum()
    pct = cnt / len(labels) * 100
    bar = '█' * int(pct / 2)
    print(f'  Cluster {c+1}: {cnt:4d} pelanggan ({pct:.1f}%)  {bar}')

# === Cell 5 ===# ============================================================
# Visualisasi: Scatter + Kurva Konvergensi
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle(f'K-means-DE — Hasil Clustering (K={K_OPTIMAL}, F=0.5 konstan)', fontsize=13, fontweight='bold')

# Panel 1: Scatter PC1 vs PC2
ax1 = axes[0]
for c in range(K_OPTIMAL):
    mask = labels == c
    ax1.scatter(X_pca[mask,0], X_pca[mask,1], c=CLUSTER_COLORS[c],
                label=f'C{c+1}', s=15, alpha=0.5, edgecolors='none')
ax1.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1],
            c='black', marker='*', s=250, zorder=10)
ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2')
ax1.set_title('(a) PC1 vs PC2')
ax1.legend(fontsize=8)

# Panel 2: Kurva konvergensi
ax2 = axes[1]
ax2.plot(model.convergence_curve_, color='#2A9D8F', linewidth=2)
ax2.set_xlabel('Iterasi'); ax2.set_ylabel('SSE')
ax2.set_title('(b) Kurva Konvergensi DE')
ax2.grid(True, alpha=0.3)

# Panel 3: Distribusi cluster
ax3 = axes[2]
counts = [(labels == c).sum() for c in range(K_OPTIMAL)]
bars = ax3.bar([f'C{c+1}' for c in range(K_OPTIMAL)], counts,
               color=CLUSTER_COLORS, edgecolor='white', linewidth=1.2)
for bar, cnt in zip(bars, counts):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f'{cnt:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax3.set_xlabel('Cluster'); ax3.set_ylabel('Jumlah Pelanggan')
ax3.set_title('(c) Distribusi Cluster')
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
os.makedirs('../models', exist_ok=True)
plt.savefig(f'../models/result_{ALGO_NAME}.png', bbox_inches='tight', dpi=150)
plt.show()
print(f'✓ Plot disimpan → models/result_{ALGO_NAME}.png')

# === Cell 6 ===# ============================================================
# Simpan Hasil Clustering
# ============================================================
import os
os.makedirs('../data/Labeled', exist_ok=True)
os.makedirs('../models', exist_ok=True)

# Dataset berlabel
df_labeled = df_raw.copy()
df_labeled['Cluster_kmeans_de'] = labels
df_labeled.to_csv('../data/Labeled/hasildata_kmeans-de.csv')

print('File berhasil disimpan: ../data/Labeled/hasildata_kmeans-de.csv')


# === Cell 7 ===# ============================================================
# Setup labels for visualization
# ============================================================
df_pca['Cluster'] = labels
df_scaled['Cluster'] = labels
df_raw['Cluster'] = labels


# === Cell 8 ===# ============================================================
# Visualisasi 1: Kurva Konvergensi K-means-DE
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(model.convergence_curve_, color='#2A9D8F', linewidth=2.5, label='K-means-DE SSE')
ax.fill_between(range(len(model.convergence_curve_)),
                model.convergence_curve_, alpha=0.15, color='#2A9D8F')

ax.set_title('Kurva Konvergensi K-means-DE\n(Q-Learning Based Differential Evolution)', fontsize=13, fontweight='bold')
ax.set_xlabel('Iterasi')
ax.set_ylabel('SSE (Sum of Squared Errors)')
ax.legend(fontsize=11)

# Annotasi SSE awal dan akhir
ax.annotate(f'SSE Awal\n{model.convergence_curve_[0]:,.0f}',
            xy=(0, model.convergence_curve_[0]),
            xytext=(5, model.convergence_curve_[0] * 1.02),
            fontsize=9, color='#264653',
            arrowprops=dict(arrowstyle='->', color='#264653'))
ax.annotate(f'SSE Akhir\n{model.convergence_curve_[-1]:,.0f}',
            xy=(len(model.convergence_curve_)-1, model.convergence_curve_[-1]),
            xytext=(len(model.convergence_curve_)-30, model.convergence_curve_[-1] * 1.05),
            fontsize=9, color='#E63946',
            arrowprops=dict(arrowstyle='->', color='#E63946'))

plt.tight_layout()
plt.savefig('../models/de_convergence.png', bbox_inches='tight', dpi=150)
plt.show()

improvement = (model.convergence_curve_[0] - model.convergence_curve_[-1]) / model.convergence_curve_[0] * 100
print(f'Perbaikan SSE: {improvement:.2f}%')

# === Cell 9 ===# ============================================================
# Visualisasi 2: Proporsi Pelanggan per Cluster
# Mereplikasi Fig 10 paper
# ============================================================
cluster_counts = pd.Series(model.labels_).value_counts().sort_index()
cluster_pcts   = cluster_counts / cluster_counts.sum() * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Distribusi Pelanggan per Cluster\n(K-means-DE, K=6)', fontsize=14, fontweight='bold')

# Pie Chart
ax = axes[0]
wedges, texts, autotexts = ax.pie(
    cluster_counts.values,
    labels=[f'Cluster {c+1}' for c in cluster_counts.index],
    colors=CLUSTER_COLORS,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.75,
    wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2)
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight('bold')
ax.set_title('(a) Donut Chart Proporsi', fontsize=12)

# Bar Chart
ax = axes[1]
bars = ax.bar(
    [f'Cluster {c+1}' for c in cluster_counts.index],
    cluster_counts.values,
    color=CLUSTER_COLORS,
    edgecolor='white', linewidth=1.5
)
for bar, cnt, pct in zip(bars, cluster_counts.values, cluster_pcts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'{cnt}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_title('(b) Jumlah Pelanggan per Cluster', fontsize=12)
ax.set_ylabel('Jumlah Pelanggan')
ax.set_ylim(0, cluster_counts.max() * 1.2)

plt.tight_layout()
plt.savefig('../models/cluster_distribution_de.png', bbox_inches='tight', dpi=150)
plt.show()

# === Cell 10 ===# ============================================================
# Visualisasi 3: Scatter Plot 3D pada PC1, PC2, PC3
# Mereplikasi Fig 11 paper
# ============================================================
labels = model.labels_

fig = plt.figure(figsize=(16, 6))
fig.suptitle('Hasil Clustering K-means-DE pada Ruang Principal Component\n'
             '(Mereplikasi Fig 11 Paper — Wang, 2025)',
             fontsize=13, fontweight='bold')

# ── Panel 1: 3D scatter PC1 vs PC2 vs PC3 ──
ax1 = fig.add_subplot(121, projection='3d')
for c in range(K_OPTIMAL):
    mask = labels == c
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                c=CLUSTER_COLORS[c], label=f'Cluster {c+1}',
                s=15, alpha=0.6, edgecolors='none')

# Plot cluster centers
centers_2d = model.cluster_centers_
ax1.scatter(centers_2d[:, 0], centers_2d[:, 1], centers_2d[:, 2],
            c='black', marker='*', s=200, zorder=10, label='Centers')

ax1.set_xlabel('PC1', fontsize=9)
ax1.set_ylabel('PC2', fontsize=9)
ax1.set_zlabel('PC3', fontsize=9)
ax1.set_title('(a) 3D: PC1 vs PC2 vs PC3', fontsize=11)
ax1.legend(loc='upper left', fontsize=8, markerscale=1.5)

# ── Panel 2: 2D scatter PC1 vs PC2 ──
ax2 = fig.add_subplot(122)
for c in range(K_OPTIMAL):
    mask = labels == c
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=CLUSTER_COLORS[c], label=f'Cluster {c+1}',
                s=20, alpha=0.5, edgecolors='none')

ax2.scatter(centers_2d[:, 0], centers_2d[:, 1],
            c='black', marker='*', s=250, zorder=10, label='Centers')

ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_title('(b) 2D Projection: PC1 vs PC2', fontsize=11)
ax2.legend(fontsize=9, markerscale=1.5)

plt.tight_layout()
plt.savefig('../models/cluster_3d_scatter_de.png', bbox_inches='tight', dpi=150)
plt.show()

# === Cell 11 ===# ============================================================
# Visualisasi 4: Heatmap Cluster Centers (11 fitur, Z-score)
# Mereplikasi konsep Fig 12 paper
# ============================================================

# Hitung cluster centers pada scaled data (11 fitur)
feature_cols = [f'Var{i}' for i in range(1, 12)]
feature_desc = [
    'Recency\n(Var1)', 'Frequency\n(Var2)', 'Total Qty\n(Var3)',
    'Monetary\n(Var4)', 'Avg Cost\n(Var5)', 'Product Types\n(Var6)',
    'Avg Days\n(Var7)', 'Next Purchase\n(Var8)', 'UK Origin\n(Var9)',
    'Cancellation\n(Var10)', 'Monthly Spend\n(Var11)'
]

# Cluster centers pada fitur Z-score
centers_df = df_scaled.groupby('Cluster')[feature_cols].mean()
centers_df.index = [f'Cluster {i+1}' for i in centers_df.index]

fig, ax = plt.subplots(figsize=(14, 6))

im = ax.imshow(centers_df.values, cmap='RdYlGn', aspect='auto',
               vmin=centers_df.values.min(), vmax=centers_df.values.max())

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Z-score (Nilai Fitur Ternormalisasi)', fontsize=10)

ax.set_xticks(range(len(feature_cols)))
ax.set_xticklabels(feature_desc, fontsize=9)
ax.set_yticks(range(len(centers_df)))
ax.set_yticklabels(centers_df.index, fontsize=10)

# Nilai pada sel
for i in range(len(centers_df)):
    for j in range(len(feature_cols)):
        val = centers_df.values[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=7.5, color=color, fontweight='bold')

ax.set_title('Heatmap Cluster Centers — Nilai Z-score per Fitur\n'
             '(Mereplikasi konsep Fig 12 Paper — Wang, 2025)',
             fontsize=13, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('../models/cluster_centers_heatmap_de.png', bbox_inches='tight', dpi=150)
plt.show()

print('\nCluster Centers (Z-score):')
print(centers_df.round(3).to_string())

# === Cell 12 ===# ============================================================
# Visualisasi 5: Radar Chart (Spider Plot) tiap cluster
# ============================================================

# Gunakan nilai mentah (raw) ternormalisasi ke [0,1] untuk radar
centers_raw = df_raw.groupby('Cluster')[feature_cols].mean()
centers_raw.index = [f'Cluster {i+1}' for i in centers_raw.index]

# Min-max normalize cluster centers untuk tampilan radar
from sklearn.preprocessing import MinMaxScaler
scaler_mm = MinMaxScaler()
centers_norm = pd.DataFrame(
    scaler_mm.fit_transform(centers_raw.values),
    index=centers_raw.index,
    columns=feature_cols
)

# Short labels untuk radar
radar_labels = ['Recency', 'Frequency', 'Total Qty', 'Monetary', 'Avg Cost',
                'Prod Types', 'Avg Days', 'Next Buy', 'UK Origin', 'Cancellation', 'Monthly $']

N = len(radar_labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the loop

fig, axes = plt.subplots(2, 3, figsize=(16, 10), subplot_kw=dict(polar=True))
fig.suptitle('Profil Radar 6 Cluster — Karakteristik Pelanggan\n'
             '(K-means-DE, K=6 — Wang, 2025)',
             fontsize=14, fontweight='bold', y=1.01)

cluster_profiles = [
    ('Cluster 1', 'High-Value Customer\n(VIP Pelanggan Utama)'),
    ('Cluster 2', 'Price-Sensitive Customer\n(Konsumen Harga)'),
    ('Cluster 3', 'High-Expectation (UK)\n(Ekspektasi Tinggi)'),
    ('Cluster 4', 'Uncertain Buyer\n(Pembeli Tidak Pasti)'),
    ('Cluster 5', 'Cautious Consumer\n(Konsumen Berhati-hati)'),
    ('Cluster 6', 'Balanced Customer\n(Pelanggan Seimbang)'),
]

for ax, (cname, profile), color in zip(axes.flatten(), cluster_profiles, CLUSTER_COLORS):
    values = centers_norm.loc[cname].values.tolist()
    values += values[:1]  # close the loop
    
    ax.plot(angles, values, color=color, linewidth=2.5)
    ax.fill(angles, values, color=color, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=7.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['0.25', '0.50', '0.75'], fontsize=6.5, color='gray')
    
    cnt = (df_pca['Cluster'] == int(cname.split()[-1]) - 1).sum()
    pct = cnt / len(df_pca) * 100
    ax.set_title(f'{cname} ({cnt}, {pct:.1f}%)\n{profile}',
                 fontsize=9, fontweight='bold', color=color, pad=15)

plt.tight_layout()
plt.savefig('../models/cluster_radar_de.png', bbox_inches='tight', dpi=150)
plt.show()

# === Cell 13 ===# ============================================================
# Evaluasi Metrik Clustering
# ============================================================
labels = model.labels_

sse     = model.inertia_
sil     = silhouette_score(X_pca, labels)
db      = davies_bouldin_score(X_pca, labels)
ch      = calinski_harabasz_score(X_pca, labels)

print('=' * 55)
print('  EVALUASI KUALITAS CLUSTERING — K-means-DE (K=6)')
print('=' * 55)
print(f'  SSE (Inertia)          : {sse:>15,.2f}  ↓ (kecil = baik)')
print(f'  Silhouette Score       : {sil:>15.4f}  ↑ (mendekati 1 = baik)')
print(f'  Davies-Bouldin Index   : {db:>15.4f}  ↓ (kecil = baik)')
print(f'  Calinski-Harabasz Index: {ch:>15.2f}  ↑ (besar = baik)')
print('=' * 55)

# Simpan metrik untuk perbandingan nanti
results_de = {
    'Algorithm'  : 'K-means-DE (Paper)',
    'SSE'        : sse,
    'Silhouette' : sil,
    'Davies-Bouldin': db,
    'Calinski-Harabasz': ch,
    'K'          : K_OPTIMAL
}
print('\nHasil disimpan untuk perbandingan dengan algoritma lain.')

# === Cell 14 ===# ============================================================
# Analisis Karakteristik Cluster
# ============================================================
feature_desc_short = {
    'Var1'  : 'Recency (hari sejak transaksi terakhir)',
    'Var2'  : 'Frequency (jumlah transaksi)',
    'Var3'  : 'Total produk dibeli',
    'Var4'  : 'Monetary (total pengeluaran)',
    'Var5'  : 'Rata-rata biaya per transaksi',
    'Var6'  : 'Jumlah tipe produk',
    'Var7'  : 'Rata-rata hari antar pembelian',
    'Var8'  : 'Perkiraan hari pembelian berikutnya',
    'Var9'  : 'Asal UK (0/1)',
    'Var10' : 'Frekuensi pembatalan',
    'Var11' : 'Rata-rata pengeluaran bulanan',
}

centers_raw_display = df_raw.groupby('Cluster')[feature_cols].mean().round(2)
centers_raw_display.index = [f'Cluster {i+1}' for i in centers_raw_display.index]
centers_raw_display.columns = [f'{v} ({feature_desc_short[v][:25]}...)' if len(feature_desc_short[v]) > 25
                                else f'{v} ({feature_desc_short[v]})'
                                for v in feature_cols]

# Deskripsi cluster sesuai paper Section 4.2
cluster_descriptions = {
    'Cluster 1': '🏆 HIGH-VALUE (VIP) — Pengeluaran bulanan tinggi, produk terbanyak, cancellation rendah.\n'
                 '   → Kelompok terpenting bagi perusahaan (revenue terbesar). Perlu program loyalitas eksklusif.',
    'Cluster 2': '💰 PRICE-SENSITIVE — Volume pembelian tinggi tapi frequency & spending rendah.\n'
                 '   → Konsumen rasional/hemat. Strategi: promo diskon, bundle harga.',
    'Cluster 3': '🇬🇧 HIGH-EXPECTATION (UK) — Mayoritas pelanggan UK, cancellation sedang.\n'
                 '   → Ekspektasi tinggi pada kualitas produk. Strategi: klarifikasi info produk.',
    'Cluster 4': '❓ UNCERTAIN BUYER — Cancellation tinggi, berasal dari non-UK.\n'
                 '   → Tidak pasti dalam keputusan pembelian. Strategi: optimalkan proses pembayaran.',
    'Cluster 5': '🤔 CAUTIOUS CONSUMER — Recency tinggi, frekuensi rendah, berhati-hati.\n'
                 '   → Banyak pertimbangan sebelum beli. Strategi: promosi akhir pekan, ulasan produk.',
    'Cluster 6': '⚖️  BALANCED CUSTOMER — Performa seimbang di semua indikator.\n'
                 '   → Tanpa preferensi ekstrem. Strategi: marketing diversifikasi.',
}

print('=' * 70)
print('  KARAKTERISTIK 6 CLUSTER — K-means-DE')
print('  Referensi: Section 4.2, Fig 12 — Wang (2025)')
print('=' * 70)

for c_name, desc in cluster_descriptions.items():
    c_idx = int(c_name.split()[-1]) - 1
    cnt   = (df_pca['Cluster'] == c_idx).sum()
    pct   = cnt / len(df_pca) * 100
    print(f'\n{c_name} — {cnt} pelanggan ({pct:.1f}%)')
    print(f'  {desc}')

# === Cell 15 ===# ============================================================
# Visualisasi 6: Bar Chart Perbandingan Cluster Centers (Raw)
# ============================================================
centers_raw_plot = df_raw.groupby('Cluster')[feature_cols].mean()
centers_raw_plot.index = [f'Cluster {i+1}' for i in centers_raw_plot.index]

# Pilih fitur paling informatif untuk perbandingan
key_features = ['Var1', 'Var2', 'Var4', 'Var6', 'Var10', 'Var11']
key_labels   = ['Recency\n(hari)', 'Frequency\n(transaksi)', 'Monetary\n(total spend)',
                'Prod Types', 'Cancellation\nFreq', 'Monthly\nSpend']

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Perbandingan Cluster Centers pada Fitur Kunci\n'
             '(Nilai Rata-rata Raw per Cluster)',
             fontsize=13, fontweight='bold')

for ax, feat, label in zip(axes.flatten(), key_features, key_labels):
    vals = centers_raw_plot[feat].values
    bars = ax.bar(
        [f'C{i+1}' for i in range(K_OPTIMAL)],
        vals,
        color=CLUSTER_COLORS,
        edgecolor='white', linewidth=1
    )
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.set_ylabel('Nilai Rata-rata')
    
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals) * 0.01,
                f'{v:.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('../models/cluster_features_comparison_de.png', bbox_inches='tight', dpi=150)
plt.show()

# === Cell 16 ===# Dataset berlabel
df_labeled = df_raw.copy()
df_labeled['Cluster_kmeans_de'] = labels

# Tambahkan Label Bisnis
df_labeled['Label_Bisnis'] = df_labeled['Cluster_kmeans_de'].map({
    0: 'HIGH-VALUE (VIP)',
    1: 'PRICE-SENSITIVE',
    2: 'HIGH-EXPECTATION (UK)',
    3: 'UNCERTAIN BUYER',
    4: 'CAUTIOUS CONSUMER',
    5: 'BALANCED CUSTOMER'
})

df_labeled.to_csv('../data/Labeled/hasildata_kmeans-de.csv')
print('File berhasil disimpan: ../data/Labeled/hasildata_kmeans-de.csv')


