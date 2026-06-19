# === Cell 1 ===# ============================================================
# Import
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 110


# === Cell 2 ===# ============================================================
# Fungsi untuk meload metrik dari folder models/
# ============================================================
def load_metrics(algo_prefix):
    try:
        sse = float(np.load(f'../models/{algo_prefix}_inertia.npy'))
        sil = float(np.load(f'../models/{algo_prefix}_silhouette.npy'))
        db  = float(np.load(f'../models/{algo_prefix}_db_score.npy'))
        ch  = float(np.load(f'../models/{algo_prefix}_ch_score.npy'))
        rt  = float(np.load(f'../models/{algo_prefix}_runtime.npy'))
        return {'SSE': sse, 'Silhouette': sil, 'DB Index': db, 'CH Index': ch, 'Runtime (s)': rt}
    except FileNotFoundError:
        return None

algos = {
    'K-means-QLDE': 'qlde',
    'Standard K-means': 'kmeans-standard',
    'K-means-DE': 'kmeans-de',
    'K-means-PSO': 'kmeans-pso',
    'K-means-EOA': 'kmeans-eoa'
}

results = {}
missing = []

for name, prefix in algos.items():
    metrics = load_metrics(prefix)
    if metrics:
        results[name] = metrics
    else:
        missing.append(name)

if missing:
    print(f'⚠️ PERINGATAN: Hasil untuk {missing} tidak ditemukan.')
    print('Pastikan Anda telah menjalankan notebook 02.1 hingga 02.4, dan menyimpan hasil QLDE.')

df_comp = pd.DataFrame(results).T
if not df_comp.empty:
    display(df_comp.style.highlight_min(subset=['SSE', 'DB Index', 'Runtime (s)'], color='lightgreen')
                          .highlight_max(subset=['Silhouette', 'CH Index'], color='lightgreen')
                          .format('{:.4f}', subset=['Silhouette', 'DB Index'])
                          .format('{:,.2f}', subset=['SSE', 'CH Index'])
                          .format('{:.2f}', subset=['Runtime (s)']))

# === Cell 3 ===# ============================================================
# Visualisasi Bar Chart Perbandingan
# ============================================================
if not df_comp.empty:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Perbandingan Performa Clustering Berdasarkan Metrik', fontsize=16, fontweight='bold', y=1.02)
    
    colors = sns.color_palette("Set2", len(df_comp))
    
    # --- 1. SSE ---
    ax = axes[0, 0]
    bars = ax.bar(df_comp.index, df_comp['SSE'], color=colors)
    ax.set_title('Sum of Squared Errors (SSE) ↓', fontweight='bold')
    ax.set_xticklabels(df_comp.index, rotation=30, ha='right')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{bar.get_height():,.0f}", ha='center', va='bottom', fontsize=9)

    # --- 2. Silhouette ---
    ax = axes[0, 1]
    bars = ax.bar(df_comp.index, df_comp['Silhouette'], color=colors)
    ax.set_title('Silhouette Score ↑', fontweight='bold')
    ax.set_xticklabels(df_comp.index, rotation=30, ha='right')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha='center', va='bottom', fontsize=9)

    # --- 3. Davies-Bouldin ---
    ax = axes[1, 0]
    bars = ax.bar(df_comp.index, df_comp['DB Index'], color=colors)
    ax.set_title('Davies-Bouldin Index ↓', fontweight='bold')
    ax.set_xticklabels(df_comp.index, rotation=30, ha='right')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha='center', va='bottom', fontsize=9)

    # --- 4. Calinski-Harabasz ---
    ax = axes[1, 1]
    bars = ax.bar(df_comp.index, df_comp['CH Index'], color=colors)
    ax.set_title('Calinski-Harabasz Index ↑', fontweight='bold')
    ax.set_xticklabels(df_comp.index, rotation=30, ha='right')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{bar.get_height():,.0f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('../models/comparison_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('✓ Plot perbandingan disimpan → models/comparison_metrics.png')
else:
    print("Belum ada data untuk divisualisasikan.")

# === Cell 4 ===# ============================================================
# Visualisasi Kurva Konvergensi
# ============================================================
plt.figure(figsize=(10, 6))

colors_map = {
    'qlde': ('#D62828', 'K-means-QLDE'),
    'kmeans-de': ('#F77F00', 'K-means-DE (F Konstan)'),
    'kmeans-pso': ('#003049', 'K-means-PSO'),
    'kmeans-eoa': ('#4CAF50', 'K-means-EOA')
}

plotted = False
for prefix, (color, label) in colors_map.items():
    try:
        curve = np.load(f'../models/{prefix}_convergence.npy')
        plt.plot(curve, label=label, color=color, linewidth=2)
        plotted = True
    except FileNotFoundError:
        pass

if plotted:
    plt.title('Kurva Konvergensi Algoritma Metaheuristik', fontsize=14, fontweight='bold')
    plt.xlabel('Iterasi', fontsize=12)
    plt.ylabel('Sum of Squared Errors (SSE)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../models/comparison_convergence.png', dpi=150)
    plt.show()
    print('✓ Plot konvergensi disimpan → models/comparison_convergence.png')
else:
    print("Belum ada kurva konvergensi yang bisa diload.")

