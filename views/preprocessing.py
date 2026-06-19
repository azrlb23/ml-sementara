"""
views/preprocessing.py
Preprocessing Page — Visualizes feature scaling (before/after) and isolated anomalies using PCA.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
from utils.data_loader import load_clean_data, load_anomalous_data, get_feature_columns

def show_preprocessing():
    # ── Load Data ─────────────────────────────────────────────────────────────────
    df_inlier = load_clean_data()
    df_anomaly = load_anomalous_data()
    features = get_feature_columns()

    # ── Header ────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hdr">
        <div>
            <h1>Data Preprocessing & Scaling</h1>
            <p>Prepare features for clustering: engineering RFM++, outlier isolation, and standardization.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Outlier Summary Cards ─────────────────────────────────────────────────────
    st.markdown('<div class="sl">Outlier Summary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-grid cols-2">'
                f'<div class="mc-highlight">'
                f'<div class="mc-val">{len(df_inlier):,}</div>'
                f'<div class="mc-lbl">Inliers (Cleaned Customers)</div>'
                f'<div style="font-size: 12px; color: var(--color-ash); margin-top: 6px; line-height: 1.2;">Used to construct main segments and train prediction classifiers</div>'
                f'</div>'
                f'<div class="mc">'
                f'<div class="mc-val">{len(df_anomaly):,}</div>'
                f'<div class="mc-lbl">Isolated Anomalies (Noise)</div>'
                f'<div style="font-size: 12px; color: var(--color-ash); margin-top: 6px; line-height: 1.2;">VIP/Extreme spenders isolated by DBSCAN to prevent centroid skewing</div>'
                f'</div>'
                f'</div>', unsafe_allow_html=True)

    # ── Before / After Scaling Statistics ─────────────────────────────────────────
    st.markdown('<div class="sl">Feature Scaling (Standardization)</div>', unsafe_allow_html=True)
    st.caption("Standardization ensures all features (Recency, Frequency, Monetary, etc.) have equal weight in distance calculations (Euclidean).")

    # Compute scaled metrics
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_inlier[features])
    df_scaled = pd.DataFrame(X_scaled, columns=features)

    stats_before = df_inlier[features].describe().loc[['mean', 'std', 'min', 'max']].T
    stats_after = df_scaled[features].describe().loc[['mean', 'std', 'min', 'max']].T

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Statistics BEFORE Standardization (Raw Features)**")
        st.dataframe(stats_before.style.format("{:.2f}"), use_container_width=True)

    with col2:
        st.markdown("**Statistics AFTER Standardization (StandardScaler Output)**")
        st.dataframe(stats_after.style.format("{:.2f}"), use_container_width=True)

    # ── PCA 2D Outlier Scatter Plot ───────────────────────────────────────────────
    st.markdown('<div class="sl">2D Dimensionality Reduction (PCA) Outlier View</div>', unsafe_allow_html=True)
    st.caption("Visualizing high-dimensional customer features projected on 2 principal components.")

    with st.spinner("Computing PCA..."):
        # Combine inliers and outliers for PCA visualization
        df_inlier_copy = df_inlier.copy()
        df_anomaly_copy = df_anomaly.copy()
        
        df_inlier_copy['Status'] = 'Inlier'
        df_anomaly_copy['Status'] = 'Anomaly'
        
        combined = pd.concat([df_inlier_copy, df_anomaly_copy], ignore_index=True)
        X_comb = scaler.fit_transform(combined[features])
        
        pca = PCA(n_components=2, random_state=42)
        pca_coords = pca.fit_transform(X_comb)
        
        combined['PCA 1'] = pca_coords[:, 0]
        combined['PCA 2'] = pca_coords[:, 1]
        
        fig_pca = px.scatter(
            combined, x="PCA 1", y="PCA 2", color="Status",
            color_discrete_map={"Inlier": "#ffffff", "Anomaly": "#7089ba"},
            hover_data=["CustomerID", "Monetary", "Frequency"],
            opacity=0.75,
            template="plotly_dark"
        )
        fig_pca.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(showgrid=False, title="Principal Component 1"),
            yaxis=dict(gridcolor="#1c1c1c", title="Principal Component 2")
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    # ── Anomaly Table ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sl">Isolated Anomalous Customers Details</div>', unsafe_allow_html=True)
    st.caption("Isolated customer profiles displaying extreme spending or activity features.")
    if not df_anomaly.empty:
        st.dataframe(
            df_anomaly.sort_values(by="Monetary", ascending=False).style.format({
                "Monetary": "£{:,.2f}",
                "AvgSpending": "£{:.2f}",
                "AvgMonthlySpending": "£{:.2f}",
            }),
            use_container_width=True,
            height=280
        )
    else:
        st.info("No anomalous records loaded.")

    # ── Footer ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:var(--color-ash); font-size:12px; font-family:var(--font-geist-mono); letter-spacing:0.02em;'>"
        "KELOMPOK 6 · ML FINAL PROJECT · PREPROCESSING & SCALING"
        "</div>",
        unsafe_allow_html=True,
    )
