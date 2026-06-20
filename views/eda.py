"""
views/eda.py
Exploratory Data Analysis — Visualizes dataset characteristics and feature distributions.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_clean_data, load_scaled_data, get_feature_columns, FEATURE_LABELS
from utils.visualizer import (
    plot_correlation_heatmap,
    plot_boxplots,
    plot_all_features_distributions,
)

def show_eda():
    # ── Load Data ─────────────────────────────────────────────────────────────────
    df = load_clean_data()
    scaled_df = load_scaled_data()
    features = get_feature_columns()

    # ── Sidebar Controls ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## EDA Controls")
        show_raw_stats = st.checkbox("Show descriptive statistics", value=True)
        show_distributions = st.checkbox("Feature Distributions (All 11 Features)", value=True)
        if show_distributions:
            dist_mode = st.radio("Distribution Mode:", ["Raw Features (clip p99)", "Normalized Features (Z-Score)"], index=0)
        show_corr = st.checkbox("Correlation Heatmap", value=True)
        show_boxplot = st.checkbox("Box Plots (Outliers)", value=True)
        st.markdown("---")
        st.caption("Kelompok 6 · ML Final Project")

    # ── Header ────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hdr">
        <div>
            <h1>Exploratory Data Analysis</h1>
            <p>Analyze feature distributions, correlations, and outliers of the retail customer dataset.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Dataset Summary Cards (Lime Highlight) ──────────────────────────────
    st.markdown('<div class="sl">Dataset Metadata Summary</div>', unsafe_allow_html=True)
    summary_cards = [
        ("Total Customers", f"{len(df):,}", "Records in dataset"),
        ("Feature Count", f"{len(features)}", "Engineered ML features"),
        ("Missing Values", "0", "All null entries imputed/resolved"),
        ("Duplicates", "0", "Duplicate records removed"),
    ]

    cards_html = '<div class="metric-grid cols-4">'
    for label, value, desc in summary_cards:
        cards_html += f'<div class="mc-highlight">' \
                      f'<div class="mc-val">{value}</div>' \
                      f'<div class="mc-lbl">{label}</div>' \
                      f'<div style="font-size: 12px; color: var(--color-ash); margin-top: 6px; line-height: 1.2;">{desc}</div>' \
                      f'</div>'
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # ── Descriptive Stats ─────────────────────────────────────────────────────────
    if show_raw_stats:
        st.markdown('<div class="sl">Descriptive Statistics</div>', unsafe_allow_html=True)
        tab_raw, tab_scaled = st.tabs(["Raw Features (Nilai Mentah)", "Normalized Features (Z-Score/Yeo-Johnson)"])
        with tab_raw:
            st.dataframe(df[features].describe().T.style.format("{:.2f}"), use_container_width=True)
        with tab_scaled:
            st.dataframe(scaled_df[features].describe().T.style.format("{:.4f}"), use_container_width=True)

    # ── Feature Distributions ─────────────────────────────────────────────────────
    if show_distributions:
        st.markdown(f'<div class="sl">Feature Distributions — {dist_mode}</div>', unsafe_allow_html=True)
        if "Normalized" in dist_mode:
            st.plotly_chart(plot_all_features_distributions(scaled_df, features, scaled=True), use_container_width=True)
        else:
            st.plotly_chart(plot_all_features_distributions(df, features, scaled=False), use_container_width=True)

        # Interactive single feature distribution selector
        st.markdown("**Analyze Specific Feature Distribution:**")
        selected_dist_feat = st.selectbox("Select Feature for Interactive Histogram:", features, index=2)
        
        detail_df = scaled_df if "Normalized" in dist_mode else df
        fig_hist = px.histogram(
            detail_df, x=selected_dist_feat, 
            color_discrete_sequence=["#7089ba"], 
            opacity=0.8,
            template="plotly_dark"
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#1c1c1c")
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Correlation ───────────────────────────────────────────────────────────────
    if show_corr:
        st.markdown('<div class="sl">Feature Correlation Matrix (Normalized Features)</div>', unsafe_allow_html=True)
        st.caption("Matriks korelasi antar fitur yang telah dinormalisasi menggunakan PowerTransformer (Yeo-Johnson). Sesuai dengan Gambar 7 pada paper.")
        st.plotly_chart(plot_correlation_heatmap(scaled_df, features), use_container_width=True)
        
        # Highlight high correlation pairs (|r| > 0.6)
        corr_matrix = scaled_df[features].corr()
        high_corr = [
            (i, j, corr_matrix.loc[i, j])
            for i in features
            for j in features
            if i < j and abs(corr_matrix.loc[i, j]) > 0.6
        ]
        
        if high_corr:
            st.markdown("**Pasangan Fitur dengan Korelasi Tinggi (|r| > 0.6):**")
            cols_corr = st.columns(2)
            for idx_c, (a, b, r) in enumerate(sorted(high_corr, key=lambda x: -abs(x[2]))):
                col_to_use = cols_corr[idx_c % 2]
                label_a = FEATURE_LABELS.get(a, a)
                label_b = FEATURE_LABELS.get(b, b)
                col_to_use.markdown(
                    f"""
                    <div class="glass-card" style="margin-bottom: 8px; padding: 12px; border-left: 3px solid {'#e74c3c' if r < 0 else '#2ecc71'};">
                        <span style="color: var(--color-steel); font-size: 13px;">{label_a} &harr; {label_b}</span>
                        <div style="font-weight: bold; color: {'#e74c3c' if r < 0 else '#2ecc71'}; font-size: 16px;">r = {r:+.2f}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

    # ── Box Plots ─────────────────────────────────────────────────────────────────
    if show_boxplot:
        st.markdown('<div class="sl">Outlier Identification Box Plots</div>', unsafe_allow_html=True)
        selected_feats = st.multiselect(
            "Select features to plot:", features, default=["Recency", "Frequency", "Monetary"]
        )
        if selected_feats:
            st.plotly_chart(plot_boxplots(df, selected_feats), use_container_width=True)

    # ── Footer ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:var(--color-ash); font-size:12px; font-family:var(--font-geist-mono); letter-spacing:0.02em;'>"
        "KELOMPOK 6 · ML FINAL PROJECT · EXPLORATORY DATA ANALYSIS"
        "</div>",
        unsafe_allow_html=True,
    )
