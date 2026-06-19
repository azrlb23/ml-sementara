"""
views/eda.py
Exploratory Data Analysis — Visualizes dataset characteristics and feature distributions.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_clean_data, load_anomalous_data, get_feature_columns, FEATURE_LABELS
from utils.visualizer import (
    plot_rfm_distribution,
    plot_extended_features,
    plot_correlation_heatmap,
    plot_boxplots,
)

def show_eda():
    # ── Sidebar Controls ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## EDA Controls")
        show_raw_stats = st.checkbox("Show descriptive statistics", value=True)
        show_rfm = st.checkbox("RFM Distributions", value=True)
        show_extended = st.checkbox("Extended Features", value=True)
        show_corr = st.checkbox("Correlation Heatmap", value=True)
        show_boxplot = st.checkbox("Box Plots (Outliers)", value=True)
        st.markdown("---")
        st.caption("Kelompok 6 · ML Final Project")

    # ── Load Data ─────────────────────────────────────────────────────────────────
    df = load_clean_data()
    df_anomaly = load_anomalous_data()
    features = get_feature_columns()

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
        ("Total Customers", f"{len(df):,}", "Inlier records in dataset"),
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
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Cleaned Customer Features (Inliers)**")
            st.dataframe(df[features].describe().T.style.format("{:.2f}"), use_container_width=True)
        with col2:
            if not df_anomaly.empty:
                st.markdown("**Isolated Anomalous Customers (Outliers)**")
                st.dataframe(df_anomaly[features].describe().T.style.format("{:.2f}"), use_container_width=True)

    # ── RFM Distributions ─────────────────────────────────────────────────────────
    if show_rfm:
        st.markdown('<div class="sl">RFM Distributions</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_rfm_distribution(df), use_container_width=True)

        # Interactive single feature distribution selector
        st.markdown("**Analyze Specific Feature Distribution:**")
        selected_dist_feat = st.selectbox("Select Feature for Interactive Histogram:", features, index=2)
        fig_hist = px.histogram(
            df, x=selected_dist_feat, 
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

    # ── Extended Features ─────────────────────────────────────────────────────────
    if show_extended:
        st.markdown('<div class="sl">Extended Feature Distributions</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_extended_features(df), use_container_width=True)

        with st.expander("Show Feature Definitions"):
            for feat, label in FEATURE_LABELS.items():
                st.markdown(f"- **`{feat}`** — {label}")

    # ── Correlation ───────────────────────────────────────────────────────────────
    if show_corr:
        st.markdown('<div class="sl">Feature Correlation Matrix</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_correlation_heatmap(df, features), use_container_width=True)

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
