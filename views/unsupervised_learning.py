"""
views/unsupervised_learning.py
Unsupervised Learning Page — Compares standard K-Means with DE, PSO, EOA, and QLDE optimization.
"""

import streamlit as st
import pandas as pd
from utils.data_loader import load_clean_data
from utils.mock_models import (
    run_unsupervised_algorithm,
    get_convergence_curves,
    get_cluster_profiles,
    get_algorithm_comparison_metrics,
)
from utils.visualizer import (
    plot_scatter_3d,
    plot_scatter_2d,
    plot_cluster_radar,
    plot_algorithm_comparison,
    plot_convergence_curve,
    plot_donut_chart,
    CLUSTER_NAMES,
)

def show_unsupervised_learning():
    # ── Load Data ─────────────────────────────────────────────────────────────────
    df = load_clean_data()

    # ── Header ────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hdr">
        <div>
            <h1>Unsupervised Learning (Clustering)</h1>
            <p>Centroid optimization comparing K-Means Baseline against Evolutionary and Adaptive Metaheuristic Algorithms.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Algorithm Selector & Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Clustering Controls")
        algorithm = st.selectbox(
            "Select Algorithm to Analyze:",
            ["K-Means Standard", "K-Means + DE", "K-Means QLDE"],
            index=2,
            help="Select the clustering optimization strategy to display in detail."
        )
        n_clusters = 6
        st.info("Number of Clusters (K): **6** (Optimal selection locked based on the Elbow & Silhouette analysis in the repository)")
        st.markdown("---")
        x_axis = st.selectbox("2D Scatter X-axis", ["Recency", "Frequency", "Monetary", "AvgSpending"], index=0)
        y_axis = st.selectbox("2D Scatter Y-axis", ["Recency", "Frequency", "Monetary", "AvgSpending"], index=2)

    # ── Run Selected Clustering ───────────────────────────────────────────────────
    with st.spinner(f"Optimizing centroids using {algorithm}..."):
        df_clustered, metrics = run_unsupervised_algorithm(df, algorithm, k=n_clusters)

    # ── 4 Metric Cards (Highlight Lime for QLDE - best algorithm) ─────────────────
    st.markdown(f'<div class="sl">{algorithm} — Model Metrics</div>', unsafe_allow_html=True)

    # Determine if this algorithm yields the overall best results (QLDE at k=6 is the best)
    is_best = (algorithm == "K-Means QLDE" and n_clusters == 6)
    card_class = "mc-highlight" if is_best else "mc"

    # Metric dictionary keys
    metric_list = [
        ("SSE (Sum of Squared Errors)", f"{metrics['SSE']:,}", "Lower SSE = tighter clusters", "down"),
        ("Silhouette Score", f"{metrics['Silhouette Score']:.4f}", "Higher = better separated clusters", "up"),
        ("Davies-Bouldin Index", f"{metrics['Davies-Bouldin Index']:.3f}", "Lower = less overlapping clusters", "down"),
        ("Calinski-Harabasz Index", f"{metrics['Calinski-Harabasz Index']:.1f}", "Higher = denser, more distinct clusters", "up"),
    ]

    cards_html = '<div class="metric-grid cols-4">'
    for label, value, desc, direction in metric_list:
        badge = " (Best)" if is_best else ""
        cards_html += f'<div class="{card_class}">' \
                      f'<div class="mc-val">{value}</div>' \
                      f'<div class="mc-lbl">{label}{badge}</div>' \
                      f'<div style="font-size: 12px; color: var(--color-ash); margin-top: 6px; line-height: 1.2;">{desc}</div>' \
                      f'</div>'
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    if is_best:
        st.markdown("""
        <div class="success-banner" style="margin-top: 15px; background: rgba(112, 137, 186, 0.05);">
            <b>K-Means QLDE</b> (Q-Learning Differential Evolution) achieves the global optimal segmentation with the lowest SSE and highest silhouette scores.
        </div>
        """, unsafe_allow_html=True)

    # ── Visualizations (Scatter, Donut, Radar) ──────────────────────────────────
    st.markdown("---")
    col_vis1, col_vis2 = st.columns([3, 2])

    with col_vis1:
        st.markdown('<div class="sl">3D Customer Segment Space (RFM)</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_scatter_3d(df_clustered), use_container_width=True)

    with col_vis2:
        st.markdown('<div class="sl">Cluster Proportion (Donut Chart)</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_donut_chart(df_clustered), use_container_width=True)

    col_vis3, col_vis4 = st.columns(2)
    with col_vis3:
        st.markdown('<div class="sl">2D Feature Interaction Scatter</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_scatter_2d(df_clustered, x_axis, y_axis), use_container_width=True)

    with col_vis4:
        st.markdown('<div class="sl">Cluster Profile Radar (Normalized Features)</div>', unsafe_allow_html=True)
        profiles = get_cluster_profiles(df_clustered)
        from sklearn.preprocessing import MinMaxScaler
        feat_cols = [
            "Recency", "Frequency", "TotalProducts", "Monetary", "AvgSpending", 
            "UniqueProducts", "AvgDaysToPurchase", "ExpectedPurchaseDays", "FromUK", 
            "CancelFrequency", "AvgMonthlySpending"
        ]
        scaler = MinMaxScaler()
        normalized_profiles = profiles.copy()
        normalized_profiles[feat_cols] = scaler.fit_transform(profiles[feat_cols])
        st.plotly_chart(plot_cluster_radar(normalized_profiles), use_container_width=True)

    # ── Metaheuristic Performance Comparisons (Bar & Convergence) ────────────────
    st.markdown("---")
    col_comp1, col_comp2 = st.columns(2)

    with col_comp1:
        st.markdown('<div class="sl">Optimization Convergence Curve</div>', unsafe_allow_html=True)
        st.caption("SSE reduction profiles across iterations for K-Means + DE and K-Means QLDE.")
        df_conv = get_convergence_curves(max_iter=50)
        st.plotly_chart(plot_convergence_curve(df_conv), use_container_width=True)

    with col_comp2:
        st.markdown('<div class="sl">Performance Comparison (Selected Algorithms)</div>', unsafe_allow_html=True)
        st.caption("Detailed 4-metric comparison across the three clustering strategies.")
        comp_metrics = get_algorithm_comparison_metrics(df, k=n_clusters)
        st.plotly_chart(plot_algorithm_comparison(comp_metrics), use_container_width=True)

    # ── Cluster Segment Profiles Table ───────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sl">Cluster Profiles & Descriptions</div>', unsafe_allow_html=True)

    raw_profiles = get_cluster_profiles(df_clustered)
    raw_profiles["Segment Name"] = raw_profiles["Cluster"].map(
        lambda c: CLUSTER_NAMES.get(c, f"Cluster {c}")
    )

    cols = list(raw_profiles.columns)
    cols.insert(1, cols.pop(cols.index("Segment Name")))
    raw_profiles = raw_profiles[cols]

    st.dataframe(
        raw_profiles.style.format({
            "Recency": "{:.1f} days",
            "Frequency": "{:.1f} orders",
            "TotalProducts": "{:.0f} units",
            "Monetary": "£{:,.2f}",
            "AvgSpending": "£{:.2f}",
            "UniqueProducts": "{:.0f} products",
            "AvgDaysToPurchase": "{:.1f} days",
            "ExpectedPurchaseDays": "{:.1f} days",
            "FromUK": "{:.0f}",
            "CancelFrequency": "{:.1f} cancels",
            "AvgMonthlySpending": "£{:.2f}",
        }),
        use_container_width=True
    )

    # ── Footer ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:var(--color-ash); font-size:12px; font-family:var(--font-geist-mono); letter-spacing:0.02em;'>"
        "KELOMPOK 6 · ML FINAL PROJECT · UNSUPERVISED CLUSTERING ANALYSIS"
        "</div>",
        unsafe_allow_html=True,
    )
