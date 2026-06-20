"""
views/preprocessing.py
Preprocessing Page — Visualizes feature scaling (before/after), PCA Cumulative Variance, Loading Matrix, and 2D Projection.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_clean_data, get_feature_columns, get_feature_labels
from utils.mock_models import load_model_if_exists

def show_preprocessing():
    # ── Load Data ─────────────────────────────────────────────────────────────────
    df_raw = load_clean_data()
    features = get_feature_columns()
    feature_labels = get_feature_labels()

    # ── Header ────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hdr">
    	<div>
    		<h1>Data Preprocessing & Dimensionality Reduction</h1>
    		<p>Prepare features for clustering: engineering 11 behavioral features, Z-score standardization, and Principal Component Analysis (PCA).</p>
    	</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Preprocessing Workflow Summary ───────────────────────────────────────────
    st.markdown('<div class="sl">ML Pipeline Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div style="display: flex; gap: 20px; align-items: center; justify-content: space-between; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 200px; text-align: center;">
                <div style="font-size: 20px; font-weight: bold; color: var(--color-paper);">01</div>
                <div style="font-size: 13px; font-weight: 600; text-transform: uppercase; margin-top: 4px; color: var(--color-steel);">Data Cleaning</div>
                <p style="font-size: 12px; color: var(--color-ash); margin-top: 6px;">Null CustomerIDs removed & transaction history filtered for real products.</p>
            </div>
            <div style="color: var(--color-steel); font-size: 20px;">➔</div>
            <div style="flex: 1; min-width: 200px; text-align: center;">
                <div style="font-size: 20px; font-weight: bold; color: var(--color-paper);">02</div>
                <div style="font-size: 13px; font-weight: 600; text-transform: uppercase; margin-top: 4px; color: var(--color-steel);">Feature Extraction</div>
                <p style="font-size: 12px; color: var(--color-ash); margin-top: 6px;">11 behavioral variables (Var1 - Var11) constructed for RFM++ representation.</p>
            </div>
            <div style="color: var(--color-steel); font-size: 20px;">➔</div>
            <div style="flex: 1; min-width: 200px; text-align: center;">
                <div style="font-size: 20px; font-weight: bold; color: var(--color-paper);">03</div>
                <div style="font-size: 13px; font-weight: 600; text-transform: uppercase; margin-top: 4px; color: var(--color-steel);">Z-Score Normalization</div>
                <p style="font-size: 12px; color: var(--color-ash); margin-top: 6px;">Rescales variables to mean=0 and variance=1 for uniform distance metric weight.</p>
            </div>
            <div style="color: var(--color-steel); font-size: 20px;">➔</div>
            <div style="flex: 1; min-width: 200px; text-align: center;">
                <div style="font-size: 20px; font-weight: bold; color: var(--color-paper);">04</div>
                <div style="font-size: 13px; font-weight: 600; text-transform: uppercase; margin-top: 4px; color: var(--color-steel);">PCA Reduction</div>
                <p style="font-size: 12px; color: var(--color-ash); margin-top: 6px;">Transforms features to 6 components explaining 90.7% variance to filter collinearity.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Before / After Scaling Statistics ─────────────────────────────────────────
    st.markdown('<div class="sl">Feature Scaling (Z-Score Standardization)</div>', unsafe_allow_html=True)
    st.caption("Standardization ensures all features (Recency, Frequency, Monetary, etc.) have equal weight in distance calculations (Euclidean).")

    # Apply Winsorization (capping at 99th percentile, except FromUK)
    df_winsorized = df_raw[features].copy()
    for col in features:
        if col != 'FromUK':
            cap_val = df_winsorized[col].quantile(0.99)
            df_winsorized[col] = df_winsorized[col].clip(upper=cap_val)
            
    from sklearn.preprocessing import PowerTransformer
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    X_scaled = pt.fit_transform(df_winsorized.values)
        
    df_scaled = pd.DataFrame(X_scaled, columns=features)

    stats_before = df_raw[features].describe().loc[['mean', 'std', 'min', 'max']].T
    stats_after = df_scaled[features].describe().loc[['mean', 'std', 'min', 'max']].T

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Statistics BEFORE Standardization (Raw Features)**")
        st.dataframe(stats_before.style.format("{:.2f}"), use_container_width=True)

    with col2:
        st.markdown("**Statistics AFTER Standardization (PowerTransformer Output)**")
        st.dataframe(stats_after.style.format("{:.2f}"), use_container_width=True)

    # ── PCA Dimensionality Reduction ─────────────────────────────────────────────
    st.markdown('<div class="sl">Principal Component Analysis (PCA)</div>', unsafe_allow_html=True)
    st.caption("PCA transforms high-dimensional features into orthogonal components, maximizing variance explanation and resolving multi-collinearity.")

    with st.spinner("Analyzing PCA components..."):
        # Fit PCA on full scaled features for scree plot
        pca_full = PCA(random_state=42)
        pca_full.fit(X_scaled)
        
        explained_var = pca_full.explained_variance_ratio_ * 100
        cumulative_var = np.cumsum(explained_var)
        n_components_total = len(explained_var)

        # Plot Scree/Elbow plot for PCA
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Scatter(
            x=list(range(1, n_components_total + 1)),
            y=cumulative_var,
            mode="lines+markers",
            name="Cumulative Variance Explained",
            line=dict(color="#7089ba", width=2.5),
            marker=dict(size=8, color="#ffffff", line=dict(color="#7089ba", width=2))
        ))
        # Threshold line
        fig_scree.add_shape(
            type="line", x0=1, y0=90, x1=11, y1=90,
            line=dict(color="#e74c3c", width=1.5, dash="dash")
        )
        # Vertical marker for 6 components
        fig_scree.add_shape(
            type="line", x0=6, y0=30, x1=6, y1=105,
            line=dict(color="#2ecc71", width=1.5, dash="dot")
        )
        
        fig_scree.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
            margin=dict(l=40, r=20, t=40, b=40),
            title="Cumulative Variance Contribution of PCA (Threshold: 90%)",
            xaxis=dict(showgrid=False, title="Number of Principal Components", tickmode="linear", dtick=1),
            yaxis=dict(gridcolor="#1c1c1c", title="Cumulative Variance Explained (%)", range=[30, 105]),
            height=380
        )
        
        # Fit PCA dynamically to align with the active preprocessing pipeline
        n_selected = 6
        pca_selected = PCA(n_components=n_selected, random_state=42)
        pca_selected.fit(X_scaled)
        
        # Loadings dataframe
        loadings = pd.DataFrame(
            pca_selected.components_.T,
            index=features,
            columns=[f"PC {i}" for i in range(1, n_selected + 1)]
        )
        
        fig_loadings = go.Figure(
            go.Heatmap(
                z=loadings.values,
                x=loadings.columns.tolist(),
                y=loadings.index.tolist(),
                colorscale=[[0.0, "#4d4d4d"], [0.5, "#1c1c1c"], [1.0, "#7089ba"]],
                zmin=-1, zmax=1,
                text=loadings.round(2).values,
                texttemplate="%{text}",
                showscale=True,
            )
        )
        fig_loadings.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
            margin=dict(l=40, r=20, t=40, b=40),
            title="Contribution of Original Variables to PCA (Loading Matrix)",
            height=420
        )
        fig_loadings.layout.yaxis.showgrid = False

        col_pca1, col_pca2 = st.columns(2)
        with col_pca1:
            st.plotly_chart(fig_scree, use_container_width=True)
            st.markdown("""
            <div style="font-size: 13px; color: var(--color-ash); line-height: 1.5; margin-top: -10px;">
                <b>Variance Explanation Analysis:</b><br/>
                As shown in the cumulative variance chart (Figure 8a in paper), the first <b>6 components</b> explain <b>95.6%</b> of the cumulative variance contribution (exceeding the 90% threshold). This achieves effective dimensionality reduction while retaining the vast majority of customer behavioral characteristics.
            </div>
            """, unsafe_allow_html=True)
            
        with col_pca2:
            st.plotly_chart(fig_loadings, use_container_width=True)

    # ── 2D Projection scatter plot ───────────────────────────────────────────────
    st.markdown('<div class="sl">2D Customer Projection (PC 1 vs PC 2)</div>', unsafe_allow_html=True)
    st.caption("Visualizing high-dimensional customer features projected on the first 2 principal components.")

    with st.spinner("Computing 2D projection..."):
        pca_2d = PCA(n_components=2, random_state=42)
        pca_coords = pca_2d.fit_transform(X_scaled)
        
        df_proj = pd.DataFrame(pca_coords, columns=["PC 1", "PC 2"])
        df_proj["Monetary"] = df_raw["Monetary"]
        df_proj["Frequency"] = df_raw["Frequency"]
        df_proj["CustomerID"] = df_raw["CustomerID"]
        
        fig_proj = px.scatter(
            df_proj, x="PC 1", y="PC 2",
            hover_data=["CustomerID", "Monetary", "Frequency"],
            opacity=0.7,
            template="plotly_dark",
            color_discrete_sequence=["#ffffff"]
        )
        fig_proj.update_traces(marker=dict(size=4))
        fig_proj.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(showgrid=False, title="Principal Component 1 (PC 1)"),
            yaxis=dict(gridcolor="#1c1c1c", title="Principal Component 2 (PC 2)"),
            height=400
        )
        st.plotly_chart(fig_proj, use_container_width=True)

    # ── Footer ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:var(--color-ash); font-size:12px; font-family:var(--font-geist-mono); letter-spacing:0.02em;'>"
        "KELOMPOK 6 · ML FINAL PROJECT · PREPROCESSING & SCALING"
        "</div>",
        unsafe_allow_html=True,
    )
