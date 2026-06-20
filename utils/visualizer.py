"""
visualizer.py
Centralized chart creation functions using Plotly.
All charts follow the premium dark mode aesthetic with Lime and Lavender accents.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from utils.data_loader import FEATURE_LABELS

# ── Design Tokens ─────────────────────────────────────────────────────────────
CLUSTER_COLORS = {
    0: "#ffffff",   # High-Value (White)
    1: "#cccccc",   # Price-Sensitive (Light Gray)
    2: "#999999",   # High-Expectation-UK (Medium Gray)
    3: "#666666",   # Uncertain-Buyer (Dark Gray)
    4: "#444444",   # Cautious-Consumer (Very Dark Gray)
    5: "#7089ba",   # Balanced (Periwinkle Accent)
}

CLUSTER_NAMES = {
    0: "High-Value",
    1: "Price-Sensitive",
    2: "High-Expectation-UK",
    3: "Uncertain-Buyer",
    4: "Cautious-Consumer",
    5: "Balanced",
}

PLOTLY_TEMPLATE = "plotly_dark"
PAPER_COLOR = "rgba(0,0,0,0)"
PLOT_COLOR = "rgba(0,0,0,0)"
FONT_FAMILY = "Inter, sans-serif"

BASE_LAYOUT = dict(
    template=PLOTLY_TEMPLATE,
    paper_bgcolor=PAPER_COLOR,
    plot_bgcolor=PLOT_COLOR,
    font=dict(family=FONT_FAMILY, color="#FFFFFF"),
    margin=dict(l=60, r=20, t=50, b=50),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#1c1c1c",
        bordercolor="rgba(255,255,255,0.1)",
        font=dict(family=FONT_FAMILY, color="#FFFFFF"),
    ),
    xaxis=dict(
        showgrid=False, color="#808080",
        tickfont=dict(family="JetBrains Mono, monospace", size=10)
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#1c1c1c", zeroline=False,
        color="#808080",
        tickfont=dict(family="JetBrains Mono, monospace", size=10)
    ),
    legend=dict(
        bgcolor="rgba(28, 28, 28, 0.85)",
        bordercolor="rgba(255,255,255,0.1)", borderwidth=1,
        font=dict(size=10, family=FONT_FAMILY)
    )
)


def _apply_base(fig: go.Figure, title: str = "") -> go.Figure:
    layout_dict = BASE_LAYOUT.copy()
    layout_dict["title"] = dict(text=title, font=dict(size=16))
    fig.update_layout(layout_dict)
    return fig


# ── EDA Charts ────────────────────────────────────────────────────────────────

def plot_rfm_distribution(df: pd.DataFrame) -> go.Figure:
    """3-panel histogram: Recency, Frequency, Monetary distributions."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Recency (days)", "Frequency (transactions)", "Monetary (£)"),
    )
    features = ["Recency", "Frequency", "Monetary"]
    colors = ["#ffffff", "#808080", "#7089ba"]

    for i, (feat, color) in enumerate(zip(features, colors), start=1):
        fig.add_trace(
            go.Histogram(
                x=df[feat],
                name=feat,
                marker_color=color,
                opacity=0.85,
                nbinsx=40,
                showlegend=False,
            ),
            row=1, col=i,
        )

    layout_dict = {k: v for k, v in BASE_LAYOUT.items() if k not in ["xaxis", "yaxis"]}
    layout_dict.update(
        title="RFM Feature Distributions",
        height=350,
    )
    fig.update_layout(layout_dict)
    fig.update_xaxes(showgrid=False, color="#808080", tickfont=dict(family="JetBrains Mono, monospace", size=10))
    fig.update_yaxes(showgrid=True, gridcolor="#1c1c1c", zeroline=False, color="#808080", tickfont=dict(family="JetBrains Mono, monospace", size=10))
    fig.update_traces(marker_line_color="rgba(0,0,0,0.5)", marker_line_width=0.5)
    return fig


def plot_extended_features(df: pd.DataFrame) -> go.Figure:
    """4-panel histogram: extended features."""
    features = ["AvgSpending", "UniqueProducts", "CancelFrequency", "AvgMonthlySpending"]
    titles = ["Avg Spending / Item", "Unique Products", "Cancel Frequency", "Avg Monthly Spending"]
    colors = ["#ababab", "#ffffff", "#808080", "#7089ba"]

    fig = make_subplots(rows=1, cols=4, subplot_titles=titles)
    for i, (feat, color) in enumerate(zip(features, colors), start=1):
        fig.add_trace(
            go.Histogram(x=df[feat], name=feat, marker_color=color, opacity=0.85,
                         nbinsx=30, showlegend=False),
            row=1, col=i,
        )
    layout_dict = {k: v for k, v in BASE_LAYOUT.items() if k not in ["xaxis", "yaxis"]}
    layout_dict.update(title="Extended Feature Distributions", height=320)
    fig.update_layout(layout_dict)
    fig.update_xaxes(showgrid=False, color="#808080", tickfont=dict(family="JetBrains Mono, monospace", size=10))
    fig.update_yaxes(showgrid=True, gridcolor="#1c1c1c", zeroline=False, color="#808080", tickfont=dict(family="JetBrains Mono, monospace", size=10))
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, features: list[str]) -> go.Figure:
    """Correlation matrix heatmap using a diverging Lavender-to-Lime colormap."""
    corr = df[features].corr()
    
    # Custom Diverging Colormap: Graphite -> Carbon -> Periwinkle
    purple_to_lime = [
        [0.0, "#4d4d4d"],
        [0.5, "#1c1c1c"],
        [1.0, "#7089ba"]
    ]
    
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=purple_to_lime,
            zmin=-1, zmax=1,
            text=corr.round(2).values,
            texttemplate="%{text}",
            showscale=True,
        )
    )
    layout_dict = BASE_LAYOUT.copy()
    layout_dict.update(title="Feature Correlation Matrix", height=400)
    fig.update_layout(layout_dict)
    fig.layout.yaxis.showgrid = False
    return fig


def plot_all_features_distributions(df: pd.DataFrame, features: list[str], scaled: bool = False) -> go.Figure:
    """4x3 panel histogram showing distributions for all 11 features (raw or normalized)."""
    if not scaled:
        subplot_titles = [f"{FEATURE_LABELS.get(f, f)} (Skew: {df[f].skew():.2f})" for f in features] + [""]
    else:
        subplot_titles = [f"{FEATURE_LABELS.get(f, f)} (μ: {df[f].mean():.2f}, σ: {df[f].std():.2f})" for f in features] + [""]

    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.06
    )

    colors = ["#7089ba", "#ababab", "#ffffff", "#808080", "#7089ba", "#ababab", "#ffffff", "#808080", "#7089ba", "#ababab", "#ffffff"]

    for idx, feat in enumerate(features):
        r = (idx // 3) + 1
        c = (idx % 3) + 1

        x_data = df[feat]
        if not scaled:
            if feat != "FromUK":
                cap = x_data.quantile(0.99)
                x_data = x_data.clip(upper=cap)

        fig.add_trace(
            go.Histogram(
                x=x_data,
                name=feat,
                marker_color=colors[idx % len(colors)],
                opacity=0.85,
                nbinsx=40,
                showlegend=False,
            ),
            row=r, col=c,
        )

    layout_dict = {k: v for k, v in BASE_LAYOUT.items() if k not in ["xaxis", "yaxis"]}
    title_text = "Distribusi 11 Fitur Pelanggan (Setelah Z-Score Normalization)" if scaled else "Distribusi 11 Fitur Pelanggan (Nilai Mentah, clip p99)"
    layout_dict.update(
        title=title_text,
        height=1000,
        margin=dict(l=40, r=20, t=80, b=40)
    )
    fig.update_layout(layout_dict)
    fig.update_xaxes(showgrid=False, color="#808080", tickfont=dict(family="JetBrains Mono, monospace", size=10))
    fig.update_yaxes(showgrid=True, gridcolor="#1c1c1c", zeroline=False, color="#808080", tickfont=dict(family="JetBrains Mono, monospace", size=10))
    fig.update_traces(marker_line_color="rgba(0,0,0,0.5)", marker_line_width=0.5)
    return fig


def plot_boxplots(df: pd.DataFrame, features: list[str]) -> go.Figure:
    """Box plots for outlier visibility in dark theme."""
    fig = go.Figure()
    colors = list(CLUSTER_COLORS.values())
    for i, feat in enumerate(features):
        fig.add_trace(go.Box(
            y=df[feat], name=feat,
            marker_color=colors[i % len(colors)],
            boxmean=True,
        ))
    layout_dict = BASE_LAYOUT.copy()
    layout_dict.update(title="Feature Box Plots (Outlier View)", height=400)
    fig.update_layout(layout_dict)
    return fig


# ── Clustering Charts ─────────────────────────────────────────────────────────

def plot_scatter_3d(df: pd.DataFrame, cluster_col: str = "Cluster") -> go.Figure:
    """3D scatter: Recency × Frequency × Monetary colored by cluster."""
    df = df.copy()
    df["ClusterName"] = df[cluster_col].map(
        lambda c: CLUSTER_NAMES.get(c, f"Cluster {c}")
    )

    fig = px.scatter_3d(
        df, x="Recency", y="Frequency", z="Monetary",
        color="ClusterName",
        color_discrete_map={v: CLUSTER_COLORS.get(k, "#999") for k, v in CLUSTER_NAMES.items()},
        hover_data=["CustomerID", "AvgSpending", "UniqueProducts"],
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=3))
    
    layout_dict = BASE_LAYOUT.copy()
    layout_dict.update(
        title="3D Customer Segmentation (RFM)",
        height=520,
        legend=dict(orientation="v", x=0.85),
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#333333", showbackground=False),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#333333", showbackground=False),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#333333", showbackground=False),
        ),
    )
    fig.update_layout(layout_dict)
    return fig


def plot_scatter_2d(df: pd.DataFrame, x: str, y: str, cluster_col: str = "Cluster") -> go.Figure:
    """2D scatter for any two features, colored by cluster."""
    df = df.copy()
    df["ClusterName"] = df[cluster_col].map(lambda c: CLUSTER_NAMES.get(c, f"Cluster {c}"))
    fig = px.scatter(
        df, x=x, y=y, color="ClusterName",
        color_discrete_map={v: CLUSTER_COLORS.get(k, "#999") for k, v in CLUSTER_NAMES.items()},
        hover_data=["CustomerID"],
        opacity=0.7,
    )
    layout_dict = BASE_LAYOUT.copy()
    layout_dict.update(title=f"{x} vs {y} by Cluster", height=400)
    fig.update_layout(layout_dict)
    return fig


def plot_cluster_radar(cluster_profiles: pd.DataFrame) -> go.Figure:
    """Radar / spider chart showing average feature values per cluster."""
    features = [
        "Recency", "Frequency", "TotalProducts", "Monetary", "AvgSpending", 
        "UniqueProducts", "AvgDaysToPurchase", "ExpectedPurchaseDays", "FromUK", 
        "CancelFrequency", "AvgMonthlySpending"
    ]
    fig = go.Figure()

    for _, row in cluster_profiles.iterrows():
        cluster_id = int(row["Cluster"])
        values = [row[f] for f in features]
        values += [values[0]]  # close the polygon

        color = CLUSTER_COLORS.get(cluster_id, "#999999")
        
        # Convert hex to rgba for transparent fills
        fill_color = "rgba(153, 153, 153, 0.15)"
        if color.startswith("#"):
            h = color.lstrip("#")
            if len(h) == 6:
                r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                fill_color = f"rgba({r}, {g}, {b}, 0.12)"

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=features + [features[0]],
            fill="toself",
            name=CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}"),
            line=dict(color=color, width=2.5),
            fillcolor=fill_color,
        ))

    layout_dict = BASE_LAYOUT.copy()
    layout_dict.update(
        title="Cluster Profiles (Normalized Avg Features)",
        polar=dict(
            radialaxis=dict(visible=True, color="#808080", gridcolor="#1c1c1c"),
            angularaxis=dict(gridcolor="#1c1c1c", color="#FFFFFF"),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=450,
    )
    fig.update_layout(layout_dict)
    return fig


def plot_cluster_bar(df: pd.DataFrame, cluster_col: str = "Cluster") -> go.Figure:
    """Bar chart showing cluster sizes."""
    counts = df[cluster_col].value_counts().reset_index()
    counts.columns = ["Cluster", "Count"]
    counts["Label"] = counts["Cluster"].map(lambda c: CLUSTER_NAMES.get(c, f"Cluster {c}"))
    counts["Color"] = counts["Cluster"].map(lambda c: CLUSTER_COLORS.get(c, "#999999"))

    fig = go.Figure(go.Bar(
        x=counts["Label"], y=counts["Count"],
        marker_color=counts["Color"].tolist(),
        text=counts["Count"], textposition="outside",
    ))
    layout_dict = BASE_LAYOUT.copy()
    layout_dict.update(title="Cluster Size Distribution", height=350,
                      xaxis_title="Segment", yaxis_title="# Customers")
    fig.update_layout(layout_dict)
    return fig


# ── Algorithm Comparison ──────────────────────────────────────────────────────

def plot_algorithm_comparison(metrics: dict) -> go.Figure:
    """
    Grouped bar chart comparing algorithms across all metrics.
    `metrics` format: {"Algorithm1": {"Metric1": Val, ...}, "Algorithm2": {...}, ...}
    Uses a single shared chart — metrics on x-axis, algorithms as grouped bars.
    Values are normalized per-metric to [0,1] so all 4 metrics share one y-axis.
    Hover shows the real original values.
    """
    algo_names = list(metrics.keys())
    metric_names = list(next(iter(metrics.values())).keys())

    # Normalize per metric so they're comparable on a shared y-axis
    raw = {m: [metrics[a].get(m, 0) for a in algo_names] for m in metric_names}
    norm = {}
    for m, vals in raw.items():
        mn, mx = min(vals), max(vals)
        span = mx - mn if mx != mn else 1.0
        norm[m] = [(v - mn) / span for v in vals]

    # Colors: one per algorithm
    algo_colors = ["#4d4d4d", "#808080", "#7089ba"]

    fig = go.Figure()

    for i, algo in enumerate(algo_names):
        norm_vals = [norm[m][i] for m in metric_names]
        raw_vals  = [raw[m][i]  for m in metric_names]
        hover_texts = [
            f"<b>{algo}</b><br>{m}: {rv:.4g}"
            for m, rv in zip(metric_names, raw_vals)
        ]
        fig.add_trace(go.Bar(
            name=algo,
            x=metric_names,
            y=norm_vals,
            marker=dict(
                color=algo_colors[i % len(algo_colors)],
                line=dict(width=0),          # no border lines on bars
            ),
            text=[f"{rv:.4g}" for rv in raw_vals],
            textposition="outside",
            textfont=dict(size=9, family="JetBrains Mono, monospace", color="#cccccc"),
            hovertext=hover_texts,
            hoverinfo="text",
        ))

    layout_dict = {k: v for k, v in BASE_LAYOUT.items() if k not in ["xaxis", "yaxis"]}
    layout_dict.update(
        title=None,
        barmode="group",
        bargap=0.22,
        bargroupgap=0.06,
        height=370,
        margin=dict(t=50, b=60, l=30, r=10),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            color="#808080",
            tickfont=dict(family="JetBrains Mono, monospace", size=10),
            tickangle=0,
        ),
        yaxis=dict(
            showgrid=False,          # no horizontal grid lines
            zeroline=False,
            showticklabels=False,    # hide normalized y ticks — real values shown in labels
            color="#808080",
            range=[-0.05, 1.35],     # leave headroom for outside labels
        ),
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=1.12, yanchor="top",
            font=dict(size=10, family=FONT_FAMILY),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
    )
    fig.update_layout(layout_dict)
    return fig


def plot_convergence_curve(df_conv: pd.DataFrame) -> go.Figure:
    """Convergence line chart for metaheuristic algorithms overlayed."""
    fig = go.Figure()
    
    # Map algorithms to colors
    colors = {
        "K-Means + DE": "#4d4d4d",
        "K-Means + PSO": "#808080",
        "K-Means + EOA": "#ffffff",
        "K-Means QLDE": "#7089ba",
    }
    
    for col in df_conv.columns:
        if col == "Iteration":
            continue
        fig.add_trace(go.Scatter(
            x=df_conv["Iteration"],
            y=df_conv[col],
            mode="lines+markers",
            name=col,
            line=dict(color=colors.get(col, "#999"), width=2.5),
            marker=dict(size=4),
        ))
        
    layout_dict = BASE_LAYOUT.copy()
    layout_dict.update(
        title=None, # Disable internal title to prevent Streamlit heading collision
        xaxis_title="Iteration",
        yaxis_title="SSE",
        height=340,
        margin=dict(t=40, b=50, l=50, r=10)
    )
    fig.update_layout(layout_dict)
    return fig


def plot_donut_chart(df: pd.DataFrame, cluster_col: str = "Cluster") -> go.Figure:
    """Donut chart showing cluster proportions."""
    counts = df[cluster_col].value_counts().reset_index()
    counts.columns = ["Cluster", "Count"]
    counts["Label"] = counts["Cluster"].map(lambda c: CLUSTER_NAMES.get(c, f"Cluster {c}"))
    colors_list = [CLUSTER_COLORS.get(c, "#999999") for c in counts["Cluster"]]

    fig = go.Figure(go.Pie(
        labels=counts["Label"],
        values=counts["Count"],
        hole=0.4,
        marker=dict(colors=colors_list, line=dict(color="#000000", width=2)),
        textinfo="percent",
        textposition="inside",
        hoverinfo="label+value+percent",
    ))
    layout_dict = BASE_LAYOUT.copy()
    layout_dict.update(
        title="Cluster Distribution Proportions",
        height=380,
        legend=dict(orientation="v", y=0.5, x=1.05, yanchor="middle", xanchor="left"),
        margin=dict(l=20, r=100, t=50, b=20)
    )
    fig.update_layout(layout_dict)
    return fig


def plot_confusion_matrix(cm: list[list], labels: list[str]) -> go.Figure:
    """Confusion matrix heatmap for classification results in dark theme."""
    # Custom colorscale: Carbon -> Periwinkle
    cmap = [[0.0, "#1c1c1c"], [1.0, "#7089ba"]]
    
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=cmap,
        text=cm, texttemplate="%{text}",
        showscale=False,
    ))
    layout_dict = BASE_LAYOUT.copy()
    layout_dict.update(
        title="Confusion Matrix Heatmap",
        xaxis_title="Predicted Segment", yaxis_title="Actual Segment",
        height=400,
    )
    fig.update_layout(layout_dict)
    fig.layout.yaxis.showgrid = False
    return fig


def plot_feature_importance(importances: dict[str, float], title: str = "Feature Importance") -> go.Figure:
    """Horizontal bar chart for feature importance in dark theme."""
    items = sorted(importances.items(), key=lambda x: x[1])
    features, values = zip(*items)
    # Highlight with transparency gradients using Periwinkle (#7089ba)
    colors = [f"rgba(112, 137, 186, {0.4 + 0.6 * v / max(values)})" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=features,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
    ))
    layout_dict = BASE_LAYOUT.copy()
    layout_dict.update(title=title, height=max(300, len(features) * 45), xaxis_title="Importance Score")
    fig.update_layout(layout_dict)
    return fig


def plot_classification_comparison_chart(df_results: pd.DataFrame) -> go.Figure:
    """Replicates 04.5 classification accuracy battle using Plotly Bar."""
    fig = px.bar(
        df_results,
        x="Dataset",
        y="Test Set Accuracy (%)",
        color="Method",
        barmode="group",
        color_discrete_map={
            "Decision Tree (DT)": "#808080",
            "Kernel SVM (KSVM)": "#7089ba"
        },
        text="Test Set Accuracy (%)"
    )
    fig.update_traces(textposition="outside", texttemplate="%{text:.1f}%")
    layout_dict = BASE_LAYOUT.copy()
    layout_dict.update(
        title="Accuracy Battle: Decision Tree vs SVM across Clustering Methods",
        yaxis=dict(range=[0, 110], showgrid=True, gridcolor="#1c1c1c", zeroline=False, color="#808080"),
        height=400
    )
    fig.update_layout(layout_dict)
    return fig

