"""
app.py — Landing Page / Dashboard Home
Customer Segmentation Web App — Kelompok 6
Arsitektur Single-Page Application (SPA)
"""

import streamlit as st
from utils.data_loader import load_clean_data, get_summary_stats

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SegmentIQ — Customer Segmentation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
with open("style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Option Menu & Modular Page Imports ─────────────────────────────────────────
from streamlit_option_menu import option_menu
from views.eda import show_eda
from views.preprocessing import show_preprocessing
from views.unsupervised_learning import show_unsupervised_learning
from views.supervised_learning import show_supervised_learning
from views.about import show_about

# ── Sidebar Navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    # Premium branding block matching the Index wireframe theme
    st.markdown("""
    <div style="padding: 24px 16px; border: 1px dashed var(--color-graphite); border-radius: 20px; margin-top: 16px; margin-bottom: 24px; text-align: center;">
        <div style="font-family: var(--font-raveo-variable); font-size: 24px; font-weight: 900; color: var(--color-paper); letter-spacing: -0.24px;">[ SegmentIQ ]</div>
        <div style="font-family: var(--font-geist-mono); font-size: 9px; color: var(--color-steel); text-transform: uppercase; letter-spacing: 0.02em; margin-top: 8px; font-weight: 500;">Customer Segmentation</div>
    </div>
    """, unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["Home", "EDA", "Preprocessing", "Unsupervised", "Supervised", "About"],
        icons=["house", "bar-chart", "sliders", "diagram-3", "cpu", "info-circle"],
        default_index=0,
        styles={
            "container": {"padding": "0px", "background-color": "transparent"},
            "icon": {"color": "var(--color-steel)", "font-size": "0.95rem"}, 
            "nav-link": {
                "font-size": "0.88rem", 
                "text-align": "left", 
                "margin": "6px 0px", 
                "color": "var(--text-muted)",
                "font-family": "Inter, sans-serif",
                "padding": "10px 16px",
                "border-radius": "8px",
                "border": "1px dashed transparent",
                "transition": "all 0.15s ease"
            },
            "nav-link-selected": {
                "background-color": "rgba(112, 137, 186, 0.08)", 
                "color": "#ffffff", 
                "border": "1px dashed var(--color-periwinkle-annotation)",
                "font-weight": "600"
            },
        }
    )


def show_home():
    # ── Load Data ─────────────────────────────────────────────────────────────────
    df = load_clean_data()
    df_anomaly = pd.DataFrame()
    stats = get_summary_stats(df)

    # ── Hero Banner ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hdr">
        <div>
            <h1>SegmentIQ</h1>
            <p>Customer Segmentation in Digital Marketing — ML Final Project (Kelompok 6)</p>
        </div>
        <div style="margin-top: 16px; padding-top: 16px; border-top: 1px dashed var(--color-graphite); font-family: var(--font-geist-mono); font-size: 12px; font-weight: 500; color: var(--color-steel); letter-spacing: 0.02em;">
            HYBRID STRATEGY: DBSCAN NOISE ISOLATION ➜ METAHEURISTICS CENTROID OPTIMIZATION ➜ SUPERVISED CLASSIFICATION
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Metrics ───────────────────────────────────────────────────────────────
    st.markdown('<div class="sl">Dataset Overview</div>', unsafe_allow_html=True)

    metrics_display = [
        ("Total Inliers", f"{stats['total_customers']:,}", "Active clusterable customers", True),
        ("Avg Recency", f"{stats['avg_recency']:.1f} days", "Since last purchase", False),
        ("Avg Frequency", f"{stats['avg_frequency']:.1f}x", "Transactions per customer", False),
        ("Avg Monetary", f"£{stats['avg_monetary']:,.1f}", "Spend per customer", False),
        ("Cancel Rate", f"{stats['avg_cancel_rate']:.1f}%", "Customers with cancellations", False),
    ]

    cards_html = '<div class="metric-grid cols-5">'
    for label, value, desc, highlight in metrics_display:
        card_class = "mc-highlight" if highlight else "mc"
        cards_html += f'<div class="{card_class}">' \
                      f'<div class="mc-val">{value}</div>' \
                      f'<div class="mc-lbl">{label}</div>' \
                      f'<div style="font-size: 12px; color: var(--color-ash); margin-top: 6px; font-family: var(--font-raveo-variable); line-height: 1.2;">{desc}</div>' \
                      f'</div>'
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # ── Pipeline Overview ─────────────────────────────────────────────────────────
    st.markdown('<div class="sl">ML Pipeline Overview</div>', unsafe_allow_html=True)

    pipeline_steps = [
        ("01", "EDA", "Understand distributions, skewness, and inter-feature correlations.", "var(--color-steel)", "Completed"),
        ("02", "Preprocessing", "RFM++ engineering, DBSCAN noise isolation, and Standard Scaling.", "var(--color-ash)", "Completed"),
        ("03", "Unsupervised", "K-Means centroid optimization using Adaptive Q-Learning DE.", "var(--color-periwinkle-annotation)", "Completed"),
        ("04", "Supervised", "Classifier training (Random Forest) for real-time customer predictions.", "var(--color-paper)", "Completed"),
    ]

    pipeline_html = '<div class="metric-grid cols-4">'
    for num, title, desc, color, status in pipeline_steps:
        pipeline_html += f'<div class="glass-card" style="height:100%; border-top: 2px dashed {color} !important;">' \
                         f'<div style="font-size: 24px; font-family: var(--font-raveo-variable); margin-bottom: 8px; color: {color}; font-weight: 900; letter-spacing: -0.24px;">{num}</div>' \
                         f'<div style="font-weight: 500; color: var(--color-paper); margin-bottom: 6px; font-size: 16px; font-family: var(--font-raveo-variable);">{title}</div>' \
                         f'<div style="font-size: 14px; color: var(--color-steel); margin-bottom: 12px; line-height: 1.6; letter-spacing: -0.14px;">{desc}</div>' \
                         f'<div style="font-size: 9px; font-family: var(--font-geist-mono); color: var(--color-steel); font-weight: 500; text-transform: uppercase; letter-spacing: 0.02em;">[{status}]</div>' \
                         f'</div>'
    pipeline_html += '</div>'
    st.markdown(pipeline_html, unsafe_allow_html=True)

    # ── Dataset Preview ───────────────────────────────────────────────────────────
    st.markdown("---")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="sl">Cleaned Customer Features (Inliers)</div>', unsafe_allow_html=True)
        st.dataframe(
            df.head(10).style.format({
                "Monetary": "£{:,.2f}",
                "AvgSpending": "£{:.2f}",
                "AvgMonthlySpending": "£{:.2f}",
            }),
            use_container_width=True,
            height=320,
        )

    with col_right:
        st.markdown('<div class="sl">Anomaly Isolated Customers</div>', unsafe_allow_html=True)
        if not df_anomaly.empty:
            st.markdown(f"""
            <div class="success-banner" style="margin-bottom: 12px;">
                <b style="color:var(--color-periwinkle-annotation)">{len(df_anomaly)} anomalous customers</b> were isolated by <b>DBSCAN</b> before clustering. 
                These represent outlier VIP or wholesale accounts whose extreme monetary/frequency volumes would distort standard cluster centroids.
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(
                df_anomaly[["CustomerID", "Recency", "Frequency", "Monetary", "CancelFrequency"]].head(8),
                use_container_width=True,
                height=200,
            )
        else:
            st.info("Anomaly dataset not loaded.")

    # ── Footer ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:var(--color-ash); font-size:12px; font-family:var(--font-geist-mono); letter-spacing:0.02em;'>"
        "KELOMPOK 6 · ML FINAL PROJECT · CUSTOMER SEGMENTATION IN DIGITAL MARKETING · 2026"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Dynamic SPA Router ─────────────────────────────────────────────────────────
if selected == "Home":
    show_home()
elif selected == "EDA":
    show_eda()
elif selected == "Preprocessing":
    show_preprocessing()
elif selected == "Unsupervised":
    show_unsupervised_learning()
elif selected == "Supervised":
    show_supervised_learning()
elif selected == "About":
    show_about()
