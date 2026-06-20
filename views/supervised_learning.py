"""
views/supervised_learning.py
Supervised Learning Page — Compares Decision Tree and SVM classifiers across multiple clustering datasets and provides an interactive prediction form.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from utils.data_loader import load_clean_data, COLUMN_MAPPING, FEATURE_COLUMNS, FEATURE_LABELS
from utils.mock_models import (
    run_decision_tree,
    run_svm,
    run_cross_dataset_comparison,
    _prep_classification_data,
    FILE_DATASETS,
)
from utils.visualizer import (
    plot_classification_comparison_chart,
    plot_feature_importance,
    plot_confusion_matrix,
    CLUSTER_COLORS,
    CLUSTER_NAMES,
)

def load_selected_labeled_data(dataset_name: str) -> pd.DataFrame:
    filepath = FILE_DATASETS.get(dataset_name)
    df = pd.read_csv(filepath)
    df["CustomerID"] = df["CustomerID"].astype(int)
    df = df.rename(columns=COLUMN_MAPPING)
    return df

def show_supervised_learning():
    # ── Load Data & Generate Target Labels ────────────────────────────────────────
    df_raw = load_clean_data()
    
    # ── Header ────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hdr">
        <div>
            <h1>Supervised Learning & Real-time Prediction</h1>
            <p>Train classification models using various customer segments as targets to predict customer profiles in real-time.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar Controls ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Classifier Settings")
        
        # Combined options matching datasets and models
        combination_options = [
            "QLDE & Decision Tree",
            "QLDE & SVM",
            "STANDARD & Decision Tree",
            "STANDARD & SVM",
            "DE & Decision Tree",
            "DE & SVM",
            "PSO & Decision Tree",
            "PSO & SVM",
            "EOA & Decision Tree",
            "EOA & SVM"
        ]
        
        selected_combo = st.selectbox(
            "Select Dataset & Model:",
            combination_options,
            index=0,
            help="Choose a combination of target clustering dataset and classification model."
        )
        
        # Parse selected dataset and algorithm
        if "QLDE" in selected_combo:
            selected_dataset = "QLDE (Paper)"
        elif "STANDARD" in selected_combo:
            selected_dataset = "STANDARD (Baseline)"
        elif "DE" in selected_combo:
            selected_dataset = "DE"
        elif "PSO" in selected_combo:
            selected_dataset = "PSO"
        else:
            selected_dataset = "EOA"
            
        if "Decision Tree" in selected_combo:
            selected_algo = "Decision Tree"
        else:
            selected_algo = "SVM"
            
        st.markdown("---")
        st.caption(f"Target: {selected_dataset} | Model: {selected_algo}")

    # Load labeled dataset based on selection
    df_clustered = load_selected_labeled_data(selected_dataset)

    # ── Train Classifiers and Collect Metrics ─────────────────────────────────────
    with st.spinner(f"Loading/Training models on {selected_dataset}..."):
        dt_metrics, dt_importance, dt_model = run_decision_tree(df_clustered)
        svm_metrics, svm_importance, svm_model, svm_scaler = run_svm(df_clustered)

    # Prepare classifier results
    classifier_results = {
        "Decision Tree": {"metrics": dt_metrics, "importance": dt_importance, "model": dt_model},
        "SVM": {"metrics": svm_metrics, "importance": svm_importance, "model": svm_model},
    }

    # ── 4 Metric Cards for selected classifier ────────────────────────────────────
    st.markdown(f'<div class="sl">{selected_algo} Metrics on {selected_dataset} Labels</div>', unsafe_allow_html=True)

    metrics = classifier_results[selected_algo]["metrics"]
    card_class = "mc-highlight"

    metric_list = [
        ("Accuracy", f"{metrics['Accuracy']:.4f}", "Overall prediction accuracy on test set"),
        ("F1-Score (macro)", f"{metrics['F1-Score']:.4f}", "Harmonic mean of precision and recall"),
        ("Precision (macro)", f"{metrics['Precision']:.4f}", "Proportion of positive identifications that were correct"),
        ("Recall (macro)", f"{metrics['Recall']:.4f}", "Proportion of actual positives that were correct"),
    ]

    cards_html = '<div class="metric-grid cols-4">'
    for label, value, desc in metric_list:
        cards_html += f'<div class="{card_class}">' \
                      f'<div class="mc-val">{value}</div>' \
                      f'<div class="mc-lbl">{label}</div>' \
                      f'<div style="font-size: 12px; color: var(--color-ash); margin-top: 6px; line-height: 1.2;">{desc}</div>' \
                      f'</div>'
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # ── Confusion Matrix & Feature Importance ─────────────────────────────────────
    st.markdown("---")
    col_det1, col_det2 = st.columns(2)

    # Compute actual confusion matrix for detail view
    X_train, X_test, y_train, y_test = _prep_classification_data(df_clustered)
    
    if selected_algo == "Decision Tree":
        y_pred = dt_model.predict(X_test)
    else:
        X_test_scaled = svm_scaler.transform(X_test)
        y_pred = svm_model.predict(X_test_scaled)
        
    cm = confusion_matrix(y_test, y_pred).tolist()
    labels = [CLUSTER_NAMES[c] for c in sorted(np.unique(y_train))]

    with col_det1:
        st.markdown('<div class="sl">Confusion Matrix Heatmap</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_confusion_matrix(cm, labels), use_container_width=True)

    with col_det2:
        st.markdown('<div class="sl">Feature Importance Scores</div>', unsafe_allow_html=True)
        importance = classifier_results[selected_algo]["importance"]
        st.plotly_chart(
            plot_feature_importance(importance, f"{selected_algo} — Feature Importance"),
            use_container_width=True
        )

    # ── Decision Tree Logic (Business Rules) ──────────────────────────────────────
    if selected_algo == "Decision Tree":
        st.markdown("---")
        st.markdown('<div class="sl">Decision Tree Logical Business Rules (Aturan Bisnis)</div>', unsafe_allow_html=True)
        rules_text = export_text(dt_model, feature_names=FEATURE_COLUMNS)
        st.markdown("The following nested conditional logic shows the business rules extracted from the Decision Tree model:")
        st.code(rules_text, language="text")

    # ── Bar Chart Comparison (All Classifiers & Datasets) ──────────────────────────
    st.markdown("---")
    st.markdown('<div class="sl">Cross-Dataset Performance Comparison (Accuracy Battle)</div>', unsafe_allow_html=True)

    df_results = run_cross_dataset_comparison()
    st.plotly_chart(plot_classification_comparison_chart(df_results), use_container_width=True)
    
    st.markdown("**Detailed Metrics Table:**")
    st.dataframe(df_results, use_container_width=True)

    # ── Interactive Prediction Form ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sl">Real-time Segment Predictor</div>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("**Enter Customer Transactional Metrics:**")
        with st.form("predict_form"):
            recency = st.slider("Recency (days since last purchase)", min_value=1, max_value=365, value=93)
            frequency = st.slider("Frequency (number of transactions)", min_value=1, max_value=250, value=4)
            total_products = st.number_input("Total Products Purchased", min_value=1, max_value=100000, value=1500, step=50)
            monetary = st.number_input("Monetary (total spending £)", min_value=0.0, max_value=300000.0, value=1500.0, step=50.0)
            avg_spending = st.number_input("Average Spending per Item (£)", min_value=0.0, max_value=10000.0, value=35.0, step=1.0)
            unique_products = st.slider("Unique Products Purchased", min_value=1, max_value=1000, value=100)
            avg_days = st.slider("Average Days Between Purchases", min_value=0, max_value=365, value=30)
            expected_days = st.slider("Expected Days to Next Purchase", min_value=0, max_value=730, value=120)
            from_uk = st.selectbox("Is Customer from United Kingdom?", ["Yes", "No"])
            from_uk_val = 1 if from_uk == "Yes" else 0
            cancel_freq = st.slider("Cancellation Frequency", min_value=0, max_value=150, value=0)
            avg_monthly = st.number_input("Avg Monthly Spending (£)", min_value=0.0, max_value=50000.0, value=300.0, step=10.0)

            submitted = st.form_submit_button("Predict Customer Segment", use_container_width=True)

    with col_result:
        if submitted:
            input_data = pd.DataFrame([{
                "Recency": recency,
                "Frequency": frequency,
                "TotalProducts": total_products,
                "Monetary": monetary,
                "AvgSpending": avg_spending,
                "UniqueProducts": unique_products,
                "AvgDaysToPurchase": avg_days,
                "ExpectedPurchaseDays": expected_days,
                "FromUK": from_uk_val,
                "CancelFrequency": cancel_freq,
                "AvgMonthlySpending": avg_monthly,
            }])
            
            # Predict and calculate probabilities
            input_arr = input_data[FEATURE_COLUMNS].values
            
            if selected_algo == "Decision Tree":
                pred_cluster = int(dt_model.predict(input_arr)[0])
                pred_proba = dt_model.predict_proba(input_arr)[0]
                classes = dt_model.classes_
            else:
                input_scaled = svm_scaler.transform(input_arr)
                pred_cluster = int(svm_model.predict(input_scaled)[0])
                classes = svm_model.classes_
                # Check if model has probability capability
                if getattr(svm_model, "probability", False):
                    pred_proba = svm_model.predict_proba(input_scaled)[0]
                else:
                    # Fallback for SVC without probability calibration
                    pred_proba = np.zeros(len(classes))
                    try:
                        pred_idx = np.where(classes == pred_cluster)[0][0]
                        pred_proba[pred_idx] = 1.0
                    except IndexError:
                        if len(classes) > 0:
                            pred_proba[0] = 1.0
            
            segment_name = CLUSTER_NAMES.get(pred_cluster, f"Cluster {pred_cluster}")
            color = CLUSTER_COLORS.get(pred_cluster, "#A39CF7")
            confidence = pred_proba.max() * 100

            st.markdown(f"""
            <div class="mc-highlight" style="padding: 24px; border-color: {color};">
                <div style="font-size:12px; color:var(--color-steel); margin-bottom:4px; font-family:var(--font-geist-mono); text-transform:uppercase; letter-spacing:0.02em;">Predicted Segment</div>
                <div class="mc-val" style="color:{color}; font-size: 32px;">{segment_name}</div>
                <div style="font-size:14px; color:var(--color-paper); margin-top:8px;">
                    Prediction Confidence: <b style="color:{color}">{confidence:.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br><b>Probability Breakdown per Segment:</b>", unsafe_allow_html=True)
            for i, cls in enumerate(classes):
                name = CLUSTER_NAMES.get(int(cls), f"Cluster {cls}")
                prob = pred_proba[i] * 100
                color_cls = CLUSTER_COLORS.get(int(cls), "#888888")
                st.markdown(f"""
                <div style="margin:8px 0;">
                    <div style="display:flex; justify-content:space-between; font-size:14px; margin-bottom:3px;">
                        <span style="color:var(--color-steel)">{name}</span>
                        <span style="color:{color_cls}; font-weight:600">{prob:.1f}%</span>
                    </div>
                    <div style="background:var(--color-carbon); border:1px dashed var(--color-graphite); border-radius:4px; height:6px;">
                        <div style="width:{prob}%; background:{color_cls}; border-radius:4px; height:6px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="sl">Strategic Marketing Recommendations</div>', unsafe_allow_html=True)
            recommendations = {
                0: "<b>High-Value Segment:</b> Deliver red-carpet treatment. Offer premium loyalty program tiers, exclusive VIP events, dedicated support, and bespoke gifts.",
                1: "<b>Price-Sensitive Segment:</b> Offer seasonal clearance promotions, high-discount bundles, coupon codes, and cost-efficient alternatives to drive volume.",
                2: "<b>High-Expectation-UK Segment:</b> Provide high-standard shipping, local UK customer support, and customized product recommendations aligned with domestic trends.",
                3: "<b>Uncertain-Buyer Segment:</b> Re-engage with surveys to understand friction. Send product guides, testimonials, and small trigger incentives to convert them to active users.",
                4: "<b>Cautious-Consumer Segment:</b> Offer money-back guarantees, detailed product specifications, social proof, and clear FAQs to ease security and quality concerns.",
                5: "<b>Balanced Segment:</b> Keep engaged with standard newsletter updates, cross-selling related categories, and milestone-based loyalty rewards.",
            }
            rec = recommendations.get(pred_cluster, "Design personalized engagement campaigns targeting specific customer behaviors.")
            st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid {color};">
                {rec}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding: 60px; color:var(--color-ash); font-size: 14px;">
                Fill in the customer metrics on the left form and click <b>Predict Customer Segment</b> to calculate results.
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:var(--color-ash); font-size:12px; font-family:var(--font-geist-mono); letter-spacing:0.02em;'>"
        "KELOMPOK 6 · ML FINAL PROJECT · SUPERVISED CLASSIFICATION & PREDICTION"
        "</div>",
        unsafe_allow_html=True,
    )
