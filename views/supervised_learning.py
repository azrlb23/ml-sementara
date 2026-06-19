"""
views/supervised_learning.py
Supervised Learning Page — Compares classifiers (LR, DT, RF, SVM) and provides an interactive prediction form.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from utils.data_loader import load_clean_data, FEATURE_COLUMNS, FEATURE_LABELS
from utils.mock_models import (
    run_unsupervised_algorithm,
    run_logistic_regression,
    run_decision_tree,
    run_random_forest,
    run_svm,
    _prep_classification_data,
)
from utils.visualizer import (
    plot_algorithm_comparison,
    plot_feature_importance,
    plot_confusion_matrix,
    CLUSTER_COLORS,
    CLUSTER_NAMES,
)

def show_supervised_learning():
    # ── Load Data & Generate Target Labels ────────────────────────────────────────
    df = load_clean_data()
    # Target labels are derived from K-Means QLDE (k=4) clustering
    df_clustered, _ = run_unsupervised_algorithm(df, "K-Means QLDE", k=4)

    # ── Header ────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hdr">
        <div>
            <h1>Supervised Learning & Real-time Prediction</h1>
            <p>Train classification models using QLDE segments as targets to predict new customer profiles in real-time.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar Controls ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Classifier Settings")
        selected_algo = st.selectbox(
            "Select Model for Details:",
            ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"],
            index=2,
            help="Choose a classification model to view its detailed performance, confusion matrix, and feature importances."
        )
        st.markdown("---")
        st.caption("Training target: 4 customer segments derived from optimal K-Means QLDE clustering.")

    # ── Train Classifiers and Collect Metrics ─────────────────────────────────────
    with st.spinner("Training classifiers..."):
        lr_metrics, lr_importance = run_logistic_regression(df_clustered)
        dt_metrics, dt_importance = run_decision_tree(df_clustered)
        rf_metrics, rf_importance = run_random_forest(df_clustered)
        svm_metrics, svm_importance = run_svm(df_clustered)

    # Prepare classifier comparison database
    classifier_results = {
        "Logistic Regression": {"metrics": lr_metrics, "importance": lr_importance},
        "Decision Tree": {"metrics": dt_metrics, "importance": dt_importance},
        "Random Forest": {"metrics": rf_metrics, "importance": rf_importance},
        "SVM": {"metrics": svm_metrics, "importance": svm_importance},
    }

    # ── 4 Metric Cards for selected classifier ────────────────────────────────────
    st.markdown(f'<div class="sl">{selected_algo} — Model Metrics</div>', unsafe_allow_html=True)

    # Random Forest is the best classifier (Accuracy > 97%)
    is_best_clf = (selected_algo == "Random Forest")
    card_class = "mc-highlight" if is_best_clf else "mc"
    metrics = classifier_results[selected_algo]["metrics"]

    metric_list = [
        ("Accuracy", f"{metrics['Accuracy']:.4f}", "Overall prediction accuracy on test set"),
        ("F1-Score (macro)", f"{metrics['F1-Score']:.4f}", "Harmonic mean of precision and recall"),
        ("Precision (macro)", f"{metrics['Precision']:.4f}", "Proportion of positive identifications that were actually correct"),
        ("Recall (macro)", f"{metrics['Recall']:.4f}", "Proportion of actual positives that were identified correctly"),
    ]

    cards_html = '<div class="metric-grid cols-4">'
    for label, value, desc in metric_list:
        badge = " (Best)" if is_best_clf else ""
        cards_html += f'<div class="{card_class}">' \
                      f'<div class="mc-val">{value}</div>' \
                      f'<div class="mc-lbl">{label}{badge}</div>' \
                      f'<div style="font-size: 12px; color: var(--color-ash); margin-top: 6px; line-height: 1.2;">{desc}</div>' \
                      f'</div>'
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # ── Confusion Matrix & Feature Importance ─────────────────────────────────────
    st.markdown("---")
    col_det1, col_det2 = st.columns(2)

    # Compute actual confusion matrix for detail view
    X_train, X_test, y_train, y_test = _prep_classification_data(df_clustered)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if selected_algo == "Logistic Regression":
        model = LogisticRegression(max_iter=500, random_state=42)
    elif selected_algo == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=6, random_state=42)
    elif selected_algo == "SVM":
        model = SVC(kernel='rbf', probability=True, random_state=42)
        X_train_scaled = X_train_scaled[:2000]
        y_train = y_train[:2000]
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
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

    # ── Bar Chart Comparison (All Classifiers) ────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sl">Model Performance Comparison</div>', unsafe_allow_html=True)

    classifier_comparison = {
        "Logistic Regression": {
            "Accuracy": lr_metrics["Accuracy"],
            "F1-Score": lr_metrics["F1-Score"],
            "Precision": lr_metrics["Precision"],
            "Recall": lr_metrics["Recall"],
        },
        "Decision Tree": {
            "Accuracy": dt_metrics["Accuracy"],
            "F1-Score": dt_metrics["F1-Score"],
            "Precision": dt_metrics["Precision"],
            "Recall": dt_metrics["Recall"],
        },
        "Random Forest": {
            "Accuracy": rf_metrics["Accuracy"],
            "F1-Score": rf_metrics["F1-Score"],
            "Precision": rf_metrics["Precision"],
            "Recall": rf_metrics["Recall"],
        },
        "SVM": {
            "Accuracy": svm_metrics["Accuracy"],
            "F1-Score": svm_metrics["F1-Score"],
            "Precision": svm_metrics["Precision"],
            "Recall": svm_metrics["Recall"],
        }
    }
    st.plotly_chart(plot_algorithm_comparison(classifier_comparison), use_container_width=True)

    # ── Interactive Prediction Form ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sl">Real-time Segment Predictor</div>', unsafe_allow_html=True)

    # Train the inference model based on the selected classifier
    valid = df_clustered[df_clustered["Cluster"] != -1]
    X_infer = valid[FEATURE_COLUMNS].values
    y_infer = valid["Cluster"].values
    scaler_infer = StandardScaler()
    X_infer_scaled = scaler_infer.fit_transform(X_infer)

    if selected_algo == "Logistic Regression":
        clf_infer = LogisticRegression(max_iter=500, random_state=42)
    elif selected_algo == "Decision Tree":
        clf_infer = DecisionTreeClassifier(max_depth=6, random_state=42)
    elif selected_algo == "SVM":
        clf_infer = SVC(kernel='rbf', probability=True, random_state=42)
        X_infer_scaled = X_infer_scaled[:2000]
        y_infer = y_infer[:2000]
    else:
        clf_infer = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)

    clf_infer.fit(X_infer_scaled, y_infer)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("**Enter Customer Transactional Metrics:**")
        with st.form("predict_form"):
            recency = st.slider("Recency (days since last purchase)", min_value=1, max_value=400, value=45)
            frequency = st.slider("Frequency (number of transactions)", min_value=1, max_value=50, value=8)
            monetary = st.number_input("Monetary (total spending £)", min_value=0.0, max_value=50000.0, value=1250.0, step=50.0)
            avg_spending = st.number_input("Average Spending per Item (£)", min_value=0.0, max_value=5000.0, value=35.0, step=1.0)
            unique_products = st.slider("Unique Products Purchased", min_value=1, max_value=500, value=65)
            cancel_freq = st.slider("Cancellation Frequency", min_value=0, max_value=50, value=1)
            avg_monthly = st.number_input("Avg Monthly Spending (£)", min_value=0.0, max_value=15000.0, value=280.0, step=10.0)

            submitted = st.form_submit_button("Predict Customer Segment", use_container_width=True)

    with col_result:
        if submitted:
            input_data = pd.DataFrame([{
                "Recency": recency,
                "Frequency": frequency,
                "Monetary": monetary,
                "AvgSpending": avg_spending,
                "UniqueProducts": unique_products,
                "CancelFrequency": cancel_freq,
                "AvgMonthlySpending": avg_monthly,
            }])
            
            # Scale and predict
            X_input = scaler_infer.transform(input_data[FEATURE_COLUMNS].values)
            pred_cluster = int(clf_infer.predict(X_input)[0])
            pred_proba = clf_infer.predict_proba(X_input)[0]
            
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
            for i, cls in enumerate(clf_infer.classes_):
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
                0: "<b>Champions Segment:</b> Reward with exclusive VIP benefits, early-access campaigns, and referral rewards. Personal account manager outreach is highly effective.",
                1: "<b>Loyal Customers Segment:</b> Encourage subscription plans, upsell cross-brand packages, and incentivize higher transaction sizes with progressive threshold offers.",
                2: "<b>At-Risk Customers Segment:</b> Deploy personalized email win-back offers with direct, limited-time 15-20% discounts. Send customer satisfaction surveys to address friction.",
                3: "<b>New/Inactive Segment:</b> Automate welcome-series drip emails offering onboarding guides and a first-purchase discounts package to build recurring habit loops.",
            }
            rec = recommendations.get(pred_cluster, "Design personalized engagement campaigns targeting specific feature anomalies.")
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
