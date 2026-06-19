"""
utils/mock_models.py
Implementations for Clustering (Standard & Metaheuristic) & Classification models.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
import streamlit as st
from pathlib import Path
import joblib

MODELS_DIR = Path(__file__).parent.parent / "models"
FEATURE_COLUMNS = [
    "Recency", "Frequency", "Monetary",
    "AvgSpending", "UniqueProducts", "CancelFrequency", "AvgMonthlySpending",
]

def _scale_features(df: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(df[FEATURE_COLUMNS])

# ── Unsupervised Algorithms (Clustering & Metaheuristics) ──────────────────────

# Metaheuristic mock metrics database (for k=4) - kept for backward compatibility
ALGO_METRICS_K4 = {
    "K-Means Standard": {
        "SSE": 8200.5,
        "Silhouette Score": 0.3852,
        "Davies-Bouldin Index": 1.254,
        "Calinski-Harabasz Index": 1850.4,
    },
    "K-Means + DE": {
        "SSE": 7950.2,
        "Silhouette Score": 0.4125,
        "Davies-Bouldin Index": 1.152,
        "Calinski-Harabasz Index": 1980.2,
    },
    "K-Means + PSO": {
        "SSE": 7850.1,
        "Silhouette Score": 0.4281,
        "Davies-Bouldin Index": 1.084,
        "Calinski-Harabasz Index": 2010.5,
    },
    "K-Means + EOA": {
        "SSE": 7720.6,
        "Silhouette Score": 0.4357,
        "Davies-Bouldin Index": 1.021,
        "Calinski-Harabasz Index": 2085.1,
    },
    "K-Means QLDE": {
        "SSE": 7510.4,
        "Silhouette Score": 0.4589,
        "Davies-Bouldin Index": 0.941,
        "Calinski-Harabasz Index": 2190.8,
    }
}

@st.cache_data(show_spinner="Running Clustering Algorithm...")
def run_unsupervised_algorithm(df: pd.DataFrame, algorithm: str, k: int = 4) -> tuple[pd.DataFrame, dict]:
    """
    Runs actual clustering optimization on scaled customer features.
    """
    from utils.algorithms import QLDE, KMeansDE, KMeansPSO, KMeansEOA
    
    X = _scale_features(df)
    
    if algorithm == "K-Means Standard":
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        labels = model.labels_
        sse = model.inertia_
    elif algorithm == "K-Means + DE":
        model = KMeansDE(n_clusters=k, pop_size=30, max_iter=100, random_state=42)
        model.fit(X)
        labels = model.labels_
        sse = model.inertia_
    elif algorithm == "K-Means + PSO":
        model = KMeansPSO(n_clusters=k, pop_size=30, max_iter=100, random_state=42)
        model.fit(X)
        labels = model.labels_
        sse = model.inertia_
    elif algorithm == "K-Means + EOA":
        model = KMeansEOA(n_clusters=k, pop_size=30, max_iter=100, random_state=42)
        model.fit(X)
        labels = model.labels_
        sse = model.inertia_
    elif algorithm == "K-Means QLDE":
        model = QLDE(n_clusters=k, pop_size=30, max_iter=100, random_state=42)
        model.fit(X)
        labels = model.labels_
        sse = model.inertia_
    else:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        labels = model.labels_
        sse = model.inertia_

    result = df.copy()
    result["Cluster"] = labels
    
    if len(np.unique(labels)) > 1:
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
    else:
        sil = 0.0
        db = 9.99
        ch = 0.0
        
    metrics = {
        "SSE": round(sse, 1),
        "Silhouette Score": round(sil, 4),
        "Davies-Bouldin Index": round(db, 3),
        "Calinski-Harabasz Index": round(ch, 1),
        "n_clusters": k
    }
    
    return result, metrics


@st.cache_data(show_spinner="Generating Convergence Curves...")
def get_convergence_curves(max_iter: int = 50) -> pd.DataFrame:
    """Generates actual SSE convergence data over iterations for the metaheuristics on default dataset."""
    from utils.data_loader import load_clean_data
    from utils.algorithms import QLDE, KMeansDE, KMeansPSO, KMeansEOA
    
    df = load_clean_data()
    X = _scale_features(df)
    k = 4
    
    de = KMeansDE(n_clusters=k, pop_size=15, max_iter=max_iter, random_state=42)
    de.fit(X)
    
    pso = KMeansPSO(n_clusters=k, pop_size=15, max_iter=max_iter, random_state=42)
    pso.fit(X)
    
    eoa = KMeansEOA(n_clusters=k, pop_size=15, max_iter=max_iter, random_state=42)
    eoa.fit(X)
    
    qlde = QLDE(n_clusters=k, pop_size=15, max_iter=max_iter, random_state=42)
    qlde.fit(X)
    
    l = min(len(de.convergence_curve_), len(pso.convergence_curve_), len(eoa.convergence_curve_), len(qlde.convergence_curve_))
    
    return pd.DataFrame({
        "Iteration": np.arange(1, l + 1),
        "K-Means + DE": de.convergence_curve_[:l],
        "K-Means + PSO": pso.convergence_curve_[:l],
        "K-Means + EOA": eoa.convergence_curve_[:l],
        "K-Means QLDE": qlde.convergence_curve_[:l],
    })


@st.cache_data(show_spinner="Evaluating all algorithms for comparison...")
def get_algorithm_comparison_metrics(df: pd.DataFrame, k: int = 4) -> dict:
    """Evaluate all 5 algorithms dynamically to produce the comparison chart."""
    comparison = {}
    for algo in ["K-Means Standard", "K-Means + DE", "K-Means + PSO", "K-Means + EOA", "K-Means QLDE"]:
        _, metrics = run_unsupervised_algorithm(df, algo, k=k)
        comparison[algo] = {
            "SSE": metrics["SSE"],
            "Silhouette Score": metrics["Silhouette Score"],
            "Davies-Bouldin Index": metrics["Davies-Bouldin Index"],
            "Calinski-Harabasz Index": metrics["Calinski-Harabasz Index"],
        }
    return comparison


def run_kmeans(df: pd.DataFrame, k: int = 4) -> tuple[pd.DataFrame, dict]:
    """Backward compatibility wrapper for standard K-Means."""
    return run_unsupervised_algorithm(df, "K-Means Standard", k)


def get_cluster_profiles(df_clustered: pd.DataFrame, cluster_col: str = "Cluster") -> pd.DataFrame:
    """Compute per-cluster average feature values for radar and profile charts."""
    return (
        df_clustered.groupby(cluster_col)[FEATURE_COLUMNS]
        .mean()
        .reset_index()
        .rename(columns={cluster_col: "Cluster"})
    )


# ── Supervised Algorithms (Classification) ───────────────────────────────────

def _prep_classification_data(df_clustered: pd.DataFrame, cluster_col: str = "Cluster"):
    valid = df_clustered[df_clustered[cluster_col] != -1].copy()
    X = valid[FEATURE_COLUMNS].values
    y = valid[cluster_col].values
    
    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min() if len(counts) > 0 else 0
    
    if min_count >= 2:
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        return train_test_split(X, y, test_size=0.2, random_state=42)


def _compute_metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "F1-Score": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }


@st.cache_data(show_spinner="Training Logistic Regression...")
def run_logistic_regression(df_clustered: pd.DataFrame) -> tuple[dict, dict]:
    """Logistic Regression classifier (baseline)."""
    X_train, X_test, y_train, y_test = _prep_classification_data(df_clustered)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = _compute_metrics(y_test, y_pred)
    importances = np.mean(np.abs(model.coef_), axis=0)
    if np.sum(importances) > 0:
        importances /= np.sum(importances)
    importance_dict = dict(zip(FEATURE_COLUMNS, importances.round(4).tolist()))
    return metrics, importance_dict


@st.cache_data(show_spinner="Training Decision Tree...")
def run_decision_tree(df_clustered: pd.DataFrame) -> tuple[dict, dict]:
    """Decision Tree classifier."""
    X_train, X_test, y_train, y_test = _prep_classification_data(df_clustered)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = DecisionTreeClassifier(max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = _compute_metrics(y_test, y_pred)
    importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_.round(4).tolist()))
    return metrics, importance


@st.cache_data(show_spinner="Training Random Forest...")
def run_random_forest(df_clustered: pd.DataFrame) -> tuple[dict, dict]:
    """Random Forest classifier."""
    X_train, X_test, y_train, y_test = _prep_classification_data(df_clustered)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = _compute_metrics(y_test, y_pred)
    importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_.round(4).tolist()))
    return metrics, importance


@st.cache_data(show_spinner="Training SVM...")
def run_svm(df_clustered: pd.DataFrame) -> tuple[dict, dict]:
    """Support Vector Machine (SVM) classifier."""
    X_train, X_test, y_train, y_test = _prep_classification_data(df_clustered)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Limit training data to 2000 samples to keep training fast, matching the notebook
    X_train_svm = X_train[:2000]
    y_train_svm = y_train[:2000]

    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train_svm, y_train_svm)
    y_pred = model.predict(X_test)

    metrics = _compute_metrics(y_test, y_pred)
    
    # Permutation importance for SVM feature importance
    perm_importance = permutation_importance(model, X_test, y_test, random_state=42, n_repeats=5)
    importances = np.abs(perm_importance.importances_mean)
    if np.sum(importances) > 0:
        importances /= np.sum(importances)
    importance_dict = dict(zip(FEATURE_COLUMNS, importances.round(4).tolist()))
    return metrics, importance_dict


def load_model_if_exists(filename: str):
    """Attempt to load a .pkl model. Returns None if not found."""
    path = MODELS_DIR / filename
    if path.exists():
        return joblib.load(path)
    return None
