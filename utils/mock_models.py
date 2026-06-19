"""
utils/mock_models.py
Implementations for Clustering (Standard & Metaheuristic) & Classification models.
Aligned with the nplrahman922 'kayis' branch.
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
LABELED_DIR = Path(__file__).parent.parent / "data" / "Labeled"
FEATURE_COLUMNS = [
    "Recency",
    "Frequency",
    "TotalProducts",
    "Monetary",
    "AvgSpending",
    "UniqueProducts",
    "AvgDaysToPurchase",
    "ExpectedPurchaseDays",
    "FromUK",
    "CancelFrequency",
    "AvgMonthlySpending",
]

def _scale_features(df: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(df[FEATURE_COLUMNS])

# ── Unsupervised Algorithms (Clustering & Metaheuristics) ──────────────────────

@st.cache_data(show_spinner="Running Clustering Algorithm...")
def run_unsupervised_algorithm(df: pd.DataFrame, algorithm: str, k: int = 6) -> tuple[pd.DataFrame, dict]:
    """
    Runs actual clustering optimization on PCA-reduced customer features (X_pca).
    """
    from utils.algorithms import QLDE, KMeansDE, KMeansPSO, KMeansEOA
    from utils.data_loader import load_pca_data
    
    # Load PCA features (6 principal components)
    df_pca = load_pca_data()
    
    # Align by CustomerID
    df_pca = df_pca.set_index("CustomerID")
    df_align = df.set_index("CustomerID")
    df_pca_aligned = df_pca.reindex(df_align.index)
    X_pca = df_pca_aligned.values
    
    if algorithm == "K-Means Standard":
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_pca)
        labels = model.labels_
        sse = model.inertia_
    elif algorithm == "K-Means + DE":
        model = KMeansDE(n_clusters=k, pop_size=30, max_iter=100, random_state=42)
        model.fit(X_pca)
        labels = model.labels_
        sse = model.inertia_
    elif algorithm == "K-Means + PSO":
        model = KMeansPSO(n_clusters=k, pop_size=30, max_iter=100, random_state=42)
        model.fit(X_pca)
        labels = model.labels_
        sse = model.inertia_
    elif algorithm == "K-Means + EOA":
        model = KMeansEOA(n_clusters=k, pop_size=30, max_iter=100, random_state=42)
        model.fit(X_pca)
        labels = model.labels_
        sse = model.inertia_
    elif algorithm == "K-Means QLDE":
        model = QLDE(n_clusters=k, pop_size=30, max_iter=100, random_state=42)
        model.fit(X_pca)
        labels = model.labels_
        sse = model.inertia_
    else:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_pca)
        labels = model.labels_
        sse = model.inertia_

    result = df.copy()
    result["Cluster"] = labels
    
    if len(np.unique(labels)) > 1:
        sil = silhouette_score(X_pca, labels)
        db = davies_bouldin_score(X_pca, labels)
        ch = calinski_harabasz_score(X_pca, labels)
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
    """Generates actual SSE convergence data over iterations for the metaheuristics on X_pca."""
    from utils.data_loader import load_pca_data
    from utils.algorithms import QLDE, KMeansDE, KMeansPSO, KMeansEOA
    
    df_pca = load_pca_data().drop(columns=["CustomerID"])
    X_pca = df_pca.values
    k = 6
    
    de = KMeansDE(n_clusters=k, pop_size=15, max_iter=max_iter, random_state=42)
    de.fit(X_pca)
    
    pso = KMeansPSO(n_clusters=k, pop_size=15, max_iter=max_iter, random_state=42)
    pso.fit(X_pca)
    
    eoa = KMeansEOA(n_clusters=k, pop_size=15, max_iter=max_iter, random_state=42)
    eoa.fit(X_pca)
    
    qlde = QLDE(n_clusters=k, pop_size=15, max_iter=max_iter, random_state=42)
    qlde.fit(X_pca)
    
    l = min(len(de.convergence_curve_), len(pso.convergence_curve_), len(eoa.convergence_curve_), len(qlde.convergence_curve_))
    
    return pd.DataFrame({
        "Iteration": np.arange(1, l + 1),
        "K-Means + DE": de.convergence_curve_[:l],
        "K-Means + PSO": pso.convergence_curve_[:l],
        "K-Means + EOA": eoa.convergence_curve_[:l],
        "K-Means QLDE": qlde.convergence_curve_[:l],
    })


@st.cache_data(show_spinner="Evaluating all algorithms for comparison...")
def get_algorithm_comparison_metrics(df: pd.DataFrame, k: int = 6) -> dict:
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


def run_kmeans(df: pd.DataFrame, k: int = 6) -> tuple[pd.DataFrame, dict]:
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

# Mapping of all 5 clustering datasets as defined in the repository
FILE_DATASETS = {
    'QLDE (Paper)': LABELED_DIR / 'hasildata_kmeans-qlde.csv',
    'STANDARD (Baseline)': LABELED_DIR / 'hasildata_kmeans-standard.csv',
    'DE': LABELED_DIR / 'hasildata_kmeans-de.csv',
    'PSO': LABELED_DIR / 'hasildata_kmeans-pso.csv',
    'EOA': LABELED_DIR / 'hasildata_kmeans-eoa.csv'
}

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


@st.cache_data(show_spinner="Training Decision Tree...")
def run_decision_tree(df_clustered: pd.DataFrame) -> tuple[dict, dict, DecisionTreeClassifier]:
    """Decision Tree classifier (unscaled features, max_depth=4)."""
    X_train, X_test, y_train, y_test = _prep_classification_data(df_clustered)

    # Decision tree does not require scaling (matching 04.1_classification_decision_tree.py)
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = _compute_metrics(y_test, y_pred)
    importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_.round(4).tolist()))
    return metrics, importance, model


@st.cache_data(show_spinner="Training SVM...")
def run_svm(df_clustered: pd.DataFrame) -> tuple[dict, dict, SVC, StandardScaler]:
    """Support Vector Machine (SVM) classifier (Z-score scaled, class_weight='balanced')."""
    X_train, X_test, y_train, y_test = _prep_classification_data(df_clustered)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Limit training data to 2000 samples to keep training fast, matching the notebooks
    X_train_svm = X_train_scaled[:2000]
    y_train_svm = y_train[:2000]

    model = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
    model.fit(X_train_svm, y_train_svm)
    y_pred = model.predict(X_test_scaled)

    metrics = _compute_metrics(y_test, y_pred)
    
    # Permutation importance for SVM feature importance
    perm_importance = permutation_importance(model, X_test_scaled, y_test, random_state=42, n_repeats=5)
    importances = np.abs(perm_importance.importances_mean)
    if np.sum(importances) > 0:
        importances /= np.sum(importances)
    importance_dict = dict(zip(FEATURE_COLUMNS, importances.round(4).tolist()))
    return metrics, importance_dict, model, scaler


@st.cache_data(show_spinner="Running cross-dataset classification comparison...")
def run_cross_dataset_comparison() -> pd.DataFrame:
    """Replicates 04.5_classification_comparison.py across all 5 clustering datasets."""
    import time
    from sklearn.model_selection import cross_val_score
    
    tabel_hasil = []
    
    for nama_metode, filepath in FILE_DATASETS.items():
        if not filepath.exists():
            continue
            
        df = pd.read_csv(filepath)
        fitur = [f'Var{i}' for i in range(1, 12)]
        X = df[fitur].values
        y = df['Cluster'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Decision Tree (depth=4)
        model_dt = DecisionTreeClassifier(random_state=42, max_depth=4)
        mulai_dt = time.time()
        val_acc_dt = cross_val_score(model_dt, X_train, y_train, cv=5).mean() * 100
        model_dt.fit(X_train, y_train)
        test_acc_dt = accuracy_score(y_test, model_dt.predict(X_test)) * 100
        selesai_dt = time.time()
        
        tabel_hasil.append({
            'Dataset': nama_metode,
            'Method': 'Decision Tree (DT)',
            'Validation Mean Accuracy (%)': round(val_acc_dt, 2),
            'Test Set Accuracy (%)': round(test_acc_dt, 2),
            'Time(second)': round(selesai_dt - mulai_dt, 4)
        })
        
        # Kernel SVM
        X_train_svm, y_train_svm = X_train[:2000], y_train[:2000]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_svm)
        X_test_scaled = scaler.transform(X_test)
        
        model_svm = SVC(kernel='rbf', random_state=42, class_weight='balanced')
        mulai_svm = time.time()
        val_acc_svm = cross_val_score(model_svm, X_train_scaled, y_train_svm, cv=5).mean() * 100
        model_svm.fit(X_train_scaled, y_train_svm)
        test_acc_svm = accuracy_score(y_test, model_svm.predict(X_test_scaled)) * 100
        selesai_svm = time.time()
        
        tabel_hasil.append({
            'Dataset': nama_metode,
            'Method': 'Kernel SVM (KSVM)',
            'Validation Mean Accuracy (%)': round(val_acc_svm, 2),
            'Test Set Accuracy (%)': round(test_acc_svm, 2),
            'Time(second)': round(selesai_svm - mulai_svm, 4)
        })
        
    return pd.DataFrame(tabel_hasil)



def load_model_if_exists(filename: str):
    """Attempt to load a .pkl model. Returns None if not found."""
    path = MODELS_DIR / filename
    if path.exists():
        return joblib.load(path)
    return None
