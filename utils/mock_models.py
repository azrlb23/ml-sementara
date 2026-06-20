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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
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
    Loads pre-computed clustering results and metrics aligned with the notebooks.
    """
    import numpy as np
    
    labels = None
    sse = 0.0
    sil = 0.0
    db = 9.99
    ch = 0.0
    
    if algorithm == "K-Means Standard" and k == 6:
        labels_file = LABELED_DIR / "hasildata_kmeans-standard.csv"
        if labels_file.exists():
            df_std = pd.read_csv(labels_file)
            df_align = df.merge(df_std[["CustomerID", "Cluster"]], on="CustomerID", how="left")
            labels = df_align["Cluster"].fillna(0).astype(int).values
            
            sse = np.load(MODELS_DIR / "kmeans-standard_inertia.npy").item()
            sil = np.load(MODELS_DIR / "kmeans-standard_silhouette.npy").item()
            db = np.load(MODELS_DIR / "kmeans-standard_db_score.npy").item()
            ch = np.load(MODELS_DIR / "kmeans-standard_ch_score.npy").item()
            
    elif algorithm == "K-Means + DE" and k == 6:
        labels_file = LABELED_DIR / "hasildata_kmeans-de.csv"
        if labels_file.exists():
            df_de = pd.read_csv(labels_file)
            df_align = df.merge(df_de[["CustomerID", "Cluster"]], on="CustomerID", how="left")
            labels = df_align["Cluster"].fillna(0).astype(int).values
            
            sse = np.load(MODELS_DIR / "kmeans-de_inertia.npy").item()
            sil = np.load(MODELS_DIR / "kmeans-de_silhouette.npy").item()
            db = np.load(MODELS_DIR / "kmeans-de_db_score.npy").item()
            ch = np.load(MODELS_DIR / "kmeans-de_ch_score.npy").item()
            
    elif algorithm == "K-Means QLDE" and k == 6:
        labels_file = LABELED_DIR / "hasildata_kmeans-qlde.csv"
        if labels_file.exists():
            df_qlde = pd.read_csv(labels_file)
            df_align = df.merge(df_qlde[["CustomerID", "Cluster"]], on="CustomerID", how="left")
            labels = df_align["Cluster"].fillna(0).astype(int).values
            
            sse = np.load(MODELS_DIR / "qlde_inertia.npy").item()
            sil = np.load(MODELS_DIR / "qlde_silhouette.npy").item()
            db = np.load(MODELS_DIR / "qlde_db_score.npy").item()
            ch = np.load(MODELS_DIR / "qlde_ch_score.npy").item()
            
    # Fallback to dynamic calculation if files are missing or for non-optimal k
    if labels is None or k != 6:
        from sklearn.cluster import KMeans
        from utils.data_loader import load_pca_data
        df_pca = load_pca_data().set_index("CustomerID").reindex(df.set_index("CustomerID").index)
        X_pca = df_pca.values
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_pca)
        labels = model.labels_
        sse = model.inertia_
        sil = silhouette_score(X_pca, labels) if len(np.unique(labels)) > 1 else 0.0
        db = davies_bouldin_score(X_pca, labels) if len(np.unique(labels)) > 1 else 9.99
        ch = calinski_harabasz_score(X_pca, labels) if len(np.unique(labels)) > 1 else 0.0

    result = df.copy()
    result["Cluster"] = labels
    
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
    """Loads pre-computed SSE convergence data over iterations."""
    import numpy as np
    
    de_conv_file = MODELS_DIR / "kmeans-de_convergence.npy"
    qlde_conv_file = MODELS_DIR / "qlde_convergence.npy"
    
    if de_conv_file.exists() and qlde_conv_file.exists():
        de_curve = np.load(de_conv_file)
        qlde_curve = np.load(qlde_conv_file)
        l = min(len(de_curve), len(qlde_curve), max_iter)
        return pd.DataFrame({
            "Iteration": np.arange(1, l + 1),
            "K-Means + DE": de_curve[:l],
            "K-Means QLDE": qlde_curve[:l],
        })
    else:
        from utils.data_loader import load_pca_data
        from utils.algorithms import QLDE, KMeansDE
        df_pca = load_pca_data().drop(columns=["CustomerID"])
        X_pca = df_pca.values
        k = 6
        de = KMeansDE(n_clusters=k, pop_size=15, max_iter=max_iter, random_state=42)
        de.fit(X_pca)
        qlde = QLDE(n_clusters=k, pop_size=15, max_iter=max_iter, random_state=42)
        qlde.fit(X_pca)
        l = min(len(de.convergence_curve_), len(qlde.convergence_curve_), max_iter)
        return pd.DataFrame({
            "Iteration": np.arange(1, l + 1),
            "K-Means + DE": de.convergence_curve_[:l],
            "K-Means QLDE": qlde.convergence_curve_[:l],
        })


@st.cache_data(show_spinner="Evaluating all algorithms for comparison...")
def get_algorithm_comparison_metrics(df: pd.DataFrame, k: int = 6) -> dict:
    """Evaluate selected algorithms dynamically to produce the comparison chart."""
    comparison = {}
    for algo in ["K-Means Standard", "K-Means + DE", "K-Means QLDE"]:
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


# ── Supervised Algorithms (C# Mapping of focused clustering datasets as defined in the repository
FILE_DATASETS = {
    'QLDE (Paper)': LABELED_DIR / 'hasildata_kmeans-qlde.csv',
    'STANDARD (Baseline)': LABELED_DIR / 'hasildata_kmeans-standard.csv',
    'DE': LABELED_DIR / 'hasildata_kmeans-de.csv'
}

def _get_suffix(dataset_name: str) -> str:
    if "QLDE" in dataset_name:
        return "qlde"
    elif "STANDARD" in dataset_name:
        return "standard"
    else:
        return "de"

def _prep_classification_data(df_clustered: pd.DataFrame, cluster_col: str = "Cluster"):
    valid = df_clustered[df_clustered[cluster_col] != -1].copy()
    X = valid[FEATURE_COLUMNS].values
    y = valid[cluster_col].values
    return train_test_split(X, y, test_size=0.2, random_state=42)


def _compute_metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "F1-Score": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }


@st.cache_data(show_spinner="Running Decision Tree...")
def run_decision_tree(df_clustered: pd.DataFrame, dataset_name: str) -> tuple[dict, dict, DecisionTreeClassifier]:
    """Decision Tree classifier (loaded from pre-trained model or dynamically fit)."""
    X_train, X_test, y_train, y_test = _prep_classification_data(df_clustered)
    suffix = _get_suffix(dataset_name)
    
    model = load_model_if_exists(f"model_dt_classification_{suffix}.pkl")
    if model is None:
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
    y_pred = model.predict(X_test)
    metrics = _compute_metrics(y_test, y_pred)
    importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_.round(4).tolist()))
    return metrics, importance, model


@st.cache_data(show_spinner="Running SVM...")
def run_svm(df_clustered: pd.DataFrame, dataset_name: str) -> tuple[dict, dict, SVC, StandardScaler]:
    """Support Vector Machine (SVM) classifier."""
    X_train, X_test, y_train, y_test = _prep_classification_data(df_clustered)
    suffix = _get_suffix(dataset_name)
    
    model = load_model_if_exists(f"model_svm_classification_{suffix}.pkl")
    scaler = load_model_if_exists(f"scaler_svm_{suffix}.pkl")
    
    if model is not None and scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[:2000])
        X_test_scaled = scaler.transform(X_test)
        model = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
        model.fit(X_train_scaled, y_train[:2000])
        
    y_pred = model.predict(X_test_scaled)
    metrics = _compute_metrics(y_test, y_pred)
    
    perm_importance = permutation_importance(model, X_test_scaled, y_test, random_state=42, n_repeats=5)
    importances = np.abs(perm_importance.importances_mean)
    if np.sum(importances) > 0:
        importances /= np.sum(importances)
    importance_dict = dict(zip(FEATURE_COLUMNS, importances.round(4).tolist()))
    return metrics, importance_dict, model, scaler


@st.cache_data(show_spinner="Running AdaBoost...")
def run_adaboost(df_clustered: pd.DataFrame, dataset_name: str) -> tuple[dict, dict, AdaBoostClassifier]:
    """AdaBoost classifier."""
    from sklearn.ensemble import AdaBoostClassifier
    X_train, X_test, y_train, y_test = _prep_classification_data(df_clustered)
    suffix = _get_suffix(dataset_name)
    
    model = load_model_if_exists(f"model_adaboost_classification_{suffix}.pkl")
    if model is None:
        model = AdaBoostClassifier(random_state=42)
        model.fit(X_train, y_train)
        
    y_pred = model.predict(X_test)
    metrics = _compute_metrics(y_test, y_pred)
    importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_.round(4).tolist()))
    return metrics, importance, model


@st.cache_data(show_spinner="Running ANN...")
def run_ann(df_clustered: pd.DataFrame, dataset_name: str) -> tuple[dict, dict, MLPClassifier, StandardScaler]:
    """Artificial Neural Network (ANN) classifier."""
    from sklearn.neural_network import MLPClassifier
    X_train, X_test, y_train, y_test = _prep_classification_data(df_clustered)
    suffix = _get_suffix(dataset_name)
    
    model = load_model_if_exists(f"model_ann_classification_{suffix}.pkl")
    scaler = load_model_if_exists(f"scaler_ann_{suffix}.pkl")
    
    if model is not None and scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
        model.fit(X_train_scaled, y_train)
        
    y_pred = model.predict(X_test_scaled)
    metrics = _compute_metrics(y_test, y_pred)
    
    perm_importance = permutation_importance(model, X_test_scaled, y_test, random_state=42, n_repeats=5)
    importances = np.abs(perm_importance.importances_mean)
    if np.sum(importances) > 0:
        importances /= np.sum(importances)
    importance_dict = dict(zip(FEATURE_COLUMNS, importances.round(4).tolist()))
    return metrics, importance_dict, model, scaler


@st.cache_data(show_spinner="Running cross-dataset classification comparison...")
def run_cross_dataset_comparison() -> pd.DataFrame:
    """Replicates 04.4_classification_comparison.ipynb across all 3 focused clustering datasets."""
    import time
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.neural_network import MLPClassifier
    
    tabel_hasil = []
    
    for nama_metode, filepath in FILE_DATASETS.items():
        if not filepath.exists():
            continue
            
        df = pd.read_csv(filepath)
        fitur = [f'Var{i}' for i in range(1, 12)]
        X = df[fitur].values
        y = df['Cluster'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scaling full for ANN
        scaler_full = StandardScaler()
        X_train_scaled_full = scaler_full.fit_transform(X_train)
        X_test_scaled_full = scaler_full.transform(X_test)
        
        # 1. Decision Tree (DT)
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
        
        # 2. Kernel SVM (KSVM)
        X_train_svm, y_train_svm = X_train[:2000], y_train[:2000]
        scaler_svm = StandardScaler()
        X_train_scaled_svm = scaler_svm.fit_transform(X_train_svm)
        X_test_scaled_svm = scaler_svm.transform(X_test)
        
        model_svm = SVC(kernel='rbf', random_state=42, class_weight='balanced')
        mulai_svm = time.time()
        val_acc_svm = cross_val_score(model_svm, X_train_scaled_svm, y_train_svm, cv=5).mean() * 100
        model_svm.fit(X_train_scaled_svm, y_train_svm)
        test_acc_svm = accuracy_score(y_test, model_svm.predict(X_test_scaled_svm)) * 100
        selesai_svm = time.time()
        
        tabel_hasil.append({
            'Dataset': nama_metode,
            'Method': 'Kernel SVM (KSVM)',
            'Validation Mean Accuracy (%)': round(val_acc_svm, 2),
            'Test Set Accuracy (%)': round(test_acc_svm, 2),
            'Time(second)': round(selesai_svm - mulai_svm, 4)
        })

        # 3. AdaBoost
        model_ada = AdaBoostClassifier(random_state=42)
        mulai_ada = time.time()
        val_acc_ada = cross_val_score(model_ada, X_train, y_train, cv=5).mean() * 100
        model_ada.fit(X_train, y_train)
        test_acc_ada = accuracy_score(y_test, model_ada.predict(X_test)) * 100
        selesai_ada = time.time()
        
        tabel_hasil.append({
            'Dataset': nama_metode,
            'Method': 'AdaBoost',
            'Validation Mean Accuracy (%)': round(val_acc_ada, 2),
            'Test Set Accuracy (%)': round(test_acc_ada, 2),
            'Time(second)': round(selesai_ada - mulai_ada, 4)
        })

        # 4. Artificial Neural Network (ANN)
        model_ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
        mulai_ann = time.time()
        val_acc_ann = cross_val_score(model_ann, X_train_scaled_full, y_train, cv=5).mean() * 100
        model_ann.fit(X_train_scaled_full, y_train)
        test_acc_ann = accuracy_score(y_test, model_ann.predict(X_test_scaled_full)) * 100
        selesai_ann = time.time()
        
        tabel_hasil.append({
            'Dataset': nama_metode,
            'Method': 'Artificial Neural Network (ANN)',
            'Validation Mean Accuracy (%)': round(val_acc_ann, 2),
            'Test Set Accuracy (%)': round(test_acc_ann, 2),
            'Time(second)': round(selesai_ann - mulai_ann, 4)
        })
        
    df_res = pd.DataFrame(tabel_hasil)
    return df_res.sort_values(by=['Dataset', 'Method']).reset_index(drop=True)


def load_model_if_exists(filename: str):
    """Attempt to load a .pkl model. Returns None if not found."""
    path = MODELS_DIR / filename
    if path.exists():
        return joblib.load(path)
    return None
