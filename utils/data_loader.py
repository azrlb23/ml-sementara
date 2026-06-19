"""
utils/data_loader.py
Handles all data loading and caching for the Streamlit app.
Loads the exact datasets aligned with the nplrahman922 'kayis' branch.
"""

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
LABELED_DIR = Path(__file__).parent.parent / "data" / "Labeled"

RAW_CSV = DATA_DIR / "customer_features_raw.csv"
SCALED_CSV = DATA_DIR / "customer_features_scaled.csv"
PCA_CSV = DATA_DIR / "customer_features_pca.csv"
ANOMALY_CSV = DATA_DIR / "anomalous_customers.csv"

# Mapping from Var name to friendly name
COLUMN_MAPPING = {
    "Var1": "Recency",
    "Var2": "Frequency",
    "Var3": "TotalProducts",
    "Var4": "Monetary",
    "Var5": "AvgSpending",
    "Var6": "UniqueProducts",
    "Var7": "AvgDaysToPurchase",
    "Var8": "ExpectedPurchaseDays",
    "Var9": "FromUK",
    "Var10": "CancelFrequency",
    "Var11": "AvgMonthlySpending",
}

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

FEATURE_LABELS = {
    "Recency": "Recency (days)",
    "Frequency": "Frequency (transactions)",
    "TotalProducts": "Total Products Purchased",
    "Monetary": "Monetary (total spend)",
    "AvgSpending": "Avg Spending / Transaction",
    "UniqueProducts": "Unique Products Purchased",
    "AvgDaysToPurchase": "Avg Days Between Purchases",
    "ExpectedPurchaseDays": "Expected Next Purchase (days)",
    "FromUK": "Is United Kingdom Customer",
    "CancelFrequency": "Cancel Frequency",
    "AvgMonthlySpending": "Avg Monthly Spending",
}

# ── Loaders ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading raw customer features...")
def load_clean_data() -> pd.DataFrame:
    """Load the raw customer feature dataset (kayis branch)."""
    if not RAW_CSV.exists():
        st.error(
            f"❌ File not found: `{RAW_CSV}`\n\n"
            "Please place `customer_features_raw.csv` in `data/processed/`"
        )
        st.stop()
    df = pd.read_csv(RAW_CSV)
    df["CustomerID"] = df["CustomerID"].astype(int)
    # Rename variables to readable names if they exist in column mappings
    df = df.rename(columns=COLUMN_MAPPING)
    return df


@st.cache_data(show_spinner="Loading scaled customer features...")
def load_scaled_data() -> pd.DataFrame:
    """Load the scaled customer feature dataset (kayis branch)."""
    if not SCALED_CSV.exists():
        st.error(
            f"❌ File not found: `{SCALED_CSV}`\n\n"
            "Please place `customer_features_scaled.csv` in `data/processed/`"
        )
        st.stop()
    df = pd.read_csv(SCALED_CSV)
    df["CustomerID"] = df["CustomerID"].astype(int)
    df = df.rename(columns=COLUMN_MAPPING)
    return df


@st.cache_data(show_spinner="Loading PCA reduced coordinates...")
def load_pca_data() -> pd.DataFrame:
    """Load the PCA reduced features (kayis branch)."""
    if not PCA_CSV.exists():
        st.error(
            f"❌ File not found: `{PCA_CSV}`\n\n"
            "Please place `customer_features_pca.csv` in `data/processed/`"
        )
        st.stop()
    df = pd.read_csv(PCA_CSV)
    df["CustomerID"] = df["CustomerID"].astype(int)
    return df


@st.cache_data(show_spinner="Loading labeled QLDE data...")
def load_labeled_qlde_data() -> pd.DataFrame:
    """Load the final customer segment labeled data (from QLDE)."""
    qlde_labeled_path = LABELED_DIR / "hasildata_kmeans-qlde.csv"
    if not qlde_labeled_path.exists():
        st.error(
            f"❌ File not found: `{qlde_labeled_path}`\n\n"
            "Please place `hasildata_kmeans-qlde.csv` in `data/Labeled/`"
        )
        st.stop()
    df = pd.read_csv(qlde_labeled_path)
    df["CustomerID"] = df["CustomerID"].astype(int)
    df = df.rename(columns=COLUMN_MAPPING)
    return df


@st.cache_data(show_spinner="Loading anomalous customer features...")
def load_anomaly_data() -> pd.DataFrame:
    """Load the anomalous customer dataset if it exists."""
    if not ANOMALY_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(ANOMALY_CSV)
        if "CustomerID" in df.columns:
            df["CustomerID"] = df["CustomerID"].astype(int)
        df = df.rename(columns=COLUMN_MAPPING)
        return df
    except Exception:
        return pd.DataFrame()



def get_feature_columns() -> list[str]:
    """Return the list of ML feature column names."""
    return FEATURE_COLUMNS


def get_feature_labels() -> dict[str, str]:
    """Return human-readable labels for each feature."""
    return FEATURE_LABELS


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Return key summary statistics for the dashboard header."""
    return {
        "total_customers": len(df),
        "avg_recency": round(df["Recency"].mean(), 1),
        "avg_frequency": round(df["Frequency"].mean(), 1),
        "avg_monetary": round(df["Monetary"].mean(), 2),
        "avg_cancel_rate": round(
            (df["CancelFrequency"] > 0).mean() * 100, 1
        ),
    }
