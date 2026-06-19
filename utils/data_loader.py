"""
data_loader.py
Handles all data loading and caching for the Streamlit app.
Data comes from Anggota 1 (Data Engineer) output.
"""

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
CLEAN_CSV = DATA_DIR / "clean_customer_features.csv"
ANOMALOUS_CSV = DATA_DIR / "anomalous_customers.csv"

# ── Feature definitions ───────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgSpending",
    "UniqueProducts",
    "CancelFrequency",
    "AvgMonthlySpending",
]

FEATURE_LABELS = {
    "Recency": "Recency (days)",
    "Frequency": "Frequency (transactions)",
    "Monetary": "Monetary (total spend)",
    "AvgSpending": "Avg Spending / Item",
    "UniqueProducts": "Unique Products",
    "CancelFrequency": "Cancel Frequency",
    "AvgMonthlySpending": "Avg Monthly Spending",
}

# ── Loaders ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading customer data...")
def load_clean_data() -> pd.DataFrame:
    """Load the cleaned, preprocessed customer feature dataset (Anggota 1 output)."""
    if not CLEAN_CSV.exists():
        st.error(
            f"❌ File not found: `{CLEAN_CSV}`\n\n"
            "Please place `clean_customer_features.csv` in `data/processed/`"
        )
        st.stop()
    df = pd.read_csv(CLEAN_CSV)
    df["CustomerID"] = df["CustomerID"].astype(int)
    return df


@st.cache_data(show_spinner="Loading anomalous data...")
def load_anomalous_data() -> pd.DataFrame:
    """Load the anomalous/outlier customer dataset (DBSCAN noise, Anggota 1 output)."""
    if not ANOMALOUS_CSV.exists():
        st.warning("⚠️ `anomalous_customers.csv` not found. Skipping anomaly section.")
        return pd.DataFrame()
    df = pd.read_csv(ANOMALOUS_CSV)
    df["CustomerID"] = df["CustomerID"].astype(int)
    return df


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
