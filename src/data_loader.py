"""
data_loader.py
--------------
Functions to load, inspect, and clean the IBM Telco Customer Churn dataset.
"""

import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame and return it."""
    df = pd.read_csv(filepath)
    return df


def get_overview(df: pd.DataFrame) -> None:
    """Print shape, dtypes, and head of the DataFrame."""
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    print("Column dtypes:")
    print(df.dtypes.to_string())
    print(f"\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string() if missing.sum() > 0 else "  None")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps:
      1. Drop customerID (not predictive)
      2. Convert TotalCharges to numeric (whitespace strings → NaN → fill with median)
      3. Encode binary target column Churn as 0/1
      4. Strip whitespace from all object columns
    Returns a cleaned copy.
    """
    df = df.copy()

    # 1. Drop customerID
    df.drop(columns=["customerID"], inplace=True)

    # 2. Fix TotalCharges — coerce non-numeric values to NaN then impute
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isnull().sum()
    if n_missing > 0:
        median_val = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_val)
        print(f"[clean_data] Imputed {n_missing} missing TotalCharges with median ({median_val:.2f})")

    # 3. Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # 4. Encode target as integer
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode all remaining categorical columns using one-hot encoding.
    Binary yes/no columns (excluding target) are mapped to 0/1 directly.
    Returns the encoded DataFrame.
    """
    df = df.copy()

    binary_map = {"Yes": 1, "No": 0}
    binary_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    ]
    # Also handle gender
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    # One-hot encode remaining categoricals
    remaining_cat = df.select_dtypes(include="object").columns.tolist()
    if remaining_cat:
        df = pd.get_dummies(df, columns=remaining_cat, drop_first=True)

    return df
