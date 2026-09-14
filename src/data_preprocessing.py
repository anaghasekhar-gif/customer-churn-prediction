"""Data Preprocessing Module for Customer Churn Prediction.

Author: MSc AI & Data Analytics Intern
Project: Customer Churn Prediction Using Machine Learning
Dataset: IBM Telco Customer Churn (WA_Fn-UseC_-Telco-Customer-Churn.csv)

This module handles:
1. Automated dataset acquisition and local verification
2. Data loading and initial schema inspection
3. Data hygiene, missing value resolution, and type conversions
4. Target binarization (No -> 0, Yes -> 1)
5. Stratified train-test splitting to prevent target leakage
6. Scikit-learn ColumnTransformer creation (StandardScaler for numerics, OneHotEncoder for categoricals)
"""

import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Default dataset paths and URLs
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_FILE = os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

PRIMARY_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
FALLBACK_URL = "https://raw.githubusercontent.com/plotly/datasets/master/telco-customer-churn-by-IBM.csv"

# Feature definitions
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

TARGET_COL = "Churn"
DROP_COLS = ["customerID"]


def ensure_dataset(data_path: str = DATA_FILE) -> str:
    """Ensure the dataset exists locally. If not found, download automatically.

    Args:
        data_path: Target path for the CSV dataset.

    Returns:
        Absolute path to the validated CSV dataset.
    """
    if os.path.exists(data_path) and os.path.getsize(data_path) > 0:
        print(f"[INFO] Dataset already present at: {data_path}")
        return data_path

    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    print(f"[INFO] Dataset not found locally. Downloading to {data_path}...")
    try:
        urllib.request.urlretrieve(PRIMARY_URL, data_path)
        print("[INFO] Download completed from primary repository.")
    except Exception as e:
        print(f"[WARNING] Primary download failed: {e}. Trying fallback...")
        urllib.request.urlretrieve(FALLBACK_URL, data_path)
        print("[INFO] Download completed from fallback repository.")

    return data_path


def load_raw_data(data_path: str = DATA_FILE) -> pd.DataFrame:
    """Load the customer churn dataset into a pandas DataFrame.

    Args:
        data_path: Path to the CSV file.

    Returns:
        pd.DataFrame containing raw data.
    """
    valid_path = ensure_dataset(data_path)
    df = pd.read_csv(valid_path)
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """Inspect dataset shape, column datatypes, and missing counts.

    Args:
        df: Raw or processed dataframe.

    Returns:
        Dictionary with basic inspection metadata.
    """
    info_dict = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    print(f"\n{'='*50}\n[DATA INSPECTION SUMMARY]\n{'='*50}")
    print(f"Dimensions: {info_dict['shape'][0]} rows, {info_dict['shape'][1]} columns")
    print(f"Duplicate records: {info_dict['duplicate_rows']}")
    return info_dict


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Telco churn dataset.

    1. Drop irrelevant identifier columns (e.g. customerID).
    2. Convert TotalCharges from object/string to float64 (handling whitespace blanks).
    3. Fill blank TotalCharges (where tenure == 0) with 0.0.
    4. Map target variable 'Churn' to binary integer (0 for 'No', 1 for 'Yes') if present.

    Args:
        df: Raw pandas DataFrame.

    Returns:
        Cleaned pandas DataFrame ready for feature engineering and splitting.
    """
    df_clean = df.copy()

    # Drop identifier columns
    cols_to_drop = [col for col in DROP_COLS if col in df_clean.columns]
    if cols_to_drop:
        df_clean.drop(columns=cols_to_drop, inplace=True)
        print(f"[INFO] Dropped identifier columns: {cols_to_drop}")

    # Fix TotalCharges whitespace / string values
    if "TotalCharges" in df_clean.columns:
        # Replace empty spaces with NaN and convert to numeric
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"].astype(str).str.strip(), errors="coerce")
        missing_total = df_clean["TotalCharges"].isnull().sum()
        if missing_total > 0:
            print(f"[INFO] Found {missing_total} missing values in TotalCharges (tenure=0). Imputing with 0.0.")
            df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

    # Convert SeniorCitizen to string category or keep as int for consistent encoding
    if "SeniorCitizen" in df_clean.columns:
        df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].astype(int)

    # Convert Churn target to binary (0/1)
    if TARGET_COL in df_clean.columns:
        target_map = {"No": 0, "Yes": 1, "0": 0, "1": 1, 0: 0, 1: 1}
        df_clean[TARGET_COL] = df_clean[TARGET_COL].astype(str).str.strip().map(target_map).fillna(0).astype(int)
        print(f"[INFO] Binarized target column '{TARGET_COL}': 0 for No, 1 for Yes")

    return df_clean


def build_preprocessor() -> ColumnTransformer:
    """Construct Scikit-Learn ColumnTransformer for feature scaling and encoding.

    Numerical Pipeline:
      - SimpleImputer(strategy='median')
      - StandardScaler()

    Categorical Pipeline:
      - SimpleImputer(strategy='most_frequent')
      - OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

    Returns:
        sklearn.compose.ColumnTransformer object.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified train-test split.

    Args:
        df: Cleaned dataframe containing features and target.
        test_size: Proportion of dataset to include in test split (default 0.20).
        random_state: Random seed for reproducibility (default 42).

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"[INFO] Stratified train-test split complete:")
    print(f"       Training samples: {len(X_train)} (Churn rate: {y_train.mean():.2%})")
    print(f"       Testing samples:  {len(X_test)} (Churn rate: {y_test.mean():.2%})")

    return X_train, X_test, y_train, y_test


def get_preprocessed_data():
    """Convenience pipeline function that loads, cleans, and splits the data.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor, df_clean)
    """
    raw_df = load_raw_data()
    inspect_data(raw_df)
    df_clean = clean_data(raw_df)
    X_train, X_test, y_train, y_test = split_data(df_clean)
    preprocessor = build_preprocessor()
    return X_train, X_test, y_train, y_test, preprocessor, df_clean


if __name__ == "__main__":
    print("[RUNNING] Testing Data Preprocessing Pipeline...")
    X_train, X_test, y_train, y_test, preprocessor, df_clean = get_preprocessed_data()

    # Fit preprocessor strictly on training data
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Get feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERICAL_FEATURES + cat_feature_names

    print(f"[SUCCESS] Preprocessed training feature matrix shape: {X_train_proc.shape}")
    print(f"[SUCCESS] Preprocessed test feature matrix shape:     {X_test_proc.shape}")
    print(f"[SUCCESS] Total engineered/encoded features: {len(feature_names)}")
