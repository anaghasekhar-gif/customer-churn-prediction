"""Model Training Module for Customer Churn Prediction.

Author: MSc AI & Data Analytics Intern
Project: Customer Churn Prediction Using Machine Learning
Dataset: IBM Telco Customer Churn

This module:
1. Builds full Scikit-Learn Pipelines combining feature preprocessor and classifiers.
2. Configures and trains 3 primary classification models:
   - Logistic Regression (Linear baseline with balanced class weights)
   - Random Forest Classifier (Ensemble bagging model)
   - XGBoost Classifier (Gradient boosted decision trees)
3. Fits models strictly on training data to eliminate data leakage.
4. Generates both class predictions and predicted probabilities.
5. Returns trained model pipelines ready for evaluation and serialization.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from src.data_preprocessing import (
    build_preprocessor,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


def get_model_candidates(scale_pos_weight: float = 1.0) -> dict[str, object]:
    """Initialize dictionary of classification model candidates.

    Args:
        scale_pos_weight: Ratio of negative to positive samples to handle class imbalance.

    Returns:
        dict mapping model names to instantiated classifier objects.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=0.1,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }
    return models


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Pipeline]:
    """Train full pipelines for each model candidate.

    Each pipeline encapsulates:
    - Step 1: Preprocessor (StandardScaler for numerics, OneHotEncoder for categoricals)
    - Step 2: Classifier

    Args:
        X_train: Training features DataFrame.
        y_train: Training target binary Series.

    Returns:
        dict mapping model names to fitted Pipeline objects.
    """
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    print(f"[TRAINING] Training class ratio (0:1) = {neg_count}:{pos_count} (scale_pos_weight = {scale_pos_weight:.2f})")

    candidates = get_model_candidates(scale_pos_weight=scale_pos_weight)
    fitted_pipelines = {}

    for name, clf in candidates.items():
        print(f"\n[TRAINING] Fitting Pipeline for '{name}'...")
        preprocessor = build_preprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])

        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline
        print(f"[SUCCESS] '{name}' trained successfully.")

    return fitted_pipelines


def get_feature_names_from_preprocessor(preprocessor) -> list[str]:
    """Extract ordered feature names produced by the ColumnTransformer."""
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return NUMERICAL_FEATURES + cat_feature_names


if __name__ == "__main__":
    from src.data_preprocessing import get_preprocessed_data

    print("[RUNNING] Training Candidate Machine Learning Models...")
    X_train, X_test, y_train, y_test, preprocessor, df_clean = get_preprocessed_data()
    fitted_pipelines = train_all_models(X_train, y_train)
    print(f"\n[DONE] Successfully trained {len(fitted_pipelines)} models: {list(fitted_pipelines.keys())}")
