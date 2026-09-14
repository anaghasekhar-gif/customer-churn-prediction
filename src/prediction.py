"""Prediction Module for Customer Churn Inference.

Author: MSc AI & Data Analytics Intern
Project: Customer Churn Prediction Using Machine Learning
Dataset: IBM Telco Customer Churn

This module provides:
1. Production-grade loading and caching of the serialized model pipeline (models/best_model.pkl).
2. Type-checked, robust single-customer and batch inference functions.
3. Decision rules formatting:
   - "Customer is likely to CHURN"
   - "Customer is likely to STAY"
4. Churn probability calculation (e.g. 74.2%) and risk tier categorization (Low, Moderate, High).
5. Interpretable risk factor diagnostics based on customer input attributes.
"""

import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

# Cached model holder
_CACHED_MODEL_BUNDLE = None


def load_model_bundle(model_path: str = MODEL_PATH) -> dict:
    """Load and cache the trained model pipeline and metadata.

    Args:
        model_path: Filepath to the serialized best model.

    Returns:
        dict containing 'model_name', 'pipeline', and 'comparison_metrics'.
    """
    global _CACHED_MODEL_BUNDLE
    if _CACHED_MODEL_BUNDLE is not None:
        return _CACHED_MODEL_BUNDLE

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model file not found at '{model_path}'. "
            "Please train and evaluate models first by executing: python -m src.evaluate_models"
        )

    _CACHED_MODEL_BUNDLE = joblib.load(model_path)
    return _CACHED_MODEL_BUNDLE


def analyze_customer_risk_factors(customer_dict: dict) -> list[str]:
    """Identify customer-specific risk drivers based on domain findings.

    Args:
        customer_dict: Raw inputs for a single customer.

    Returns:
        list of human-readable risk observation strings.
    """
    risk_factors = []

    # Contract Type
    contract = customer_dict.get("Contract", "")
    if contract == "Month-to-month":
        risk_factors.append("Month-to-month contract (Highest churn segment across all customer cohorts)")
    elif contract in ["One year", "Two year"]:
        risk_factors.append(f"Long-term commitment ({contract}) strongly mitigates churn risk")

    # Tenure
    tenure = float(customer_dict.get("tenure", 0))
    if tenure <= 12:
        risk_factors.append(f"Early tenure stage ({int(tenure)} months) - within critical onboarding attrition window")
    elif tenure >= 48:
        risk_factors.append(f"High tenure ({int(tenure)} months) indicates strong brand loyalty")

    # Internet Service
    internet = customer_dict.get("InternetService", "")
    if internet == "Fiber optic":
        risk_factors.append("Fiber optic internet subscription (statistically higher churn rate due to cost/service sensitivity)")

    # Support & Security Services
    if customer_dict.get("OnlineSecurity") == "No":
        risk_factors.append("Absence of Online Security add-on correlates with elevated churn")
    if customer_dict.get("TechSupport") == "No":
        risk_factors.append("Absence of Tech Support service correlates with elevated churn")

    # Payment & Billing
    if customer_dict.get("PaymentMethod") == "Electronic check":
        risk_factors.append("Payment via Electronic Check (highest churn rate among payment methods)")
    if customer_dict.get("PaperlessBilling") == "Yes":
        risk_factors.append("Paperless Billing enrolled (modestly higher churn correlation)")

    # Monthly Charges
    monthly = float(customer_dict.get("MonthlyCharges", 0))
    if monthly > 75.0:
        risk_factors.append(f"High monthly charges (${monthly:.2f}/mo) increases price sensitivity")

    return risk_factors


def predict_churn(customer_input: dict | pd.DataFrame, threshold: float = 0.50) -> dict:
    """Predict churn probability and binary label for single or batch inputs.

    Args:
        customer_input: dict of features for single customer or pd.DataFrame for batch.
        threshold: Decision classification threshold (default 0.50).

    Returns:
        dict containing:
          - 'prediction_label': 'Customer is likely to CHURN' or 'Customer is likely to STAY'
          - 'churn_prediction': 1 or 0
          - 'churn_probability': float (0.0 to 1.0)
          - 'churn_probability_pct': formatted percentage string (e.g. '74.2%')
          - 'risk_level': 'Low', 'Moderate', or 'High'
          - 'risk_factors': list of qualitative factors
          - 'model_used': name of the model
    """
    bundle = load_model_bundle()
    pipeline = bundle["pipeline"]
    model_name = bundle["model_name"]

    if isinstance(customer_input, dict):
        df_input = pd.DataFrame([customer_input])
        single_mode = True
    else:
        df_input = customer_input.copy()
        single_mode = False

    # Ensure numeric columns are properly typed
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        if col in df_input.columns:
            df_input[col] = pd.to_numeric(df_input[col], errors="coerce").fillna(0.0)

    # Compute probability & prediction using the full end-to-end pipeline
    probabilities = pipeline.predict_proba(df_input)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    if single_mode:
        prob = float(probabilities[0])
        pred = int(predictions[0])
        prob_pct = f"{prob * 100:.1f}%"

        if pred == 1:
            label = "Customer is likely to CHURN"
        else:
            label = "Customer is likely to STAY"

        if prob >= 0.65:
            risk_level = "High"
        elif prob >= 0.35:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        risk_factors = analyze_customer_risk_factors(customer_input)

        return {
            "prediction_label": label,
            "churn_prediction": pred,
            "churn_probability": prob,
            "churn_probability_pct": prob_pct,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "model_used": model_name,
        }

    # Batch return
    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "model_used": model_name,
    }


if __name__ == "__main__":
    print("[TEST] Demonstrating Churn Prediction Engine with Archetype Profiles...")

    # Profile 1: High Risk Churn Archetype
    high_risk_customer = {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.80,
        "TotalCharges": 191.60,
    }

    # Profile 2: Loyal Low Risk Archetype
    low_risk_customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 65,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 45.20,
        "TotalCharges": 2938.00,
    }

    try:
        res_high = predict_churn(high_risk_customer)
        print("\n--- HIGH RISK TEST PROFILE ---")
        print(f"Result:      {res_high['prediction_label']}")
        print(f"Probability: {res_high['churn_probability_pct']} ({res_high['risk_level']} Risk)")
        print("Identified Risk Factors:")
        for rf in res_high["risk_factors"]:
            print(f"  * {rf}")

        res_low = predict_churn(low_risk_customer)
        print("\n--- LOW RISK TEST PROFILE ---")
        print(f"Result:      {res_low['prediction_label']}")
        print(f"Probability: {res_low['churn_probability_pct']} ({res_low['risk_level']} Risk)")
        print("Identified Risk Factors:")
        for rf in res_low["risk_factors"]:
            print(f"  * {rf}")

    except Exception as e:
        print(f"[NOTE] Model not yet trained or file pending: {e}")
