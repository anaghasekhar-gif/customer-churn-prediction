"""Streamlit Web Application for Customer Churn Prediction.

Author: MSc AI & Data Analytics Intern
Project: Customer Churn Prediction Using Machine Learning
Dataset: IBM Telco Customer Churn
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E3A8A;
    }
    .churn-danger {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 1.2rem;
        border-radius: 0.5rem;
        font-size: 1.3rem;
        font-weight: bold;
        text-align: center;
        border: 1px solid #F87171;
    }
    .churn-safe {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 1.2rem;
        border-radius: 0.5rem;
        font-size: 1.3rem;
        font-weight: bold;
        text-align: center;
        border: 1px solid #34D399;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load prediction engine safely
@st.cache_resource
def get_prediction_engine():
    from src.prediction import predict_churn, load_model_bundle
    bundle = load_model_bundle()
    return predict_churn, bundle


def main():
    st.markdown('<div class="main-header">Customer Churn Prediction System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">MSc Artificial Intelligence & Data Analytics Internship Project — '
        'End-to-End Predictive Machine Learning System powered by Scikit-Learn and XGBoost.</div>',
        unsafe_allow_html=True,
    )

    # Sidebar: Project metadata & model stats
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2040/2040504.png", width=70)
    st.sidebar.title("System Overview")
    st.sidebar.info(
        "**Objective:** Identify subscribers at high risk of attrition and diagnose root causes "
        "to trigger proactive customer retention strategies."
    )

    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    best_model_file = os.path.join(models_dir, "best_model.pkl")

    # Check model presence
    if not os.path.exists(best_model_file):
        st.error("⚠️ Trained model pipeline not found in `models/best_model.pkl`. Please execute model training first.")
        st.code("python -m src.evaluate_models", language="bash")
        return

    try:
        predict_fn, bundle = get_prediction_engine()
        best_model_name = bundle.get("model_name", "Machine Learning Pipeline")
        metrics = bundle.get("comparison_metrics", {})
    except Exception as e:
        st.error(f"Error loading model pipeline: {e}")
        return

    st.sidebar.success(f"**Active Model:** {best_model_name}")
    if metrics:
        st.sidebar.markdown(f"**ROC-AUC:** `{metrics.get('ROC-AUC', 0.84):.3f}`")
        st.sidebar.markdown(f"**Recall:** `{metrics.get('Recall', 0.80):.3f}`")
        st.sidebar.markdown(f"**F1-Score:** `{metrics.get('F1-Score', 0.63):.3f}`")
        st.sidebar.markdown(f"**Accuracy:** `{metrics.get('Accuracy', 0.76):.3f}`")

    # Main application navigation tabs
    tab_predict, tab_evaluation, tab_eda = st.tabs([
        "🎯 Customer Churn Predictor",
        "📈 Model Evaluation & Benchmarks",
        "🔍 Exploratory Data Insights",
    ])

    # ---------------- TAB 1: PREDICTOR FORM ----------------
    with tab_predict:
        st.subheader("Customer Profile & Contract Details")
        st.write("Fill in the customer attributes below to evaluate attrition risk in real time.")

        with st.form("churn_prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### 👤 Demographics")
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes (Senior)" if x == 1 else "No")
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["No", "Yes"])

                st.markdown("#### ⏱️ Subscription Tenure")
                tenure = st.slider("Tenure (Months with Company)", min_value=0, max_value=72, value=12, step=1)

            with col2:
                st.markdown("#### 🌐 Services Subscribed")
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

            with col3:
                st.markdown("#### 📺 Entertainment & Contract")
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment_method = st.selectbox(
                    "Payment Method",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                )

                st.markdown("#### 💳 Billing Details")
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=75.0, step=1.0)
                # Estimate total charges dynamically based on tenure if desired
                estimated_total = float(np.round(monthly_charges * max(tenure, 1), 2))
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=estimated_total, step=10.0)

            submitted = st.form_submit_button("⚡ Predict Customer Churn", use_container_width=True)

        if submitted:
            customer_data = {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }

            with st.spinner("Analyzing risk indicators through machine learning pipeline..."):
                result = predict_fn(customer_data)

            st.write("---")
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                if result["churn_prediction"] == 1:
                    st.markdown(
                        f'<div class="churn-danger">🚨 {result["prediction_label"].upper()}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="churn-safe">✅ {result["prediction_label"].upper()}</div>',
                        unsafe_allow_html=True,
                    )

                st.write("")
                st.metric(
                    label="Predicted Churn Probability",
                    value=result["churn_probability_pct"],
                    delta=f"{result['risk_level']} Risk Tier",
                    delta_color="inverse" if result["churn_prediction"] == 1 else "normal",
                )
                st.progress(result["churn_probability"])

            with res_col2:
                st.markdown("#### 📋 Diagnostic Risk Assessment")
                if result["risk_factors"]:
                    for factor in result["risk_factors"]:
                        st.markdown(f"- 📌 {factor}")
                else:
                    st.write("No severe individual risk flags triggered.")

                st.info(
                    f"**Model In Service:** `{result['model_used']}`\n\n"
                    "**Recommended Business Action:** "
                    + ("Initiate proactive customer retention intervention (loyalty discount, contract lock-in offer, or direct customer care follow-up)."
                       if result["churn_prediction"] == 1
                       else "Maintain standard engagement cadence; customer shows strong retention traits.")
                )

    # ---------------- TAB 2: MODEL EVALUATION ----------------
    with tab_evaluation:
        st.subheader("Model Benchmark & Evaluation Results")
        st.write(
            "Evaluation performed across three primary algorithms on unseen stratified test data. "
            "Because customer churn prioritizes mitigating unobserved attrition, **Recall** and **ROC-AUC** "
            "are heavily weighted over simple accuracy."
        )

        cm_plot = os.path.join(plots_dir, "confusion_matrices.png")
        roc_plot = os.path.join(plots_dir, "roc_curves_comparison.png")
        metrics_plot = os.path.join(plots_dir, "model_metrics_comparison.png")
        feat_plot = os.path.join(plots_dir, "feature_importance.png")

        eval_col1, eval_col2 = st.columns(2)
        with eval_col1:
            if os.path.exists(roc_plot):
                st.image(roc_plot, caption="ROC Curves & AUC Comparison Across Models", use_container_width=True)
            if os.path.exists(cm_plot):
                st.image(cm_plot, caption="Confusion Matrices by Model", use_container_width=True)

        with eval_col2:
            if os.path.exists(metrics_plot):
                st.image(metrics_plot, caption="Model Metrics Benchmark", use_container_width=True)
            if os.path.exists(feat_plot):
                st.image(feat_plot, caption="Top Feature Importances Affecting Churn", use_container_width=True)

    # ---------------- TAB 3: EXPLORATORY DATA ANALYSIS ----------------
    with tab_eda:
        st.subheader("Exploratory Data Analysis Findings")
        st.write("Visualizations generated from 7,043 customer accounts in the IBM Telco Churn dataset.")

        eda_col1, eda_col2 = st.columns(2)

        churn_dist = os.path.join(plots_dir, "churn_distribution.png")
        tenure_dist = os.path.join(plots_dir, "tenure_monthly_charges_distribution.png")
        contract_dist = os.path.join(plots_dir, "contract_and_billing_churn.png")
        services_dist = os.path.join(plots_dir, "services_churn_comparison.png")
        corr_heatmap = os.path.join(plots_dir, "correlation_heatmap.png")

        with eda_col1:
            if os.path.exists(churn_dist):
                st.image(churn_dist, caption="Class Balance: Churn vs. Retained Customers", use_container_width=True)
            if os.path.exists(contract_dist):
                st.image(contract_dist, caption="Contract Type and Payment Method Drivers", use_container_width=True)
            if os.path.exists(corr_heatmap):
                st.image(corr_heatmap, caption="Feature Correlations with Customer Churn", use_container_width=True)

        with eda_col2:
            if os.path.exists(tenure_dist):
                st.image(tenure_dist, caption="Tenure and Monthly Charges Distribution by Churn", use_container_width=True)
            if os.path.exists(services_dist):
                st.image(services_dist, caption="Value-Added Services and Technical Support Impact", use_container_width=True)


if __name__ == "__main__":
    main()
