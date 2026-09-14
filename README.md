# Customer Churn Prediction Using Machine Learning
**MSc Artificial Intelligence and Data Analytics Internship Project**

---

## 1. Project Title
**Customer Churn Prediction Using Machine Learning: An End-to-End Enterprise Attrition Forecasting and Risk Diagnostic System**

---

## 2. Project Overview
Customer attrition (or churn) is a critical performance indicator in the subscription economy, notably within telecommunications, SaaS, and cloud services. Predicting churn before it occurs enables customer success and retention teams to enact data-driven loyalty interventions, personalized service discounts, and proactive contract restructuring.

This project delivers an end-to-end, production-oriented Machine Learning solution trained on the **IBM Telco Customer Churn dataset** (7,043 subscriber records). The system implements strict data hygiene, exploratory data analysis, data leakage prevention, multi-model benchmarking (Logistic Regression, Random Forest, XGBoost), balanced metric evaluation (prioritizing **Recall** and **ROC-AUC**), feature importance interpretation, an interactive **Streamlit web application**, and an analytical **Jupyter Notebook**.

---

## 3. Problem Statement
Telecommunications operators face intense market competition, commoditization of connectivity services, and low switching barriers. Acquiring new subscribers costs **5 to 7 times more** than retaining existing ones. When customers churn without warning, businesses suffer:
- Immediate forfeiture of recurring revenue and Customer Lifetime Value (**CLV**).
- Wasted Customer Acquisition Cost (**CAC**).
- Negative word-of-mouth and market share erosion.

**Core Challenge:** Traditional classification workflows prioritize naive accuracy. In a dataset with 73.5% retained customers, a trivial dummy model predicting "No Churn" achieves 73.5% accuracy while identifying **zero** churners. The business objective requires maximizing **Recall (Sensitivity)** and **ROC-AUC** so that at-risk subscribers are identified and addressed before contract termination.

---

## 4. Project Objectives
1. **Automated Ingestion & Cleaning**: Ingest the IBM Telco dataset, eliminate non-predictive identifiers, and resolve subtle missing values without discarding customer records.
2. **Exploratory Data Analysis (EDA)**: Statistically map churn dynamics across customer demographics, contract commitments, payment channels, and technical add-ons.
3. **Leakage-Free Preprocessing**: Build Scikit-Learn `Pipeline` and `ColumnTransformer` modules fitted strictly on the training partition.
4. **Algorithmic Benchmarking**: Train and contrast linear baselines (Logistic Regression), bagging ensembles (Random Forest), and gradient boosted trees (XGBoost) with class-balancing strategies.
5. **Business Metric Optimization**: Prioritize **Recall** and **ROC-AUC** to penalize catastrophic False Negatives over minor False Positives.
6. **Interpretability & Diagnostics**: Quantify and visualize global feature importances (Gini importance, log-odds coefficients) to explain churn drivers.
7. **Interactive Deployment**: Provide a clean Streamlit web dashboard allowing stakeholders to test individual customer profiles and view real-time risk assessments.

---

## 5. Dataset Description
- **Dataset Name**: IBM Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Total Records**: 7,043 customer accounts
- **Total Attributes**: 21 columns (20 predictors + 1 binary target)
- **Target Variable**: `Churn` (No = 0 [73.46%], Yes = 1 [26.54%])

### Attribute Categorization:
| Category | Features | Description |
| :--- | :--- | :--- |
| **Identifier** | `customerID` | Unique alphanumeric string (dropped during preprocessing) |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Subscriber demographic characteristics and household structure |
| **Account Tenancy**| `tenure` | Duration of continuous subscription in months (0 to 72) |
| **Services Subscribed** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Telephony, fiber/DSL internet, and value-added protection add-ons |
| **Contract Terms** | `Contract`, `PaperlessBilling`, `PaymentMethod` | Month-to-month, 1-year, or 2-year terms; electronic vs. manual billing |
| **Financials** | `MonthlyCharges`, `TotalCharges` | Monthly recurring fees ($) and cumulative billing charges ($) |

---

## 6. Technologies & Libraries Used
- **Language**: Python 3.12+
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn` (Pipelines, ColumnTransformer, Imputers, Encoders, Metrics)
- **Gradient Boosting**: `xgboost`
- **Model Serialization**: `joblib`
- **Web Application**: `streamlit`

---

## 7. Project Structure
```
customer-churn-prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv      # IBM Telco Churn dataset
│
├── notebooks/
│   └── customer_churn_analysis.ipynb             # Story-driven analysis & experimentation
│
├── src/
│   ├── __init__.py                               # Package initializer
│   ├── data_preprocessing.py                     # Ingestion, cleaning, splitting & pipeline builder
│   ├── eda.py                                    # Automated EDA suite & insight engine
│   ├── train_models.py                           # Multi-model training pipeline
│   ├── evaluate_models.py                        # Metric benchmarking, ROC, CM & serialization
│   └── prediction.py                             # Inference engine & diagnostic risk factor mapper
│
├── models/
│   └── best_model.pkl                            # Serialized production pipeline
│
├── plots/                                        # High-resolution generated visual assets
│   ├── churn_distribution.png
│   ├── demographics_vs_churn.png
│   ├── contract_and_billing_churn.png
│   ├── services_churn_comparison.png
│   ├── tenure_monthly_charges_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrices.png
│   ├── roc_curves_comparison.png
│   ├── model_metrics_comparison.png
│   └── feature_importance.png
│
├── app.py                                        # Interactive Streamlit Web Application
├── requirements.txt                              # Pinned library dependencies
├── README.md                                     # Comprehensive project documentation
└── .gitignore                                    # Git exclusion rules
```

---

## 8. Data Preprocessing & Leakage Prevention
1. **Identifier Elimination**: `customerID` dropped to avoid high-cardinality overfitting.
2. **Missing Value Resolution in `TotalCharges`**:
   - In the raw data, `TotalCharges` contained 11 records with whitespace strings (`" "`).
   - Cross-referencing revealed that all 11 records had `tenure == 0` (brand-new accounts that joined during the current cycle).
   - Rather than discarding records, `TotalCharges` was converted to numeric with `0.0` imputed for zero-tenure accounts.
3. **Target Binarization**: `Churn` mapped to `0` (retained) and `1` (churned).
4. **Stratified Train-Test Splitting**:
   - 80% Training ($n=5,634$) and 20% Testing ($n=1,409$).
   - `stratify=y` guarantees an exact 26.54% churn prevalence in both partitions.
5. **ColumnTransformer Pipeline**:
   - **Numerical (`tenure`, `MonthlyCharges`, `TotalCharges`)**: `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
   - **Categorical (16 features)**: `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(drop='first', handle_unknown='ignore')`.
   - **Strict Ordering**: Transformers are fitted **solely on training data**, completely shielding test data from premature distribution leakage.

---

## 9. Exploratory Data Analysis (EDA) Highlights
EDA executed via `src/eda.py` produced key statistical findings:

1. **Contract Type is the Primary Churn Driver**:
   - **Month-to-Month**: **42.71% churn rate**
   - **One-Year**: **11.27% churn rate**
   - **Two-Year**: **2.83% churn rate**
   - *Month-to-month subscribers are 15 times more likely to cancel than two-year contract holders.*
2. **Tenure Flight Window**:
   - Median tenure for churned subscribers is **10 months**, versus **38 months** for retained subscribers.
   - Attrition is heavily concentrated in the onboarding phase (first 1–12 months).
3. **Payment Method Friction**:
   - Subscribers using **Electronic check** churn at **45.29%**, compared to <18% for automated payment methods (Bank transfer or Credit card auto-pay).
4. **Financial Impact**:
   - Median monthly charges for churners are **$79.65**, versus **$64.43** for retained customers, highlighting acute price sensitivity in premium packages.
5. **Protective Power of Value-Added Services**:
   - Customers lacking `TechSupport` or `OnlineSecurity` experience **>40% churn**.
   - Customers with both add-ons exhibit **<15% churn**, confirming that account "stickiness" increases with technical touchpoints.

---

## 10. Machine Learning Models
Three distinct algorithms were implemented with class-balancing mechanisms:

1. **Logistic Regression**:
   - Parametric linear classifier serving as a reliable benchmark.
   - Configured with L2 penalty, $C=0.1$, and `class_weight='balanced'`.
2. **Random Forest Classifier**:
   - Non-linear ensemble bagging model resistant to outliers and multicollinearity.
   - Hyperparameters: `n_estimators=200`, `max_depth=8`, `min_samples_split=10`, `min_samples_leaf=4`, `class_weight='balanced'`.
3. **XGBoost Classifier**:
   - Regularized gradient boosted decision tree ensemble optimizing log-loss.
   - Hyperparameters: `n_estimators=150`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=2.77`.

---

## 11. Evaluation Metrics & The Importance of Recall
In churn modeling, the costs of classification errors are asymmetric:
- **False Positive (FP)**: A loyal subscriber is predicted as churn-risk.
  - *Business Cost*: Nominal outreach cost (e.g. email, proactive support check-in, 10% discount voucher).
- **False Negative (FN)**: An actual churner is predicted as safe and ignored.
  - *Business Cost*: **Catastrophic** — the customer permanently defects, forfeiting thousands in Lifetime Value (**CLV**).

$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

$$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(t) \, d(\text{FPR}(t))$$

Therefore, model selection prioritizes **Recall** (capturing at-risk subscribers) and **ROC-AUC** (discriminative power across all decision thresholds) over naive Accuracy.

---

## 12. Model Comparison & Experimental Results
The models were trained on 5,634 accounts and evaluated on **1,409 unseen test accounts**:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Composite Score* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** 🏆 | **0.7544** | **0.5244** | **0.8048** | **0.6350** | **0.8446** | **0.7797** |
| **XGBoost** | 0.7509 | 0.5201 | 0.7941 | 0.6286 | 0.8444 | 0.7743 |
| **Logistic Regression** | 0.7431 | 0.5105 | 0.7834 | 0.6181 | 0.8413 | 0.7666 |

*\*Composite Score = 0.35 x Recall + 0.35 x ROC-AUC + 0.20 x F1-Score + 0.10 x Accuracy*

### Key Analytical Takeaways:
- **Winning Architecture**: **Random Forest** achieved the highest balanced score, capturing **80.48% of all actual churners** (301 out of 374 at-risk subscribers) with an **ROC-AUC of 0.8446**.
- **XGBoost** performed closely, demonstrating competitive AUC (0.8444) and 79.41% Recall.
- **Logistic Regression** showed solid recall (78.34%) and AUC (0.8413), proving that linear decision boundaries perform admirably when class weights are balanced.

---

## 13. Top Churn Factors (Feature Importance)
Extracted from the winning Random Forest pipeline:
1. `Contract_Month-to-month`: Single strongest predictor of subscriber flight.
2. `tenure`: High customer tenure solidifies brand retention.
3. `TotalCharges` & `MonthlyCharges`: High monthly fees drive cost sensitivity.
4. `InternetService_Fiber optic`: High attrition rate due to premium pricing.
5. `PaymentMethod_Electronic check`: Lack of automated recurring billing increases lapse likelihood.
6. `TechSupport_No` & `OnlineSecurity_No`: Customers lacking technical safety nets churn faster.

---

## 14. How to Run the Project

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### Step 1: Clone or Navigate to the Repository
```bash
cd customer-churn-prediction
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Data Preprocessing
Ingests data, cleans anomalies, and performs stratified splitting:
```bash
python -m src.data_preprocessing
```

### Step 5: Generate Exploratory Data Analysis Plots
Produces publication-quality plots in the `plots/` directory:
```bash
python -m src.eda
```

### Step 6: Train & Benchmark Models
Trains Logistic Regression, Random Forest, and XGBoost; generates diagnostic curves and serializes `models/best_model.pkl`:
```bash
python -m src.evaluate_models
```

### Step 7: Test Inference Engine
Runs single-customer diagnostics on sample test personas:
```bash
python -m src.prediction
```

---

## 15. How to Launch the Streamlit Application
Run the interactive web interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### App Capabilities:
- **Interactive Predictor**: Adjust sliders and dropdowns for all 19 customer features.
- **Real-Time Classification**: Instant badge (`Customer is likely to CHURN` vs. `Customer is likely to STAY`).
- **Probability Gauge & Risk Tier**: Visual risk gauge with Low / Moderate / High tier assignments.
- **Diagnostic Risk Drivers**: Bullet-pointed explanation of high-risk factors (e.g., month-to-month, lack of tech support).
- **Evaluation Dashboard Tab**: Visual display of ROC Curves, Confusion Matrices, Metric Comparisons, and Feature Importance.
- **EDA Tab**: Full gallery of demographic, contract, and service distributions.

---

## 16. Expected Outputs
1. **Serialized Artifact**: `models/best_model.pkl` containing the full Scikit-Learn `Pipeline` (preprocessing + model).
2. **Visualizations**: 10 high-resolution `.png` files saved in `plots/`:
   - `churn_distribution.png`
   - `demographics_vs_churn.png`
   - `contract_and_billing_churn.png`
   - `services_churn_comparison.png`
   - `tenure_monthly_charges_distribution.png`
   - `correlation_heatmap.png`
   - `confusion_matrices.png`
   - `roc_curves_comparison.png`
   - `model_metrics_comparison.png`
   - `feature_importance.png`
3. **Interactive Analysis**: Fully documented Jupyter Notebook `notebooks/customer_churn_analysis.ipynb`.

---

## 17. Future Enhancements
- **Threshold Optimization**: Dynamically adjust decision thresholds based on explicit financial cost matrices ($\text{Cost}_{\text{FP}}$ vs. $\text{Cost}_{\text{FN}}$).
- **Explainable AI (XAI)**: Integrate SHAP (SHapley Additive exPlanations) for local instance explanations.
- **Time-to-Event Survival Analysis**: Deploy Cox Proportional Hazards or Kaplan-Meier estimators to predict *when* a customer will churn.
- **Automated CI/CD**: Implement GitHub Actions pipeline to validate data schema and test inference regressions on pull requests.
#   c u s t o m e r - c h u r n - p r e d i c t i o n  
 #   c u s t o m e r - c h u r n - p r e d i c t i o n  
 #   c u s t o m e r - c h u r n - p r e d i c t i o n  
 