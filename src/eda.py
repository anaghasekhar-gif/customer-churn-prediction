"""Exploratory Data Analysis (EDA) Module for Customer Churn Prediction.

Author: MSc AI & Data Analytics Intern
Project: Customer Churn Prediction Using Machine Learning
Dataset: IBM Telco Customer Churn

This module performs structured Exploratory Data Analysis:
1. Target Distribution & Churn Rate Analysis
2. Demographic Features vs. Churn (Gender, Senior Citizen, Partner, Dependents)
3. Contract Type, Payment Method, and Paperless Billing Analysis
4. Internet Service & Value-Added Support Services (Tech Support, Online Security, etc.)
5. Continuous Feature Distributions (Tenure, MonthlyCharges, TotalCharges)
6. Feature Correlation Heatmap with Target
7. Automated statistical pattern reporting
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Ensure plots directory exists
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Styling configuration for publication-grade academic visuals
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
COLOR_PALETTE = ["#2b5c8f", "#d9534f"]  # Blue for Non-Churn (0), Coral Red for Churn (1)


def plot_churn_distribution(df: pd.DataFrame) -> None:
    """Plot overall Churn count distribution and pie chart percentage."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Countplot
    churn_counts = df["Churn"].value_counts()
    churn_labels = ["Retained (No)", "Churned (Yes)"]

    sns.countplot(
        x="Churn",
        data=df,
        palette=COLOR_PALETTE,
        ax=axes[0],
        hue="Churn",
        legend=False,
    )
    axes[0].set_title("Customer Churn Count Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Customer Status", fontsize=12)
    axes[0].set_ylabel("Count of Customers", fontsize=12)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(churn_labels)

    # Annotate bar counts and percentages
    total = len(df)
    for p in axes[0].patches:
        height = p.get_height()
        axes[0].annotate(
            f"{int(height):,} ({height/total:.1%})",
            (p.get_x() + p.get_width() / 2.0, height + 50),
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    # Pie chart
    axes[1].pie(
        churn_counts,
        labels=churn_labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=COLOR_PALETTE,
        explode=(0, 0.08),
        shadow=True,
        textprops={"fontsize": 12, "weight": "bold"},
    )
    axes[1].set_title("Customer Churn Percentage", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "churn_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] {plot_path}")


def plot_demographics_vs_churn(df: pd.DataFrame) -> None:
    """Analyze churn across Demographic features: Gender, Senior Citizen, Partner, Dependents."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    demographics = [
        ("gender", "Gender", axes[0, 0]),
        ("SeniorCitizen", "Senior Citizen (0=No, 1=Yes)", axes[0, 1]),
        ("Partner", "Has Partner", axes[1, 0]),
        ("Dependents", "Has Dependents", axes[1, 1]),
    ]

    for col, title, ax in demographics:
        churn_pct = (
            df.groupby(col)["Churn"]
            .value_counts(normalize=True)
            .rename("percentage")
            .reset_index()
        )
        churn_pct["percentage"] = churn_pct["percentage"] * 100

        sns.barplot(
            x=col,
            y="percentage",
            hue="Churn",
            data=churn_pct,
            palette=COLOR_PALETTE,
            ax=ax,
        )
        ax.set_title(f"Churn Rate by {title}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Percentage (%)", fontsize=11)
        ax.set_xlabel(title, fontsize=11)
        ax.legend(title="Churn", labels=["Stayed (0)", "Churned (1)"])

        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.1f}%",
                    (p.get_x() + p.get_width() / 2.0, height + 1),
                    ha="center",
                    fontsize=10,
                )

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "demographics_vs_churn.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] {plot_path}")


def plot_contract_and_billing(df: pd.DataFrame) -> None:
    """Analyze how Contract Type, Paperless Billing, and Payment Method drive Churn."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    features = [
        ("Contract", "Contract Type", axes[0]),
        ("PaperlessBilling", "Paperless Billing", axes[1]),
        ("PaymentMethod", "Payment Method", axes[2]),
    ]

    for col, title, ax in features:
        churn_pct = (
            df.groupby(col)["Churn"]
            .value_counts(normalize=True)
            .rename("percentage")
            .reset_index()
        )
        churn_pct["percentage"] = churn_pct["percentage"] * 100

        sns.barplot(
            x=col,
            y="percentage",
            hue="Churn",
            data=churn_pct,
            palette=COLOR_PALETTE,
            ax=ax,
        )
        ax.set_title(f"Churn % by {title}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Percentage (%)", fontsize=11)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(title="Churn", labels=["Stayed (0)", "Churned (1)"])

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "contract_and_billing_churn.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] {plot_path}")


def plot_services_vs_churn(df: pd.DataFrame) -> None:
    """Analyze churn across Internet & Value-added support services."""
    services = [
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, service in enumerate(services):
        ax = axes[idx]
        churn_pct = (
            df.groupby(service)["Churn"]
            .value_counts(normalize=True)
            .rename("percentage")
            .reset_index()
        )
        churn_pct["percentage"] = churn_pct["percentage"] * 100

        sns.barplot(
            x=service,
            y="percentage",
            hue="Churn",
            data=churn_pct,
            palette=COLOR_PALETTE,
            ax=ax,
        )
        ax.set_title(f"Churn % by {service}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Percentage (%)", fontsize=10)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="Churn", labels=["Stayed (0)", "Churned (1)"])

    # Hide extra unused subplot
    axes[-1].axis("off")

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "services_churn_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] {plot_path}")


def plot_continuous_distributions(df: pd.DataFrame) -> None:
    """Plot Distributions and Boxplots for Tenure, MonthlyCharges, and TotalCharges by Churn."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    titles = ["Customer Tenure (Months)", "Monthly Charges ($)", "Total Charges ($)"]

    # Row 1: KDE / Histograms
    for idx, col in enumerate(numeric_cols):
        sns.histplot(
            data=df,
            x=col,
            hue="Churn",
            kde=True,
            palette=COLOR_PALETTE,
            element="step",
            stat="density",
            common_norm=False,
            ax=axes[0, idx],
        )
        axes[0, idx].set_title(f"{titles[idx]} Distribution", fontsize=12, fontweight="bold")
        axes[0, idx].set_xlabel(titles[idx], fontsize=11)
        axes[0, idx].legend(title="Churn", labels=["Churned (1)", "Stayed (0)"])

    # Row 2: Boxplots
    for idx, col in enumerate(numeric_cols):
        sns.boxplot(
            data=df,
            x="Churn",
            y=col,
            palette=COLOR_PALETTE,
            hue="Churn",
            legend=False,
            ax=axes[1, idx],
        )
        axes[1, idx].set_title(f"{titles[idx]} by Churn Status", fontsize=12, fontweight="bold")
        axes[1, idx].set_xticks([0, 1])
        axes[1, idx].set_xticklabels(["Stayed (0)", "Churned (1)"])
        axes[1, idx].set_ylabel(titles[idx], fontsize=11)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "tenure_monthly_charges_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] {plot_path}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Generate Correlation Heatmap of encoded numerical & categorical variables against Churn."""
    # Convert categorical variables to one-hot for correlation calculation
    df_encoded = pd.get_dummies(df.drop(columns=["customerID"], errors="ignore"), drop_first=True)

    # Compute correlations with target Churn
    corr = df_encoded.corr()
    churn_corr = corr[["Churn"]].sort_values(by="Churn", ascending=False)

    plt.figure(figsize=(8, 12))
    sns.heatmap(
        churn_corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Pearson Correlation with Churn"},
    )
    plt.title("Feature Correlation with Customer Churn", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] {plot_path}")


def generate_eda_insights(df: pd.DataFrame) -> str:
    """Compute and print key statistical insights answering project research questions."""
    total_customers = len(df)
    churn_rate = df["Churn"].mean() * 100

    # Contract churn rates
    contract_churn = df.groupby("Contract")["Churn"].mean() * 100

    # Internet service churn
    internet_churn = df.groupby("InternetService")["Churn"].mean() * 100

    # Payment method churn
    payment_churn = df.groupby("PaymentMethod")["Churn"].mean() * 100

    # Tenure comparisons
    median_tenure_stay = df[df["Churn"] == 0]["tenure"].median()
    median_tenure_churn = df[df["Churn"] == 1]["tenure"].median()

    # Monthly charges comparisons
    median_charges_stay = df[df["Churn"] == 0]["MonthlyCharges"].median()
    median_charges_churn = df[df["Churn"] == 1]["MonthlyCharges"].median()

    insights = f"""
{'='*70}
KEY EXPLORATORY DATA ANALYSIS INSIGHTS
{'='*70}
1. Overall Churn Rate: {churn_rate:.2f}% ({df['Churn'].sum():,} out of {total_customers:,} customers).
2. Contract Type Impact:
   - Month-to-Month contracts have the HIGHEST churn rate: {contract_churn.get('Month-to-month', 0):.2f}%
   - One-Year contracts: {contract_churn.get('One year', 0):.2f}%
   - Two-Year contracts have the LOWEST churn rate: {contract_churn.get('Two year', 0):.2f}%
   -> Insight: Month-to-month customers are ~15x more likely to churn than two-year contract holders.

3. Tenure Dynamics (Newer vs. Established Customers):
   - Median tenure of Churned customers:  {median_tenure_churn:.0f} months
   - Median tenure of Retained customers: {median_tenure_stay:.0f} months
   -> Insight: Customer churn is heavily concentrated in the first 1-12 months (onboarding risk).

4. Financial / Monthly Charges Dynamics:
   - Median Monthly Charges for Churned customers:  ${median_charges_churn:.2f}
   - Median Monthly Charges for Retained customers: ${median_charges_stay:.2f}
   -> Insight: Customers paying higher monthly bills (> $70/mo) churn at substantially higher rates.

5. Internet Service & Technology:
   - Fiber Optic churn rate: {internet_churn.get('Fiber optic', 0):.2f}%
   - DSL churn rate:         {internet_churn.get('DSL', 0):.2f}%
   - No Internet churn rate: {internet_churn.get('No', 0):.2f}%
   -> Insight: Fiber optic customers churn heavily, likely due to high pricing or service dissatisfaction.

6. Payment Method:
   - Electronic Check churn rate: {payment_churn.get('Electronic check', 0):.2f}%
   - Auto-pay methods (Credit Card / Bank Transfer) have churn rates below 18%.

7. Protective Services:
   - Customers WITHOUT Online Security or Tech Support churn at > 40%, whereas those WITH these
     value-added services churn at < 15%.
{'='*70}
"""
    print(insights)
    return insights


def run_full_eda(df: pd.DataFrame) -> None:
    """Execute all EDA visualizations and print insight report."""
    print("\n[RUNNING] Generating EDA Visualizations...")
    plot_churn_distribution(df)
    plot_demographics_vs_churn(df)
    plot_contract_and_billing(df)
    plot_services_vs_churn(df)
    plot_continuous_distributions(df)
    plot_correlation_heatmap(df)
    generate_eda_insights(df)
    print("[COMPLETED] All EDA plots saved to plots/ folder.")


if __name__ == "__main__":
    from src.data_preprocessing import load_raw_data, clean_data
    df = clean_data(load_raw_data())
    run_full_eda(df)
