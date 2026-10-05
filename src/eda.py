"""Exploratory Data Analysis for the processed Delhi AQI dataset."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

from preprocessing import PROCESSED_DATA_DIR, FEATURE_COLUMNS, TARGET_COLUMN, PM25_COLUMN

VIZ_DIR = Path(__file__).resolve().parents[1] / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")


def run_eda():
    df = pd.read_csv(PROCESSED_DATA_DIR / "processed_dataset.csv")

    print("=== Dataset overview ===")
    print("Rows:", df.shape[0], "| Columns:", df.shape[1])
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nStatistical summary:\n", df[FEATURE_COLUMNS + [PM25_COLUMN, TARGET_COLUMN]].describe())
    print("\nAQI category counts:\n", df["AQI_Category"].value_counts())

    # 1. AQI distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(df[TARGET_COLUMN], kde=True, color="steelblue")
    plt.title("Distribution of AQI (CPCB PM2.5 sub-index)")
    plt.xlabel("AQI")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "aqi_distribution.png", dpi=120)
    plt.close()

    # 2. PM2.5 distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(df[PM25_COLUMN], kde=True, color="indianred")
    plt.title("Distribution of PM2.5 Concentration")
    plt.xlabel("PM2.5 (ug/m3)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "pm25_distribution.png", dpi=120)
    plt.close()

    # 3. Weather feature distributions (grid)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flatten(), FEATURE_COLUMNS):
        sns.histplot(df[col], kde=True, ax=ax, color="seagreen")
        ax.set_title(f"{col} distribution")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "feature_distributions.png", dpi=120)
    plt.close()

    # 4. AQI distribution grouped by source year; no dates are inferred.
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="year", y=TARGET_COLUMN, color="skyblue")
    plt.title("AQI Distribution by Source Year")
    plt.xlabel("Source year")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "aqi_by_source_year.png", dpi=120)
    plt.close()

    # 5. AQI category distribution
    plt.figure(figsize=(8, 4))
    category_order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    sns.countplot(data=df, x="AQI_Category", order=category_order, color="steelblue")
    plt.title("AQI Category Distribution")
    plt.xlabel("CPCB category")
    plt.ylabel("Observations")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "aqi_category_distribution.png", dpi=120)
    plt.close()

    # 6. Correlation heatmap
    plt.figure(figsize=(8, 6))
    corr = df[FEATURE_COLUMNS + [PM25_COLUMN, TARGET_COLUMN]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "correlation_heatmap.png", dpi=120)
    plt.close()

    # 7. Feature vs AQI relationships
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flatten(), FEATURE_COLUMNS):
        sns.scatterplot(x=df[col], y=df[TARGET_COLUMN], ax=ax, alpha=0.5, color="darkorange")
        ax.set_title(f"{col} vs AQI")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "features_vs_aqi.png", dpi=120)
    plt.close()

    print(f"\n[run_eda] charts written to {VIZ_DIR}")
    return corr


if __name__ == "__main__":
    run_eda()
