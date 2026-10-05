"""
Streamlit application for Air Quality (AQI) Prediction Using Machine Learning.

Run with:  streamlit run app/app.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# Make src/ importable regardless of working directory
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocessing import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402
from predict import predict_aqi  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

st.set_page_config(page_title="AQI Prediction", page_icon="🌫️", layout="wide")


@st.cache_data
def get_comparison_table():
    return pd.read_csv(MODELS_DIR / "model_comparison.csv", index_col=0)


@st.cache_data
def get_feature_importance():
    path = MODELS_DIR / "feature_importance.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return None


@st.cache_data
def get_model_info():
    path = MODELS_DIR / "best_model_name.json"
    if path.exists():
        import json
        with path.open(encoding="utf-8") as model_info_file:
            return json.load(model_info_file)
    return {"best_model": "Unavailable"}


@st.cache_data
def get_dataset():
    return pd.read_csv(DATA_DIR / "processed_dataset.csv")


CATEGORY_COLORS = {
    "Good": "#4caf50",
    "Satisfactory": "#8bc34a",
    "Moderate": "#ffc107",
    "Poor": "#ff9800",
    "Very Poor": "#f4511e",
    "Severe": "#b71c1c",
}

PAGES = ["Home", "Predict", "Dashboard", "Model Comparison", "About"]


def sidebar_nav():
    st.sidebar.title("🌫️ AQI Prediction")
    return st.sidebar.radio("Navigate", PAGES)


def page_home():
    st.title("🌫️ Air Quality Prediction Using Machine Learning")
    st.subheader("Machine Learning-Based AQI Estimation from Weather Parameters")
    st.write(
        "A supervised regression project that estimates a PM2.5-derived "
        "CPCB AQI sub-index from eight weather parameters using Delhi "
        "observations from source files for 2013–2016."
    )
    st.info(
        "This is not future forecasting or an official real-time measurement. "
        "The dataset has no reliable dates and contains only PM2.5 as a "
        "pollutant; the target is a single-pollutant estimate."
    )
    df = get_dataset()
    col1, col2, col3 = st.columns(3)
    col1.metric("Supervised observations", len(df))
    col2.metric("Weather features", len(FEATURE_COLUMNS))
    col3.metric("Mean derived AQI", f"{df[TARGET_COLUMN].mean():.1f}")
    st.caption("Source: four Delhi weather/PM2.5 CSV files, 2013–2016. One missing and 22 unverified zero PM2.5 targets were excluded.")


def page_predict():
    st.title("Predict AQI")
    st.write("Enter weather parameters for a model-based AQI estimate.")

    col1, col2 = st.columns(2)
    with col1:
        T = st.number_input("Mean Temperature (T, °C)", min_value=-50.0, max_value=60.0, value=20.0, step=0.1)
        TM = st.number_input("Max Temperature (TM, °C)", min_value=-50.0, max_value=70.0, value=25.0, step=0.1)
        Tm = st.number_input("Min Temperature (Tm, °C)", min_value=-90.0, max_value=50.0, value=15.0, step=0.1)
        SLP = st.number_input("Sea Level Pressure (SLP, hPa)", min_value=800.0, max_value=1100.0, value=1015.0, step=0.1)
    with col2:
        H = st.number_input("Humidity (H, %)", min_value=0, max_value=100, value=70)
        VV = st.number_input("Visibility (VV, km)", min_value=0.0, max_value=100.0, value=1.5, step=0.1)
        V = st.number_input("Mean Wind Speed (V)", min_value=0.0, max_value=200.0, value=5.0, step=0.1)
        VM = st.number_input("Max Wind Speed (VM)", min_value=0.0, max_value=200.0, value=12.0, step=0.1)

    if st.button("Predict AQI", type="primary"):
        try:
            prediction = predict_aqi({
                "T": T, "TM": TM, "Tm": Tm, "SLP": SLP,
                "H": H, "VV": VV, "V": V, "VM": VM,
            })
        except ValueError as error:
            st.error(str(error))
        else:
            st.subheader("Result")
            result_col, category_col, model_col = st.columns(3)
            result_col.metric("Predicted AQI", f"{prediction['aqi']:.0f}")
            category_col.metric("Category", prediction["category"])
            model_col.metric("Model", prediction["model"])
            st.caption("This is a machine-learning estimate based on historical Delhi weather and PM2.5 data. It is not an official real-time CPCB AQI measurement.")


def page_dashboard():
    st.title("Dashboard")
    df = get_dataset()

    metric_columns = st.columns(4)
    metric_columns[0].metric("Observations", len(df))
    metric_columns[1].metric("Features", len(FEATURE_COLUMNS))
    metric_columns[2].metric("AQI range", f"{df[TARGET_COLUMN].min():.1f}–{df[TARGET_COLUMN].max():.0f}")
    metric_columns[3].metric("Average AQI", f"{df[TARGET_COLUMN].mean():.1f}")

    category_order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("AQI Category Distribution")
        st.bar_chart(df["AQI_Category"].value_counts().reindex(category_order, fill_value=0))
    with col2:
        st.subheader("AQI Distribution by Source Year")
        fig, axis = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df, x="year", y=TARGET_COLUMN, color="skyblue", ax=axis)
        axis.set_xlabel("Source year")
        axis.set_ylabel("AQI")
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    st.subheader("Weather Feature Distributions")
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for axis, feature in zip(axes.flat, FEATURE_COLUMNS):
        sns.histplot(df[feature], ax=axis, color="teal")
        axis.set_title(feature)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.subheader("Correlation Heatmap")
    fig, axis = plt.subplots(figsize=(10, 7))
    sns.heatmap(df[FEATURE_COLUMNS + ["PM 2.5", TARGET_COLUMN]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axis)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    fi = get_feature_importance()
    if fi is not None and fi["importance"].notna().any():
        st.subheader("Feature importance (best model)")
        st.bar_chart(fi["importance"].dropna().sort_values())


def page_model_comparison():
    st.title("Model Comparison")
    st.write("Models are selected using mean 5-fold cross-validation R² within the 80% training split. The 20% test split is used only for final evaluation.")
    comp = get_comparison_table()
    st.dataframe(comp.style.format("{:.2f}"))
    selected = get_model_info().get("best_model", "Unavailable")
    st.success(f"Selected model: **{selected}** (highest mean training cross-validation R²)")


def page_about():
    st.title("About this project")
    st.markdown(
        """
**Technical description:** Machine Learning-Based AQI Estimation from Weather Parameters. This is supervised regression.

**Dataset:** Four yearly files contain 731 weather and PM2.5 observations from 2013–2016. The raw files have no reliable dates or AQI column.

**Target:** PM2.5-derived CPCB sub-index, capped at 500. The open-ended severe concentration band above 250 µg/m³ is saturated at the AQI scale maximum rather than extrapolated beyond CPCB's 0–500 scale.

**Model selection:** Linear Regression, Decision Tree, Random Forest, and Gradient Boosting; Random Forest and Gradient Boosting are tuned by 5-fold CV using training data only. Final test results are reported separately.

**Limitations:**
- Weather-only inputs omit traffic, construction dust, crop burning, and other pollution sources.
- Only 4 years of data from a single city; results won't transfer directly
  to other locations.
- AQI here reflects only PM2.5; other pollutants (PM10, NO2, SO2, CO, O3)
  are not represented.
- This tool is for educational purposes and is not an official regulatory
  or medical air-quality source.
- Exact-zero PM2.5 appears 22 times. Without sensor/source metadata these readings cannot be verified; they are treated as unverified placeholders and excluded from supervised training.

**Future scope:** integrate a live weather/pollution API, add other
pollutants, use true multi-year timestamps for real forecasting.
        """
    )


def main():
    page = sidebar_nav()
    if page == "Home":
        page_home()
    elif page == "Predict":
        page_predict()
    elif page == "Dashboard":
        page_dashboard()
    elif page == "Model Comparison":
        page_model_comparison()
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
