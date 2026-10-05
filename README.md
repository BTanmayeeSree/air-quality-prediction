# Air Quality Prediction Using Machine Learning

## Abstract

This B.Tech project implements **Machine Learning-Based AQI Estimation from Weather Parameters** as supervised regression. It uses four Delhi weather/PM2.5 CSV files (2013–2016), derives a PM2.5-based AQI target, compares four regressors, tunes Random Forest and Gradient Boosting using training-only cross-validation, and deploys the selected Joblib pipeline in Streamlit. The CV-selected Random Forest (tuned) achieved test MAE 55.7229, RMSE 75.2450, and R² 0.6474 on an untouched 20% split. There are no reliable dates or other pollutant measurements, so this is not future forecasting or a complete multi-pollutant AQI system.

## Problem Statement

Estimate a CPCB PM2.5-derived AQI sub-index from eight weather parameters using historical Delhi observations. The output is an educational estimate, not an official real-time CPCB measurement.

## Objectives

- Clean and validate the supplied CSV observations without fabricating dates.
- Derive an AQI target and category from PM2.5.
- Explore distributions, correlations, and feature/target relationships.
- Compare Linear Regression, Decision Tree, Random Forest, and Gradient Boosting.
- Tune candidate ensembles using cross-validation within the training split.
- Evaluate the selected model once on an untouched test split and deploy the full pipeline.

## Dataset

The four files contain 731 observations: 2013 (185), 2014 (183), 2015 (182), and 2016 (181). Columns are `T`, `TM`, `Tm`, `SLP`, `H`, `VV`, `V`, `VM`, and `PM 2.5`. There is no reliable date column and no precomputed AQI. Source year is retained only for grouping.

There are zero exact duplicate records, one missing PM2.5 target, and 22 exact-zero PM2.5 values. The zero readings cannot be verified without sensor metadata. Because exact-zero ambient Delhi PM2.5 is implausible, they are treated as unverified placeholders and excluded from supervised training; raw CSVs are unchanged. This is an assumption, not a confirmed sensor diagnosis. After excluding these 23 target rows, 708 observations remain. Missing or physically invalid weather values are imputed inside each training fold. Valid high-pollution observations are retained.

## AQI Methodology

PM2.5 is mapped using CPCB's sub-index linear interpolation table: 0–30 µg/m³ → AQI 0–50; 31–60 → 51–100; 61–90 → 101–200; 91–120 → 201–300; 121–250 → 301–400; and 251–500 → 401–500. Concentrations are rounded to the nearest whole µg/m³ to apply the integer breakpoints. Concentrations above 500 are capped at AQI 500; AQI is never extrapolated above the 0–500 scale. Categories are Good (0–50), Satisfactory (51–100), Moderate (101–200), Poor (201–300), Very Poor (301–400), and Severe (401–500).

## Preprocessing and Feature Engineering

Numeric values are coerced consistently; infinities and physically implausible weather measurements become missing. Rows with missing, negative, or exact-zero PM2.5 are excluded because a trustworthy target cannot be derived. Weather fields are median-imputed and standardized inside an sklearn Pipeline, fitted separately within CV folds. No outlier filtering is applied to valid high-pollution values. The eight weather columns are the model inputs; no dates, lags, rolling features, or future-target features are created.

## EDA

`src/eda.py` produces PM2.5 and AQI distributions, distributions for all weather features, AQI category counts, feature-vs-AQI plots, a correlation heatmap, and **AQI Distribution by Source Year**. It does not generate an AQI-over-time chart.

## ML Algorithms and Evaluation

Linear Regression is the baseline; Decision Tree, Random Forest, and Gradient Boosting are the other regressors. An 80/20 shuffled split with seed 42 gives 566 training and 142 test observations. Models are selected by mean 5-fold CV R² using only training rows. GridSearchCV tunes Random Forest (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`) and Gradient Boosting (`n_estimators`, `learning_rate`, `max_depth`, `min_samples_split`). The test split is not used for tuning or selection. The following actual metrics are computed afterward on that test split:

| Model | MAE | MSE | RMSE | R² | Training CV R² |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 59.0879 | 6257.8850 | 79.1068 | 0.6103 | 0.6159 |
| Decision Tree | 70.2762 | 10269.0720 | 101.3364 | 0.3605 | 0.3318 |
| Random Forest | 56.1335 | 5830.5083 | 76.3578 | 0.6369 | 0.6771 |
| Gradient Boosting | 56.3944 | 6074.2472 | 77.9375 | 0.6218 | 0.6721 |
| **Random Forest (tuned)** | **55.7229** | **5661.8129** | **75.2450** | **0.6474** | **0.6826** |
| Gradient Boosting (tuned) | 55.7632 | 5857.5916 | 76.5349 | 0.6352 | 0.6771 |

The selected model is **Random Forest (tuned)**, chosen by the highest training CV R² (0.6826), not by test-set performance. Its final test metrics are MAE 55.7229, MSE 5661.8129, RMSE 75.2450, and R² 0.6474.

### Feature Importance

The saved tuned Random Forest reports these generated impurity-based importances:

| Feature | Importance |
|---|---:|
| T | 0.352031 |
| VV | 0.168224 |
| Tm | 0.137379 |
| SLP | 0.128155 |
| H | 0.112326 |
| V | 0.044275 |
| TM | 0.040342 |
| VM | 0.017268 |

Importance indicates model use, not causation.

## Architecture

```text
Raw yearly CSVs
    -> preprocessing.py (clean, derive AQI)
    -> eda.py (statistics and charts)
    -> train.py (training-only CV, tuning, final test evaluation)
    -> models/best_model.pkl (imputer + scaler + selected regressor)
    -> predict.py (validation and prediction)
    -> app/app.py (Streamlit interface)
```

## Technologies

Python, pandas, NumPy, scikit-learn, Joblib, Matplotlib, Seaborn, and Streamlit.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Rebuild Data, EDA, and Model

From the project root:

```bash
cd src
python preprocessing.py
python eda.py
python train.py
```

## Run Streamlit

From the project root in a separate terminal:

```bash
streamlit run app/app.py
```

The app loads `models/best_model.pkl`; it does not retrain at startup. It provides Home, Predict, Dashboard, Model Comparison, and About views. Prediction inputs are checked for sensible bounds and consistent maximum/mean values.

## Limitations

- Weather-only inputs omit traffic, construction, crop burning, and other pollution sources.
- The target is a single-pollutant PM2.5-derived sub-index, not the complete multi-pollutant national AQI.
- The exact-zero exclusion is an assumption because sensor metadata is unavailable.
- Data covers one city and four source years and has no reliable daily dates; it supports neither chronological evaluation nor future forecasting.
- The model is an educational estimate, not an official real-time CPCB reading or medical guidance.

## Future Scope

Use dated, quality-flagged measurements; add other pollutants and cities; and only then consider a properly defined forecasting task or external live-data integration.
