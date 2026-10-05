# Academic Report — Air Quality Prediction Using Machine Learning

## Chapter 1 — Introduction
Air pollution is a major public health issue in cities like Delhi, where
particulate matter (PM2.5) regularly reaches hazardous levels. The Air
Quality Index (AQI) condenses pollutant concentrations into a single,
easy-to-understand number and category. This project builds a machine
learning system that estimates AQI from weather conditions, using
historical Delhi weather + PM2.5 data (2013–2016). Its technical description
is **Machine Learning-Based AQI Estimation from Weather Parameters**.

## Chapter 2 — Literature Survey
AQI prediction is commonly approached with regression models (Linear
Regression, Random Forest, Gradient Boosting) using pollutant
concentrations and/or weather variables as predictors. Weather-only
models are less common than multi-pollutant models, but are useful when
only meteorological sensors (not full pollutant monitoring) are
available, since temperature, humidity, and wind strongly influence how
pollutants disperse or accumulate.

## Chapter 3 — Problem Statement
Given eight weather measurements, estimate a PM2.5-derived CPCB AQI
sub-index. The input dataset has no reliable dates, so this is not a
future-forecasting system.

## Chapter 4 — Objectives
1. Build a clean, reproducible data pipeline from raw CSVs to a trained
   model.
2. Compare multiple regression algorithms fairly, using real metrics.
3. Select and deploy the best-performing model.
4. Present predictions through an accessible web interface.

## Chapter 5 — Existing System
Manual AQI reporting relies on physical pollutant sensors (PM2.5, PM10,
NO2, SO2, CO, O3 monitors) maintained by pollution-control boards. This
requires expensive, well-maintained monitoring infrastructure that isn't
available everywhere.

## Chapter 6 — Proposed System
A machine learning model trained on historical weather data predicts AQI
without needing live pollutant sensors, using only commonly available
weather measurements (temperature, pressure, humidity, visibility, wind).

## Chapter 7 — Methodology
After cleaning and deriving the target, data was split 80/20. The four
models and both GridSearchCV searches used only the training partition,
with mean 5-fold CV R² for selection. The untouched test partition was
used once for final MAE, MSE, RMSE, and R² evaluation.

## Chapter 8 — Dataset Description
731 records (2013–2016), 8 weather features
(T, TM, Tm, SLP, H, VV, V, VM) plus PM2.5 concentration. No AQI or other
pollutant columns were present in the raw data.

## Chapter 9 — Data Preprocessing
- Removed 0 duplicate rows (none found).
- Dropped 1 row with a missing PM2.5 target.
- Found 22 exact-zero PM2.5 values. Since sensor metadata is unavailable,
  they cannot be verified; this project treats them as unverified
  placeholders and excludes them, without modifying raw CSVs.
- Validated weather inputs; physically implausible values are imputed
  within each training fold. Valid high-pollution records are retained.
- Derived `AQI` using CPCB PM2.5 bands (including 251–500 µg/m³ →
  AQI 401–500); concentrations above 500 are capped at AQI 500.
- Assigned categories across continuous AQI boundaries, avoiding gaps
  for fractional model estimates.
- Scaled features with `StandardScaler` inside each model pipeline.

## Chapter 10 — Exploratory Data Analysis
Distributions were plotted for AQI, PM2.5, and all 8 weather features.
A correlation heatmap and feature-vs-AQI scatterplots were generated
(see `visualizations/`). Mean temperature (T) and visibility (VV) showed
the strongest relationships with AQI, consistent with the feature
importances found later.

## Chapter 11 — Machine Learning Models
Linear Regression, Decision Tree, Random Forest, and Gradient Boosting
regressors were trained, each inside a `StandardScaler → model`
pipeline for reproducibility.

## Chapter 12 — Model Evaluation
The model was selected by training CV R², not test performance. Final
test metrics on 142 held-out rows are:

| Model | MAE | MSE | RMSE | R² | CV R² |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 59.0879 | 6257.8850 | 79.1068 | 0.6103 | 0.6159 |
| Decision Tree | 70.2762 | 10269.0720 | 101.3364 | 0.3605 | 0.3318 |
| Random Forest | 56.1335 | 5830.5083 | 76.3578 | 0.6369 | 0.6771 |
| Gradient Boosting | 56.3944 | 6074.2472 | 77.9375 | 0.6218 | 0.6721 |
| Random Forest (tuned) | 55.7229 | 5661.8129 | 75.2450 | 0.6474 | 0.6826 |
| Gradient Boosting (tuned) | 55.7632 | 5857.5916 | 76.5349 | 0.6352 | 0.6771 |

## Chapter 13 — System Architecture
Raw CSVs → preprocessing.py → eda.py → train.py → models/best_model.pkl
→ predict.py → Streamlit app (app/app.py).

## Chapter 14 — Implementation
Implemented in Python using pandas, scikit-learn, Matplotlib/Seaborn,
Joblib, and Streamlit. Full source is in `src/` and `app/`.

## Chapter 15 — Results
Random Forest (tuned) was selected by highest training CV R² (0.6826).
Its untouched test metrics are MAE 55.7229, MSE 5661.8129, RMSE
75.2450, and R² 0.6474. Generated feature importances are T 0.352031,
VV 0.168224, Tm 0.137379, SLP 0.128155, H 0.112326, V 0.044275,
TM 0.040342, and VM 0.017268.

## Chapter 16 — Limitations
Weather-only inputs omit many pollution sources; the target is only a
PM2.5-derived sub-index; data covers one city and has no reliable date.
The exact-zero exclusion is uncertain because sensor-quality metadata is
not supplied.

## Chapter 17 — Future Scope
Live API integration, multi-pollutant AQI, true time-series forecasting
with dated data, multi-city support, automated retraining.

## Chapter 18 — Conclusion
This project demonstrates a complete, honest, end-to-end ML workflow —
from raw data with real limitations (no AQI column, no dates, single
pollutant) through to a deployed prediction interface — while being
transparent about what the model can and cannot explain.

## Chapter 19 — References
- CPCB National Air Quality Index methodology (Central Pollution Control
  Board, India).
- scikit-learn documentation (https://scikit-learn.org).
- Streamlit documentation (https://docs.streamlit.io).
