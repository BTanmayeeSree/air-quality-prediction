# Presentation Content (15 slides)

1. **Title** — Air Quality Prediction Using Machine Learning
2. **Technical Description** — Machine Learning-Based AQI Estimation from Weather Parameters; supervised regression
3. **Problem Statement** — Estimate a PM2.5-derived CPCB AQI sub-index from eight weather parameters; not future forecasting
4. **Objectives** — Clean data, derive target, perform EDA, select through training-only CV, deploy saved pipeline
5. **Architecture** — Raw CSVs → preprocessing → EDA → training/CV → saved Joblib pipeline → prediction module → Streamlit
6. **Dataset** — 731 source rows: 2013 (185), 2014 (183), 2015 (182), 2016 (181); eight weather fields and PM2.5; no reliable dates or other pollutants
7. **Data Quality** — No exact duplicates; one missing and 22 exact-zero PM2.5 targets excluded; zero validity cannot be verified without sensor metadata; 708 usable targets
8. **AQI Method** — CPCB PM2.5 interpolation, including 251–500 µg/m³ → AQI 401–500; concentrations above 500 capped at AQI 500
9. **EDA** — PM2.5/AQI and feature distributions, category counts, correlations, feature relationships, AQI Distribution by Source Year
10. **Regressors** — Linear Regression, Decision Tree, Random Forest, Gradient Boosting; tune Random Forest and Gradient Boosting using GridSearchCV
11. **Evaluation Design** — 80/20 shuffled split, seed 42; 566 training and 142 final test rows; selection by 5-fold CV R² on training only
12. **Actual Test Metrics** — Model | MAE | MSE | RMSE | R²:

    | Model | MAE | MSE | RMSE | R² |
    |---|---:|---:|---:|---:|
    | Linear Regression | 59.0879 | 6257.8850 | 79.1068 | 0.6103 |
    | Decision Tree | 70.2762 | 10269.0720 | 101.3364 | 0.3605 |
    | Random Forest | 56.1335 | 5830.5083 | 76.3578 | 0.6369 |
    | Gradient Boosting | 56.3944 | 6074.2472 | 77.9375 | 0.6218 |
    | Random Forest (tuned) | 55.7229 | 5661.8129 | 75.2450 | 0.6474 |
    | Gradient Boosting (tuned) | 55.7632 | 5857.5916 | 76.5349 | 0.6352 |

13. **Selected Model** — Tuned Random Forest, training CV R² 0.6826; untouched-test MAE 55.7229, RMSE 75.2450, R² 0.6474
14. **Feature Importance and Limitations** — T 0.352031, VV 0.168224, Tm 0.137379, SLP 0.128155 lead; one city, one pollutant, weather-only inputs, no reliable dates
15. **Conclusion and Next Steps** — Transparent B.Tech regression system; future improvement requires quality-flagged dated observations and additional pollutant data
