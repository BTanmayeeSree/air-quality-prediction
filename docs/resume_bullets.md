# Resume Bullet Points

- Built a supervised regression pipeline for PM2.5-derived CPCB AQI estimation from eight weather parameters across 708 usable Delhi observations; investigated 22 unverified zero-target records and preserved raw data.
- Compared Linear Regression, Decision Tree, Random Forest, and Gradient Boosting; tuned ensemble models using training-only 5-fold CV and selected tuned Random Forest by CV R² (0.6826), not test performance.
- Evaluated the selected model on an untouched 142-row test split (MAE 55.7229, RMSE 75.2450, R² 0.6474) and deployed its Joblib pipeline in a validated Streamlit dashboard.
