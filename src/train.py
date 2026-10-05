"""Train and compare multiple regression models for AQI prediction."""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import PROCESSED_DATA_DIR, FEATURE_COLUMNS, TARGET_COLUMN

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def load_dataset():
    df = pd.read_csv(PROCESSED_DATA_DIR / "processed_dataset.csv")
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def build_candidates():
    """Build pipelines so imputation and scaling are fitted on training folds."""
    def pipeline(model):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])

    return {
        "Linear Regression": pipeline(LinearRegression()),
        "Decision Tree": pipeline(DecisionTreeRegressor(random_state=RANDOM_STATE)),
        "Random Forest": pipeline(RandomForestRegressor(random_state=RANDOM_STATE)),
        "Gradient Boosting": pipeline(GradientBoostingRegressor(random_state=RANDOM_STATE)),
    }


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def run_training():
    X, y = load_dataset()
    # Independent-observation regression (no reliable date column exists to
    # justify chronological splitting -- see project notes). Random split
    # with a fixed seed for reproducibility.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = build_candidates()
    fitted = {}
    cv_scores = {}

    for name, pipe in candidates.items():
        cv_scores[name] = float(cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=1
        ).mean())
        pipe.fit(X_train, y_train)
        fitted[name] = pipe

    rf_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 8, 12],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }
    gb_grid = {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3],
        "model__min_samples_split": [2, 5],
    }
    for name, grid in [("Random Forest", rf_grid), ("Gradient Boosting", gb_grid)]:
        search = GridSearchCV(
            build_candidates()[name], grid, cv=cv, scoring="r2", n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        tuned_name = f"{name} (tuned)"
        fitted[tuned_name] = search.best_estimator_
        cv_scores[tuned_name] = float(search.best_score_)
        print(f"[tune] {tuned_name} CV R2={search.best_score_:.4f}; params={search.best_params_}")

    # The untouched test split is not consulted until this CV-only choice is made.
    best_name = max(cv_scores, key=cv_scores.get)
    best_model = fitted[best_name]
    print(f"\n[run_training] Selected by mean 5-fold training CV R2: {best_name}")

    results = {}
    for name, model in fitted.items():
        results[name] = {"CV_R2": cv_scores[name], **evaluate(y_test, model.predict(X_test))}

    comp_df = pd.DataFrame(results).T
    comp_df.index.name = "Model"
    comp_df.to_csv(MODELS_DIR / "model_comparison.csv")
    print("\n=== Final test-set comparison (selection used CV_R2 only) ===")
    print(comp_df[["MAE", "MSE", "RMSE", "R2", "CV_R2"]].to_string(float_format=lambda value: f"{value:.4f}"))

    # Feature importance (tree-based models only)
    feat_importance = None
    inner_model = best_model.named_steps["model"]
    if hasattr(inner_model, "feature_importances_"):
        feat_importance = pd.Series(
            inner_model.feature_importances_, index=FEATURE_COLUMNS
        ).sort_values(ascending=False)
        feat_importance.to_csv(MODELS_DIR / "feature_importance.csv", header=["importance"])
        print("\n=== Feature importance (best model) ===")
        print(feat_importance)
    else:
        pd.DataFrame({"importance": np.nan}, index=FEATURE_COLUMNS).to_csv(
            MODELS_DIR / "feature_importance.csv"
        )

    # Save the best pipeline (scaler + model together)
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    with open(MODELS_DIR / "best_model_name.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "selection_metric": "mean 5-fold cross-validation R2 on training split",
            "cv_r2": cv_scores[best_name],
        }, f, indent=2)

    print(f"\n[run_training] Saved best model ({best_name}) to models/best_model.pkl")
    return results, best_name


if __name__ == "__main__":
    run_training()
