"""
train_models.py

Trains and cross-validates three regression models for predicting
shipment delivery time: Linear Regression, Decision Tree, and Random
Forest. Saves the best pipeline (preprocessing + model) to disk.

Run:
    python src/train_models.py
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from preprocessing import build_preprocessor, load_and_clean, split_features_target

RANDOM_SEED = 42


def get_models() -> dict:
    return {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_SEED),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1
        ),
    }


def train_and_select(csv_path: str = "data/logistics_shipments.csv"):
    df = load_and_clean(csv_path)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    preprocessor = build_preprocessor()
    results = {}

    for name, model in get_models().items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])

        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error"
        )
        cv_rmse = -cv_scores.mean()

        pipe.fit(X_train, y_train)

        results[name] = {"pipeline": pipe, "cv_rmse": cv_rmse}
        print(f"{name:>18s}  |  5-fold CV RMSE = {cv_rmse:.3f} hours")

    best_name = min(results, key=lambda k: results[k]["cv_rmse"])
    best_pipeline = results[best_name]["pipeline"]
    print(f"\nBest model by CV RMSE: {best_name}")

    joblib.dump(best_pipeline, "outputs/best_model.joblib")
    print("Saved best pipeline to outputs/best_model.joblib")

    return results, (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    train_and_select()
