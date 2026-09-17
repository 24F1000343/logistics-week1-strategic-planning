"""
evaluate.py

Evaluates the trained models on the held-out test set using RMSE, MAE
and R^2, prints a comparison table, and reports feature importances
for the random forest model.

Run:
    python src/evaluate.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import CATEGORICAL_COLS, NUMERIC_COLS
from train_models import train_and_select


def evaluate_all(results: dict, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        pipe = r["pipeline"]
        preds = pipe.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        rows.append(
            {
                "model": name,
                "test_rmse_hours": round(rmse, 3),
                "test_mae_hours": round(mae, 3),
                "r2": round(r2, 3),
                "cv_rmse_hours": round(r["cv_rmse"], 3),
            }
        )
    return pd.DataFrame(rows).sort_values("test_rmse_hours")


def feature_importance_report(pipeline) -> pd.DataFrame:
    """Only meaningful for tree-based models with feature_importances_."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()

    ohe = pipeline.named_steps["prep"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
    all_feature_names = NUMERIC_COLS + cat_feature_names

    importances = pd.DataFrame(
        {"feature": all_feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return importances


if __name__ == "__main__":
    results, (X_train, X_test, y_train, y_test) = train_and_select()

    comparison = evaluate_all(results, X_test, y_test)
    print("\n=== Model comparison on held-out test set ===")
    print(comparison.to_string(index=False))
    comparison.to_csv("outputs/model_comparison.csv", index=False)

    rf_pipeline = results["random_forest"]["pipeline"]
    importances = feature_importance_report(rf_pipeline)
    print("\n=== Random Forest feature importances ===")
    print(importances.to_string(index=False))
    importances.to_csv("outputs/feature_importances.csv", index=False)
