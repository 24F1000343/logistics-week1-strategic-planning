"""
main.py

Runs the full Week 4 pipeline end to end:
  1. Generate the simulated dataset (if not already present)
  2. Train and cross-validate the three models
  3. Evaluate on the held-out test set
  4. Run the optimization demos

Run:
    python main.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from generate_data import generate_dataset  # noqa: E402
from evaluate import evaluate_all, feature_importance_report  # noqa: E402
from train_models import train_and_select  # noqa: E402
import optimization  # noqa: E402

DATA_PATH = "data/logistics_shipments.csv"


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print("Generating simulated dataset...")
        generate_dataset().to_csv(DATA_PATH, index=False)

    print("\nTraining models...")
    results, (X_train, X_test, y_train, y_test) = train_and_select(DATA_PATH)

    print("\nEvaluating on held-out test set...")
    comparison = evaluate_all(results, X_test, y_test)
    print(comparison.to_string(index=False))
    comparison.to_csv("outputs/model_comparison.csv", index=False)

    importances = feature_importance_report(results["random_forest"]["pipeline"])
    importances.to_csv("outputs/feature_importances.csv", index=False)

    print("\nRunning optimization demos...")
    pipe = optimization.load_model()
    import pandas as pd

    sample = pd.DataFrame(
        {
            "distance_km": [45, 120, 15],
            "shipment_weight_kg": [20, 80, 5],
            "num_stops_on_route": [2, 6, 1],
            "dispatch_hour": [8, 17, 11],
            "day_of_week": ["Mon", "Fri", "Wed"],
            "weather_condition": ["clear", "rain", "clear"],
            "vehicle_type": ["van", "large_truck", "van"],
            "traffic_index": [6.5, 8.0, 3.0],
        }
    )
    flagged = optimization.flag_at_risk_shipments(pipe, sample, promised_hours=4.0)
    print(flagged[["distance_km", "predicted_hours", "at_risk"]])

    print("\nDone. See outputs/ for saved model, metrics and importances.")


if __name__ == "__main__":
    main()
