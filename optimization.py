"""
optimization.py

Turns the trained model into three operational tools:

1. flag_at_risk_shipments  - real-time risk scoring at dispatch
2. best_stop_order         - route re-sequencing heuristic
3. best_dispatch_hour      - weather/traffic-aware schedule shifting

These are intentionally lightweight (not a full VRP solver) so they
can run in near real time inside a dispatcher's existing workflow.
"""

from itertools import permutations

import joblib
import pandas as pd

MODEL_PATH = "outputs/best_model.joblib"


def load_model(path: str = MODEL_PATH):
    return joblib.load(path)


def flag_at_risk_shipments(
    pipe, shipments_df: pd.DataFrame, promised_hours: float, buffer_hours: float = 0.5
) -> pd.DataFrame:
    """Score shipments and flag any predicted to blow past the promised window."""
    preds = pipe.predict(shipments_df)
    out = shipments_df.copy()
    out["predicted_hours"] = preds.round(2)
    out["at_risk"] = out["predicted_hours"] > (promised_hours + buffer_hours)
    return out


def best_stop_order(pipe, stops_df: pd.DataFrame, max_permutations: int = 5040):
    """
    Try different stop orderings for a set of shipments on one vehicle run
    and return the ordering with the lowest total predicted delivery time.

    stops_df: one row per stop, same feature columns the model expects,
    with num_stops_on_route already reflecting the route length.
    Not a substitute for a full VRP solver - intended for small route
    sizes (up to ~7 stops) that a dispatcher can re-order on the fly.
    """
    n = len(stops_df)
    if n > 7:
        raise ValueError("best_stop_order is a brute-force heuristic; keep routes <= 7 stops.")

    best_order, best_total = None, float("inf")
    for order in permutations(range(n)):
        ordered = stops_df.iloc[list(order)]
        total = pipe.predict(ordered).sum()
        if total < best_total:
            best_total, best_order = total, order

    return best_order, round(best_total, 2)


def best_dispatch_hour(pipe, shipment_row: pd.Series, candidate_hours=range(6, 20)):
    """
    Simulate the same shipment dispatched at different hours (holding
    distance/weight/weather/vehicle fixed, since traffic is the variable
    that changes with dispatch hour) and return the hour with the lowest
    predicted delivery time.
    """
    candidates = pd.concat([shipment_row.to_frame().T] * len(list(candidate_hours)), ignore_index=True)
    candidates["dispatch_hour"] = list(candidate_hours)

    preds = pipe.predict(candidates)
    candidates["predicted_hours"] = preds.round(2)
    best_row = candidates.loc[candidates["predicted_hours"].idxmin()]
    return int(best_row["dispatch_hour"]), float(best_row["predicted_hours"])


if __name__ == "__main__":
    import pandas as pd

    pipe = load_model()

    # --- Demo 1: risk flagging on a few sample shipments ---
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
    flagged = flag_at_risk_shipments(pipe, sample, promised_hours=4.0)
    print("=== Risk flagging demo ===")
    print(flagged[["distance_km", "predicted_hours", "at_risk"]])

    # --- Demo 2: best dispatch hour for one shipment ---
    hour, hours_pred = best_dispatch_hour(pipe, sample.iloc[1])
    print(f"\nBest dispatch hour for shipment #2: {hour}:00 -> predicted {hours_pred} hrs")
