"""
generate_data.py

Creates a simulated logistics shipment dataset for the Week 4 task
(Predictive Modeling and Optimization in Logistics Systems).

Since we don't have access to a real carrier's data during this
internship, this script builds a synthetic dataset whose structure
and relationships mirror what a genuine shipment dataset looks like:
distance, weight, stops, traffic and weather all push delivery time
up, with realistic noise, a few missing values, and a few outliers
mixed in on purpose so the cleaning steps in preprocessing.py have
something real to do.

Run:
    python src/generate_data.py
Produces:
    data/logistics_shipments.csv
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_ROWS = 5000


def generate_dataset(n_rows: int = N_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    distance_km = rng.gamma(shape=6.0, scale=8.0, size=n_rows)          # ~10-150 km
    shipment_weight_kg = rng.gamma(shape=2.0, scale=15.0, size=n_rows)  # skewed weight
    num_stops_on_route = rng.poisson(lam=3, size=n_rows)
    dispatch_hour = rng.integers(0, 24, size=n_rows)
    day_of_week = rng.choice(
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], size=n_rows
    )
    weather_condition = rng.choice(
        ["clear", "rain", "fog"], size=n_rows, p=[0.7, 0.22, 0.08]
    )
    vehicle_type = rng.choice(
        ["van", "small_truck", "large_truck"], size=n_rows, p=[0.5, 0.35, 0.15]
    )

    # Traffic is worse during rush hours (7-10 and 16-19) - mild collinearity
    # with dispatch_hour, intentionally, so the cleaning/EDA steps have to
    # deal with it.
    base_traffic = rng.uniform(1, 6, size=n_rows)
    rush_hour_mask = np.isin(dispatch_hour, [7, 8, 9, 16, 17, 18])
    traffic_index = np.clip(
        base_traffic + rush_hour_mask * rng.uniform(2, 4, size=n_rows), 1, 10
    )

    weather_multiplier = np.select(
        [weather_condition == "rain", weather_condition == "fog"],
        [0.35, 0.7],
        default=0.0,
    )
    vehicle_penalty = np.select(
        [vehicle_type == "small_truck", vehicle_type == "large_truck"],
        [0.15, 0.35],
        default=0.0,
    )

    noise = rng.normal(0, 0.4, size=n_rows)

    # Non-linear / interaction / threshold effects on purpose:
    #  - traffic has an accelerating (squared) effect as congestion builds
    #  - a hard "gridlock" penalty kicks in once traffic crosses a threshold
    #  - bad weather costs more on longer routes (interaction with distance)
    #  - a long route AND bad weather together trigger an extra compounding
    #    delay (a genuine interaction, not just additive)
    #  - routes with many stops get an extra fatigue/delay penalty past a
    #    threshold, and that penalty is worse in heavy traffic
    # These kinks and thresholds are exactly what a plain linear model
    # cannot represent but a tree-based model can, which is the gap the
    # model comparison in evaluate.py is designed to reveal.
    traffic_effect = 0.10 * traffic_index + 0.018 * traffic_index**2
    gridlock_penalty = np.where(traffic_index > 7.5, 1.4, 0.0)

    weather_distance_interaction = weather_multiplier * (distance_km / 40.0)
    long_route_bad_weather_penalty = np.where(
        (distance_km > 90) & (weather_condition != "clear"), 1.2, 0.0
    )

    many_stops_penalty = np.where(num_stops_on_route >= 6, 0.8, 0.0)
    stops_traffic_interaction = 0.05 * num_stops_on_route * (traffic_index / 5.0)
    stops_traffic_compounding = np.where(
        (num_stops_on_route >= 6) & (traffic_index > 7.5), 1.0, 0.0
    )

    delivery_time_hours = (
        0.04 * distance_km
        + 0.01 * shipment_weight_kg
        + 0.20 * num_stops_on_route
        + traffic_effect
        + gridlock_penalty
        + weather_distance_interaction
        + long_route_bad_weather_penalty
        + many_stops_penalty
        + stops_traffic_interaction
        + stops_traffic_compounding
        + vehicle_penalty
        + 1.0  # base handling/loading time
        + noise
    )
    delivery_time_hours = np.clip(delivery_time_hours, 0.5, None)

    df = pd.DataFrame(
        {
            "distance_km": distance_km.round(2),
            "shipment_weight_kg": shipment_weight_kg.round(1),
            "num_stops_on_route": num_stops_on_route,
            "dispatch_hour": dispatch_hour,
            "day_of_week": day_of_week,
            "weather_condition": weather_condition,
            "vehicle_type": vehicle_type,
            "traffic_index": traffic_index.round(2),
            "delivery_time_hours": delivery_time_hours.round(2),
        }
    )

    # --- Inject realistic imperfections on purpose ---
    # 1) A handful of missing weather readings
    missing_idx = rng.choice(n_rows, size=int(0.02 * n_rows), replace=False)
    df.loc[missing_idx, "weather_condition"] = np.nan

    # 2) A few unrealistic distance outliers (sensor / logging errors)
    outlier_idx = rng.choice(n_rows, size=int(0.005 * n_rows), replace=False)
    df.loc[outlier_idx, "distance_km"] = df.loc[outlier_idx, "distance_km"] * rng.uniform(
        6, 10, size=len(outlier_idx)
    )

    return df


if __name__ == "__main__":
    dataset = generate_dataset()
    dataset.to_csv("data/logistics_shipments.csv", index=False)
    print(f"Saved {len(dataset)} rows to data/logistics_shipments.csv")
    print(dataset.head())
