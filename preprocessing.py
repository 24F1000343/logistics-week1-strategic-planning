"""
preprocessing.py

Data cleaning and the scikit-learn ColumnTransformer used to prepare
features for modeling. Kept separate from train_models.py so it can
be reused for both training and any future scoring/inference script.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_COLS = [
    "distance_km",
    "shipment_weight_kg",
    "num_stops_on_route",
    "dispatch_hour",
    "traffic_index",
]
CATEGORICAL_COLS = ["day_of_week", "weather_condition", "vehicle_type"]
TARGET_COL = "delivery_time_hours"


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV and apply the same cleaning steps used in Week 2."""
    df = pd.read_csv(csv_path)

    # Fill missing weather readings with the most frequent category
    if df["weather_condition"].isna().any():
        mode_value = df["weather_condition"].mode()[0]
        df["weather_condition"] = df["weather_condition"].fillna(mode_value)

    # Cap unrealistic distance outliers at the 99th percentile rather than
    # dropping the rows outright - an unusually long route is still a real
    # route, we just don't want a handful of bad readings to distort the
    # model's understanding of "normal".
    cap = df["distance_km"].quantile(0.99)
    df["distance_km"] = df["distance_km"].clip(upper=cap)

    return df


def split_features_target(df: pd.DataFrame):
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET_COL]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: scale numeric features, one-hot encode categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )
