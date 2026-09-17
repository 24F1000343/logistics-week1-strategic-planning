"""
Week 3 - simulating a logistics dataset for a regional parcel delivery
company operating across a mix of metro and tier-2 cities in India.

I couldn't get hold of a real proprietary dataset (most courier companies
don't publish shipment-level data), so this is a synthetic dataset built
to mimic the structure and rough distributions you'd see in a real one -
based on patterns described in public logistics/e-commerce delivery
reports. Documented in the report itself.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

n = 2000

cities = ["Bengaluru", "Chennai", "Hyderabad", "Pune", "Coimbatore", "Nagpur", "Indore", "Kochi"]
city_tier = {
    "Bengaluru": "Tier-1", "Chennai": "Tier-1", "Hyderabad": "Tier-1", "Pune": "Tier-1",
    "Coimbatore": "Tier-2", "Nagpur": "Tier-2", "Indore": "Tier-2", "Kochi": "Tier-2"
}

modes = ["Road", "Air", "Rail"]
mode_probs = [0.72, 0.18, 0.10]

carriers = ["InHouse Fleet", "Partner-A", "Partner-B"]

df = pd.DataFrame({
    "shipment_id": range(1, n + 1),
    "origin_city": np.random.choice(cities, n),
    "dest_city": np.random.choice(cities, n),
    "mode": np.random.choice(modes, n, p=mode_probs),
    "carrier": np.random.choice(carriers, n, p=[0.5, 0.3, 0.2]),
})

# drop same-city shipments (not realistic for this dataset)
df = df[df.origin_city != df.dest_city].reset_index(drop=True)
n = len(df)

df["dest_tier"] = df["dest_city"].map(city_tier)

# distance is roughly correlated with tier-2 destinations being farther
# from the main hubs, plus random spread
base_distance = np.where(df.dest_tier == "Tier-2", 650, 350)
df["distance_km"] = np.round(base_distance + np.random.normal(0, 180, n)).clip(40, 2200)

# shipment weight in kg (right-skewed - most parcels are light, a long
# tail of bulk/commercial shipments)
df["weight_kg"] = np.round(np.random.gamma(shape=2.0, scale=3.2, size=n), 2).clip(0.2, 80)

# base delivery time depends on distance and mode, with air fastest per km
mode_speed_factor = df["mode"].map({"Road": 1.0, "Rail": 0.85, "Air": 0.35})
noise = np.random.normal(0, 0.6, n)
df["delivery_time_days"] = np.round(
    1.0 + (df["distance_km"] / 450) * mode_speed_factor + noise.clip(-0.8, 3), 1
).clip(0.5, 12)

# transportation cost: fixed handling fee + per-km + per-kg, with mode
# multiplier (air costs more)
mode_cost_factor = df["mode"].map({"Road": 1.0, "Rail": 0.8, "Air": 2.6})
df["cost_inr"] = np.round(
    (80 + df["distance_km"] * 0.9 + df["weight_kg"] * 25) * mode_cost_factor
    + np.random.normal(0, 60, n)
).clip(120, None)

# delay flag: shipments that took notably longer than the mode/distance
# would predict, plus a random operational-issue component (~12% base rate)
expected_days = 1.0 + (df["distance_km"] / 450) * mode_speed_factor
delay_prob = np.clip(0.08 + (df["delivery_time_days"] - expected_days) * 0.12, 0.02, 0.9)
df["delayed"] = np.random.binomial(1, delay_prob)

# a few genuine missing values and a couple of stray outliers, since
# that's realistic for logistics ops data and week 2's task dealt with it
missing_idx = np.random.choice(df.index, size=25, replace=False)
df.loc[missing_idx, "weight_kg"] = np.nan

outlier_idx = np.random.choice(df.index, size=6, replace=False)
df.loc[outlier_idx, "cost_inr"] = df.loc[outlier_idx, "cost_inr"] * 4.5

df["ship_date"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(
    np.random.randint(0, 240, n), unit="D"
)

df.to_csv("/home/claude/week3/logistics_shipments.csv", index=False)
print(df.shape)
print(df.head())
print(df.isna().sum())
