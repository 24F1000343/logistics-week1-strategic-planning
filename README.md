# Logistics Data Analyst Internship — 4-Week Project

A 4-week internship project that moves a single logistics scenario through
the full analytics lifecycle: strategic planning → data cleaning → 
exploratory analysis → predictive modeling and optimization. Each week
builds directly on the last, so the repo reads as one continuous project
rather than four disconnected exercises.

**Scenario:** a mid-sized regional parcel carrier moving shipments from a
central hub to ~40 delivery zones. Customers are given a delivery window,
and management wants to know, before a truck leaves the yard, which
shipments are at risk of missing it — and what to do about it.

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1 | Strategic Planning & Data Exploration | Report: scenario, KPIs, roadmap |
| 2 | Data Collection, Cleaning & Preprocessing | Report: cleaning pipeline, pandas code |
| 3 | Advanced Data Analysis & Visualization | Report: EDA, charts, insights |
| 4 | Predictive Modeling & Optimization | **This code**: trained models + optimization tools |

---

## Week 1 — Strategic Planning and Data Exploration

**Objective:** define the logistics scenario, identify KPIs, and lay out a
strategic roadmap for using data science to address it.

- **Scenario:** delivery-time reliability for a regional parcel carrier
  dispatching from one hub to ~40 zones.
- **KPIs identified:**
  1. On-time delivery rate (% of shipments within the promised window)
  2. Average delivery time per route / zone
  3. Cost per shipment (fuel + labor, proxy for route efficiency)
- **Data science techniques proposed:** regression (predict delivery time),
  clustering (group delivery zones by risk profile), optimization (route and
  schedule adjustments based on model output).
- **Roadmap:** data collection → cleaning → exploratory analysis →
  predictive modeling → optimization — i.e., exactly the arc this repo follows.

## Week 2 — Data Collection, Cleaning, and Preprocessing

**Objective:** build a reliable, analysis-ready dataset from raw (simulated)
logistics data.

- Simulated a shipment-level dataset (distance, weight, stops, dispatch hour,
  day of week, weather, vehicle type, traffic index, delivery time).
- Identified realistic data-quality issues: missing weather readings,
  outlier distances (sensor/logging errors), mild collinearity between
  traffic and dispatch hour.
- **Cleaning approach:** impute missing categorical values with the mode,
  cap outliers at the 99th percentile rather than deleting rows, one-hot
  encode categoricals, scale numerics for linear models.
- This cleaning logic is implemented directly in
  [`src/preprocessing.py`](src/preprocessing.py) and reused by every later
  step, so Week 2's plan and Week 4's code are the same pipeline.

## Week 3 — Advanced Data Analysis and Visualization

**Objective:** explore the cleaned dataset and turn it into insight through
visualization.

- Ran exploratory data analysis on the cleaned dataset: distributions of
  delivery time, correlations between distance/traffic/weather and delivery
  time, central tendency and spread per vehicle type and weather condition.
- Produced visualizations (histograms, scatter plots, correlation heatmap)
  to surface which variables actually drive delivery delays.
- **Key finding carried into Week 4:** distance and traffic index show the
  strongest relationship with delivery time, with weather and stop count as
  secondary factors — this directly shaped which features and models were
  prioritized in the predictive modeling step.

## Week 4 — Predictive Modeling and Optimization in Logistics Systems

This is the week with working code in this repo. It forecasts shipment
delivery time and turns that forecast into three operational optimization
tools.

### Problem

Predict `delivery_time_hours` for a shipment at the moment it is dispatched,
using features known at dispatch time: distance, weight, number of stops,
dispatch hour, day of week, weather, vehicle type, and a traffic index.


### Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run everything

```bash
python main.py
```

This generates the dataset (if missing), trains Linear Regression, a
Decision Tree, and a Random Forest with 5-fold cross-validation, evaluates
all three on a held-out test set, saves the best model, and runs a small
demo of the optimization tools.

### Run steps individually

```bash
python src/generate_data.py    # data/logistics_shipments.csv
python src/train_models.py     # outputs/best_model.joblib
python src/evaluate.py         # outputs/model_comparison.csv, feature_importances.csv
python src/optimization.py     # risk-flagging + dispatch-hour demo
```

### Results (on the generated dataset, seed=42)

| Model              | RMSE (hrs) | MAE (hrs) | R²   |
|---------------------|-----------:|----------:|-----:|
| Random Forest        | ~0.58      | ~0.43     | 0.85 |
| Linear Regression    | ~0.66      | ~0.49     | 0.81 |
| Decision Tree        | ~0.70      | ~0.54     | 0.78 |

Random Forest is selected as the production model. Traffic index and
distance are the two strongest predictors of delivery time, matching the
Week 3 EDA findings.

### Optimization tools (`src/optimization.py`)

- **`flag_at_risk_shipments`** — scores shipments at dispatch and flags any
  predicted to exceed the promised delivery window.
- **`best_stop_order`** — brute-force search over stop orderings (small
  routes, ≤7 stops) to minimize total predicted delivery time.
- **`best_dispatch_hour`** — simulates the same shipment across candidate
  dispatch hours to find the lowest-traffic time to send it.

### Notes

The dataset is synthetic (see `src/generate_data.py`), generated to mirror
realistic relationships (distance, traffic, weather, and stops driving up
delivery time, including non-linear and threshold effects like traffic
gridlock) with injected missing values and outliers, since no live carrier
data was available during this internship. Swapping in a real shipment
export only requires matching the column names in `src/preprocessing.py`.

---

## Overall Conclusion

Across the four weeks, this project moved from identifying a business
problem (unpredictable delivery times) and its KPIs, to preparing trustworthy
data, to understanding which factors actually drive the problem, to building
and validating a model that predicts it — and finally to converting that
model into tools a dispatcher could use the same day: a risk alert, a route
re-ordering heuristic, and a schedule-shifting recommendation. The
recommended next step beyond this internship is to validate the Week 4
model against real operational data and retrain it on a rolling basis as
routes and seasons change.
