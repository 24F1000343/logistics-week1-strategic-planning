# Week 3 – Advanced Data Analysis and Visualization in Logistics

This repo contains the Week 3 internship deliverable: exploratory data analysis and visualization on a simulated SwiftCart Logistics shipment dataset.

**Scenario:** SwiftCart Logistics, the same regional e-commerce fulfillment company from Weeks 1 and 2, now has a clean shipment dataset (per Week 2) and wants insights on delivery time, cost, and delay drivers to act on the KPIs defined in Week 1.

## Contents

- `Week3_Logistics_Analysis_Report.docx` – full report: dataset description, EDA, 6 visualizations with justification for each chart type, analytical insights, and recommendations
- `generate_data.py` – generates the synthetic shipment dataset (1,748 shipments, 8 cities, Road/Rail/Air, with realistic missing values and outliers)
- `eda_visuals.py` – runs the EDA and produces all charts (boxplot, scatter, histogram, grouped bar, correlation heatmap, dual-axis trend)
- `logistics_shipments.csv` – the generated dataset used for the analysis

## Key Findings

- Distance is the main driver of delivery time (r = 0.56); weight is the main driver of cost (r = 0.23) but barely affects speed
- Air shipping costs ~2.5–3x road/rail and cuts delivery time by close to half, but doesn't carry a matching reliability edge into Tier-2 cities
- Every mode is less reliable into Tier-2 destinations than Tier-1; Air and Rail into Tier-2 have the highest delay rates in the dataset (~12% each)
- Overall delay rate across the dataset: ~9%

## Techniques Used

- Exploratory data analysis (descriptive statistics, skewness, correlation analysis)
- Data visualization with matplotlib and seaborn (boxplot, scatter, histogram + KDE, grouped bar chart, heatmap, dual-axis time series)
- Synthetic data generation designed to mirror realistic logistics data patterns (right-skewed weight/cost, mode-dependent speed and cost, injected missing values and outliers)

## Next Step

Sets up for predictive modeling on the same feature set — a regression model for delivery time and a classifier for delay risk.
