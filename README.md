# SwiftCart Logistics – Data Science Internship Project

This repo contains the Week 1–3 internship deliverables for a data science project focused on logistics and supply chain analytics.

**Scenario:** SwiftCart Logistics, a regional e-commerce fulfillment company, wants to improve delivery reliability, reduce transportation costs, and reduce stockouts using data science.

**Key KPIs tracked throughout:**

- On-Time In-Full (OTIF) delivery rate
- Average order-to-delivery cycle time
- Inventory turnover ratio
- Transportation cost per unit shipped
- Stockout rate

---

## Week 1 – Strategic Planning: Logistics Data Science Project

Defines the SwiftCart Logistics scenario, the five KPIs above, and an end-to-end roadmap for the analysis: data collection and cleaning, exploratory analysis, demand forecasting, zone clustering, and route optimization.

**Contents**

- `Week1_Strategic_Planning_Logistics_Report.docx` – full strategic planning report (KPIs, research, roadmap, conclusion)
- `logistics_analysis.py` – Python code snippets illustrating the planned approach: data cleaning, EDA, demand forecasting (regression), zone clustering, and route optimization (VRP)

**Techniques used:** Regression, clustering, and optimization applied to demand forecasting, delivery-zone segmentation, and vehicle routing.

---

## Week 2 – Data Collection, Cleaning, and Preprocessing for Logistics Analysis

Builds a clean, analysis-ready shipment dataset for SwiftCart — handling missing values, outliers, and normalization so the Week 3 analysis isn't built on broken data.

**Contents**

- `Week2_Data_Cleaning_Preprocessing_Report.docx` – dataset characteristics, data quality issues found, cleaning methodology, and reflection on how data quality affects downstream analysis
- `data_cleaning_pipeline.py` – pandas-based preprocessing pipeline: missing value handling, outlier detection, and normalization

**Techniques used:** Missing value imputation, IQR/z-score outlier detection, Min-Max and Z-score normalization.

---

## Week 3 – Advanced Data Analysis and Visualization in Logistics

Explores the cleaned shipment dataset (1,748 shipments across 8 cities) to find what's actually driving delivery time, cost, and delays — and turns that into recommendations tied back to the Week 1 KPIs.

**Contents**

- `Week3_Logistics_Analysis_Report.docx` – dataset description, EDA, 6 visualizations with justification for each chart type, analytical insights, and recommendations
- `generate_data.py` – generates the synthetic shipment dataset (Road/Rail/Air, realistic missing values and outliers)
- `eda_visuals.py` – runs the EDA and produces all charts (boxplot, scatter, histogram, grouped bar, correlation heatmap, dual-axis trend)
- `logistics_shipments.csv` – the generated dataset used for the analysis

**Key findings:**

- Distance drives delivery time (r = 0.56); weight drives cost (r = 0.23) but barely affects speed
- Air shipping costs ~2.5–3x road/rail and is faster, but doesn't carry a matching reliability edge into Tier-2 cities
- Air and Rail into Tier-2 destinations have the highest delay rates in the dataset (~12% each), against a ~9% overall delay rate

**Techniques used:** EDA (descriptive statistics, skewness, correlation analysis), data visualization with matplotlib/seaborn, synthetic data generation.

---

## Next Steps

Predictive modeling on the same feature set — a regression model for delivery time and a classifier for delay risk.
