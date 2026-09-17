# Week 2 – Data Collection, Cleaning, and Preprocessing for Logistics Analysis

This repo contains the Week 2 internship deliverable, building on the Week 1 strategic plan by preparing a working logistics dataset for analysis.

**Scenario:** SwiftCart Logistics, the same regional e-commerce fulfillment company from Week 1, needs a clean, analysis-ready shipment dataset before any forecasting or route-optimization work can begin.

## Contents

- `Week2_Data_Cleaning_Preprocessing_Report.docx` – full report covering dataset characteristics, data quality issues found, cleaning methodology, and reflection on how data quality affects downstream analysis
- `data_cleaning_pipeline.py` – Python (pandas) code for the preprocessing pipeline: missing value handling, outlier detection, and normalization

## Key Data Quality Issues Addressed

- Missing values in shipment weight and delivery timestamps
- Outliers in transportation cost and delivery time
- Inconsistent city/zone naming across source records
- Unnormalized numeric fields (distance, weight, cost) ahead of modeling

## Techniques Used

- Missing value imputation (mean/median and rule-based, depending on the field)
- Outlier detection using IQR and z-score methods
- Min-Max and Z-score normalization for numeric features
- pandas-based cleaning pipeline (`fillna`, `drop_duplicates`, `clip`, custom validators)

## Why It Matters

Every KPI defined in Week 1 (OTIF, cycle time, cost per unit, stockout rate) depends on shipment-level data being complete and consistent. This step exists so the Week 3 exploratory analysis and any later forecasting model aren't built on top of silently broken data.
