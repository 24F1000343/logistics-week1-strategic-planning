# Submission Descriptions (for the "Write description for the report" field)

---

## Week 1 – Strategic Planning and Data Exploration in Logistics

For this task, I defined a logistics scenario around SwiftCart Logistics, a regional e-commerce fulfillment company that wants to improve delivery reliability, cut transportation costs, and reduce stockouts using data science. I chose this scenario because it covers three distinct but connected problem areas — inventory management, route optimization, and supply chain integration — which gave me enough surface area to justify a range of analytical techniques rather than forcing a single method into an artificial context.

I identified five KPIs that I felt best captured operational health for a company like this: On-Time In-Full (OTIF) delivery rate, average order-to-delivery cycle time, inventory turnover ratio, transportation cost per unit shipped, and stockout rate. Each of these maps to a specific business decision (staffing, routing, reorder points), which I think matters more than picking metrics just because they're common in logistics literature.

The report walks through background research on how data science is typically applied in logistics, then lays out an end-to-end roadmap: data collection and cleaning, exploratory analysis, demand forecasting using regression, delivery zone clustering with K-Means, and route optimization framed as a vehicle routing problem. I included Python code snippets for each stage to make the plan concrete rather than purely conceptual, and closed with expected outcomes — mainly, how tighter demand forecasts and better-clustered delivery zones should directly move the OTIF and cost-per-unit KPIs. This plan is what Weeks 2 and 3 build on directly.

---

## Week 2 – Data Collection, Cleaning, and Preprocessing for Logistics Analysis

This task picks up from the Week 1 strategic plan for SwiftCart Logistics and focuses on getting a shipment-level dataset into a state that's actually usable for analysis. I treated this as a simulated real-world dataset with the kinds of problems you'd genuinely run into — missing values in fields like shipment weight and delivery timestamps, inconsistent city/zone naming across records, and outliers in cost and delivery time that would distort any KPI calculated naively from raw data.

For each issue, I documented not just the fix but the reasoning behind choosing it. Missing values were handled with a mix of median imputation and rule-based logic depending on the field, since a blanket approach (like dropping every incomplete row) would have thrown away usable data. Outliers were flagged using IQR and z-score methods, and I normalized numeric fields (distance, weight, cost) with Min-Max and Z-score scaling so they'd be on comparable scales for any later modeling step. All of this was implemented with pandas, and I included code snippets for each transformation so the methodology isn't just described in prose.

The reflection section ties this back to why data quality matters operationally: every KPI defined in Week 1 depends on shipment records being complete and consistent, and a forecasting or clustering model trained on unclean data would just encode those errors into its predictions. This preprocessing pipeline is what the Week 3 exploratory analysis is built directly on top of.

---

## Week 3 – Advanced Data Analysis and Visualization in Logistics

This task builds on the cleaned shipment data from Week 2 and focuses on exploring it properly — running exploratory data analysis, building visualizations, and turning both into insights that would actually be useful to someone running SwiftCart's delivery operations, rather than just producing charts for their own sake.

I worked with a simulated dataset of 1,748 shipments across eight cities (a mix of Tier-1 and Tier-2 destinations), covering distance, weight, transport mode, cost, delivery time, and a delay flag. I started with descriptive statistics and a correlation matrix, which showed that distance is the main driver of delivery time while weight mainly drives cost rather than speed — two separate levers that a pricing or SLA policy probably shouldn't bundle together. From there I built six visualizations, choosing each chart type deliberately: a boxplot for delivery time by mode (to show spread, not just averages), a scatter plot for cost versus distance, a histogram for the right-skewed weight distribution, a grouped bar chart for delay rate by city tier and mode, a heatmap for the correlation matrix, and a dual-axis chart for monthly volume against delivery performance.

The most useful finding was one I didn't expect going in: Air and Rail shipments into Tier-2 cities actually have the highest delay rates in the dataset, meaning Air's speed advantage doesn't come with a matching reliability advantage once the destination is a smaller city. I flagged this as the most actionable insight rather than smoothing it into a tidier story, since it directly affects how SwiftCart should think about mode allocation and SLA-setting for those specific routes.
