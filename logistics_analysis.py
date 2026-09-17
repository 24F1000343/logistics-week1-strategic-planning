"""
Week 1 - Strategic Planning: Logistics Data Science Project
Scenario: SwiftCart Logistics - demand forecasting, zone clustering,
and route optimization for a regional e-commerce fulfillment company.

These are illustrative snippets from the Week 1 strategic planning report.
"""

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Data Loading and Cleaning
# ---------------------------------------------------------------
orders = pd.read_csv('order_history.csv', parse_dates=['order_date', 'delivery_date'])
inventory = pd.read_csv('warehouse_inventory.csv')

# Remove duplicate order IDs and rows with missing delivery timestamps
orders = orders.drop_duplicates(subset='order_id')
orders = orders.dropna(subset=['delivery_date'])

# Derive delivery lead time in hours
orders['lead_time_hrs'] = (orders['delivery_date'] - orders['order_date']).dt.total_seconds() / 3600


# ---------------------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------------------
orders['lead_time_hrs'].hist(bins=40)
plt.title('Distribution of Delivery Lead Time (hrs)')
plt.xlabel('Hours')
plt.ylabel('Order count')
plt.show()

late_rate = orders.groupby('warehouse_id')['is_late'].mean()
print(late_rate.sort_values(ascending=False))


# ---------------------------------------------------------------
# 3. Demand Forecasting (Regression)
# ---------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

features = ['day_of_week', 'month', 'is_promo', 'prior_week_sales', 'price']
X = demand_df[features]
y = demand_df['units_sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print('MAE:', mean_absolute_error(y_test, preds))


# ---------------------------------------------------------------
# 4. Customer / Zone Clustering
# ---------------------------------------------------------------
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

zone_features = zones_df[['avg_order_freq', 'avg_order_size', 'distance_to_warehouse']]
scaled = StandardScaler().fit_transform(zone_features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
zones_df['cluster'] = kmeans.fit_predict(scaled)


# ---------------------------------------------------------------
# 5. Route Optimization (Vehicle Routing, pseudocode-style)
# ---------------------------------------------------------------
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_index)
routing = pywrapcp.RoutingModel(manager)

def distance_callback(from_index, to_index):
    return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

transit_idx = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)
routing.AddDimension(transit_idx, 0, vehicle_capacity, True, 'Capacity')

search_params = pywrapcp.DefaultRoutingSearchParameters()
search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
solution = routing.SolveWithParameters(search_params)
