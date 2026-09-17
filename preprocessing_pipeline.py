"""
Week 2 - Data Collection, Cleaning, and Preprocessing
SwiftCart Logistics - preprocessing pipeline for order/shipment data.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------
# 1. Initial Data Profiling
# ---------------------------------------------------------------
df = pd.read_csv('swiftcart_orders.csv')

print(df.info())
print(df.isnull().sum())
print(df.describe(include='all'))


# ---------------------------------------------------------------
# 2. Handling Missing Values
# ---------------------------------------------------------------
# Drop unrecoverable missing delivery dates on 'delivered' orders
df = df[~((df['status'] == 'delivered') & (df['delivery_date'].isna()))]

# Median imputation by group (robust to skew/outliers)
df['distance_km'] = df.groupby(['warehouse_id', 'zone_id'])['distance_km'] \
    .transform(lambda x: x.fillna(x.median()))

df['order_value'] = df.groupby('product_category')['order_value'] \
    .transform(lambda x: x.fillna(x.median()))


# ---------------------------------------------------------------
# 3. Outlier Detection (IQR Method)
# ---------------------------------------------------------------
def flag_outliers_iqr(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (series < lower) | (series > upper), lower, upper


outlier_mask, low, high = flag_outliers_iqr(df['distance_km'])
print(f'Outliers found: {outlier_mask.sum()}')

# Cap (winsorize) rather than drop, to preserve sample size
df['distance_km'] = df['distance_km'].clip(lower=low, upper=high)

# Business-rule correction: zero distance on a delivered order is invalid
df.loc[(df['distance_km'] == 0) & (df['status'] == 'delivered'), 'distance_km'] = pd.NA


# ---------------------------------------------------------------
# 4. Deduplication and Category Cleanup
# ---------------------------------------------------------------
df = df.sort_values('last_updated').drop_duplicates(subset='order_id', keep='last')

df['shipping_mode'] = df['shipping_mode'].str.strip().str.title()
df['shipping_mode'] = df['shipping_mode'].replace({'Exp': 'Express', 'Std': 'Standard'})


# ---------------------------------------------------------------
# 5. Normalization / Scaling
# ---------------------------------------------------------------
features = ['distance_km', 'order_value', 'quantity']
X_train, X_test = train_test_split(df[features], test_size=0.2, random_state=42)

# Fit scaler on training data only, then apply to both sets (avoids data leakage)
minmax = MinMaxScaler().fit(X_train)
X_train_scaled = minmax.transform(X_train)
X_test_scaled = minmax.transform(X_test)

print('Preprocessing complete. Train shape:', X_train_scaled.shape)
