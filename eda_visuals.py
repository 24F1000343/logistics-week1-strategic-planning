import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["figure.dpi"] = 140

df = pd.read_csv("/home/claude/week3/logistics_shipments.csv", parse_dates=["ship_date"])

# quick central-tendency / spread summary for the numeric fields I care about
summary = df[["distance_km", "weight_kg", "delivery_time_days", "cost_inr"]].describe().T
summary["skew"] = df[["distance_km", "weight_kg", "delivery_time_days", "cost_inr"]].skew()
summary.to_csv("/home/claude/week3/summary_stats.csv")
print(summary)

corr = df[["distance_km", "weight_kg", "delivery_time_days", "cost_inr"]].corr()
print(corr)

# ---- 1. Distribution of delivery time by mode ----
plt.figure(figsize=(7, 4.5))
sns.boxplot(data=df, x="mode", y="delivery_time_days", hue="mode",
            palette="Set2", legend=False)
plt.title("Delivery Time by Transport Mode")
plt.xlabel("Mode")
plt.ylabel("Delivery time (days)")
plt.tight_layout()
plt.savefig("/home/claude/week3/fig1_delivery_time_by_mode.png")
plt.close()

# ---- 2. Cost vs distance, coloured by mode ----
plt.figure(figsize=(7, 4.5))
sns.scatterplot(data=df, x="distance_km", y="cost_inr", hue="mode",
                 alpha=0.55, palette="Set2", s=28)
plt.title("Shipment Cost vs Distance")
plt.xlabel("Distance (km)")
plt.ylabel("Cost (INR)")
plt.tight_layout()
plt.savefig("/home/claude/week3/fig2_cost_vs_distance.png")
plt.close()

# ---- 3. Weight distribution (right-skewed) ----
plt.figure(figsize=(7, 4.5))
sns.histplot(df["weight_kg"].dropna(), bins=30, kde=True, color="#4C72B0")
plt.title("Shipment Weight Distribution")
plt.xlabel("Weight (kg)")
plt.ylabel("Number of shipments")
plt.tight_layout()
plt.savefig("/home/claude/week3/fig3_weight_distribution.png")
plt.close()

# ---- 4. Delay rate by city tier and mode ----
delay_rate = df.groupby(["dest_tier", "mode"])["delayed"].mean().reset_index()
delay_rate["delayed_pct"] = delay_rate["delayed"] * 100
plt.figure(figsize=(7, 4.5))
sns.barplot(data=delay_rate, x="mode", y="delayed_pct", hue="dest_tier", palette="Set1")
plt.title("Delay Rate by Destination Tier and Mode")
plt.xlabel("Mode")
plt.ylabel("Delayed shipments (%)")
plt.legend(title="Destination tier")
plt.tight_layout()
plt.savefig("/home/claude/week3/fig4_delay_rate.png")
plt.close()

# ---- 5. Correlation heatmap ----
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
plt.title("Correlation Between Key Variables")
plt.tight_layout()
plt.savefig("/home/claude/week3/fig5_correlation_heatmap.png")
plt.close()

# ---- 6. Monthly shipment volume and average delivery time ----
df["ship_month"] = df["ship_date"].dt.to_period("M").astype(str)
monthly = df.groupby("ship_month").agg(
    shipments=("shipment_id", "count"),
    avg_delivery=("delivery_time_days", "mean")
).reset_index()

fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax2 = ax1.twinx()
ax1.bar(monthly["ship_month"], monthly["shipments"], color="#8CA0C4", label="Shipment volume")
ax2.plot(monthly["ship_month"], monthly["avg_delivery"], color="#C44E52",
         marker="o", label="Avg delivery time")
ax1.set_ylabel("Shipment volume")
ax2.set_ylabel("Avg delivery time (days)")
ax1.set_xlabel("Month")
plt.title("Monthly Shipment Volume vs Average Delivery Time")
ax1.tick_params(axis="x", rotation=45)
fig.tight_layout()
plt.savefig("/home/claude/week3/fig6_monthly_trend.png")
plt.close()

print("done - all figures saved")

# a couple of numbers I'll quote directly in the report text
print("\nOverall delay rate: %.1f%%" % (df["delayed"].mean() * 100))
print("Avg cost - Air vs Road:", df.groupby("mode")["cost_inr"].mean())
print("Corr(distance, delivery_time):", corr.loc["distance_km", "delivery_time_days"])
print("Corr(distance, cost):", corr.loc["distance_km", "cost_inr"])
