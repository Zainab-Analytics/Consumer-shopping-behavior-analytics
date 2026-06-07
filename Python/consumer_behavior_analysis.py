import pandas as pd
import numpy as np

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("shopping_trends.csv")

# ==========================
# DATA CLEANING
# ==========================

# Standardize column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Check missing values
missing_values = df.isnull().sum()

# Remove duplicates
df.drop_duplicates(inplace=True)

# ==========================
# KEY BUSINESS KPIs
# ==========================

total_customers = df["Customer_ID"].nunique()

total_revenue = df["Purchase_Amount_(USD)"].sum()

average_order_value = df["Purchase_Amount_(USD)"].mean()

average_rating = df["Review_Rating"].mean()

repeat_customers = (
    df[df["Previous_Purchases"] > 10]["Customer_ID"]
    .nunique()
)

# ==========================
# CUSTOMER SEGMENTATION
# ==========================

df["Customer_Segment"] = np.where(
    df["Previous_Purchases"] >= 30,
    "High Value",
    np.where(
        df["Previous_Purchases"] >= 10,
        "Medium Value",
        "Low Value"
    )
)

segment_summary = (
    df.groupby("Customer_Segment")
      .agg({
          "Purchase_Amount_(USD)": "mean",
          "Customer_ID": "count"
      })
      .rename(columns={
          "Purchase_Amount_(USD)": "Avg_Spend",
          "Customer_ID": "Customer_Count"
      })
)

# ==========================
# CATEGORY PERFORMANCE
# ==========================

category_sales = (
    df.groupby("Category")["Purchase_Amount_(USD)"]
      .sum()
      .sort_values(ascending=False)
)

# ==========================
# LOCATION ANALYSIS
# ==========================

top_locations = (
    df.groupby("Location")["Purchase_Amount_(USD)"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

# ==========================
# SEASONAL ANALYSIS
# ==========================

season_sales = (
    df.groupby("Season")["Purchase_Amount_(USD)"]
      .sum()
      .sort_values(ascending=False)
)

# ==========================
# PAYMENT METHOD ANALYSIS
# ==========================

payment_analysis = (
    df.groupby("Payment_Method")["Purchase_Amount_(USD)"]
      .sum()
      .sort_values(ascending=False)
)

# ==========================
# CUSTOMER LOYALTY ANALYSIS
# ==========================

loyal_customers = (
    df[df["Subscription_Status"] == "Yes"]
)

loyalty_revenue = loyal_customers["Purchase_Amount_(USD)"].sum()

# ==========================
# EXECUTIVE SUMMARY
# ==========================

print("=" * 50)
print("CONSUMER SHOPPING BEHAVIOR ANALYSIS")
print("=" * 50)

print(f"Total Customers: {total_customers}")
print(f"Total Revenue: ${total_revenue:,.2f}")
print(f"Average Order Value: ${average_order_value:.2f}")
print(f"Average Review Rating: {average_rating:.2f}")
print(f"Repeat Customers: {repeat_customers}")

print("\nTop Product Categories")
print(category_sales.head())

print("\nTop Revenue Locations")
print(top_locations)

print("\nSeasonal Revenue Analysis")
print(season_sales)

print("\nPayment Method Analysis")
print(payment_analysis)

print("\nCustomer Segmentation")
print(segment_summary)

print(f"\nRevenue From Subscribers: ${loyalty_revenue:,.2f}")
