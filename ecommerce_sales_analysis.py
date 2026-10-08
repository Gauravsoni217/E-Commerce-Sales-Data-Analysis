import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("ecommerce_sales_raw.csv")

# Data Cleaning Using Pandas
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
for col in ["Quantity","Unit_Price","Revenue","Customer_Age","Rating"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["Rating"] = df["Rating"].fillna(df["Rating"].median())
df["Region"] = df["Region"].str.strip().str.title()
df["Payment_Method"] = df["Payment_Method"].str.strip().str.title()
df = df.drop_duplicates().reset_index(drop=True)
df.to_csv("ecommerce_sales_cleaned.csv", index=False)

# NumPy and Pandas Analysis
print("Total orders:", len(df))
print("Total revenue:", df["Revenue"].sum())
print("Mean revenue:", df["Revenue"].mean())
print("Median revenue:", df["Revenue"].median())
print("Standard deviation:", df["Revenue"].std())
print("Average rating:", df["Rating"].mean())

category_sales = df.groupby("Category")["Revenue"].agg(["sum","mean","count"])
product_sales = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)
monthly_sales = df.groupby("Month")["Revenue"].sum()

print("\nCategory-wise sales:\n", category_sales.sort_values("sum", ascending=False))
print("\nTop products:\n", product_sales.head(10))
print("\nMonthly sales:\n", monthly_sales)

corr = df[["Quantity","Unit_Price","Discount","Revenue","Customer_Age","Rating"]].corr()
print("\nCorrelation with Revenue:\n", corr["Revenue"].sort_values(ascending=False))

arr = df["Revenue"].to_numpy()
print("\nNumPy mean:", np.mean(arr))
print("NumPy median:", np.median(arr))
print("NumPy standard deviation:", np.std(arr))

# Matplotlib and Seaborn Visualizations
plt.figure(figsize=(10,6))
monthly_sales.plot(marker="o"); plt.title("Monthly E-Commerce Revenue Trend")
plt.xlabel("Month"); plt.ylabel("Revenue"); plt.xticks(rotation=45)
plt.tight_layout(); plt.show()

plt.figure(figsize=(10,6))
category_sales["sum"].sort_values().plot(kind="barh")
plt.title("Revenue by Product Category"); plt.xlabel("Revenue")
plt.tight_layout(); plt.show()

plt.figure(figsize=(10,6))
product_sales.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Products by Revenue"); plt.xlabel("Revenue")
plt.tight_layout(); plt.show()

plt.figure(figsize=(10,6))
df["Revenue"].plot(kind="hist",bins=25)
plt.title("Distribution of Order Revenue"); plt.xlabel("Revenue")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,8))
category_sales["sum"].plot(kind="pie",autopct="%1.1f%%",startangle=90)
plt.title("Category Share of Total Revenue"); plt.ylabel("")
plt.tight_layout(); plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=df,x="Category",y="Revenue")
plt.title("Revenue Distribution by Category"); plt.xticks(rotation=20)
plt.tight_layout(); plt.show()

plt.figure(figsize=(10,6))
df["Payment_Method"].value_counts().sort_values().plot(kind="barh")
plt.title("Orders by Payment Method"); plt.xlabel("Orders")
plt.tight_layout(); plt.show()

plt.figure(figsize=(10,7))
sns.heatmap(corr,annot=True,fmt=".2f")
plt.title("Correlation Heatmap"); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,6))
sns.violinplot(data=df,x="Category",y="Revenue")
plt.title("Revenue Distribution by Category"); plt.xticks(rotation=20)
plt.tight_layout(); plt.show()

plt.figure(figsize=(10,6))
plt.scatter(df["Quantity"],df["Revenue"],alpha=.45)
plt.title("Quantity vs Revenue"); plt.xlabel("Quantity"); plt.ylabel("Revenue")
plt.tight_layout(); plt.show()
