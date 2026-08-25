import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Load dataset
df = pd.read_csv(BASE_DIR / "dataset" / "creditcard.csv")

# ==============================
# EDA - Exploratory Data Analysis
# ==============================

# Plot Class Distribution
plt.figure(figsize=(6, 4))

df["Class"].value_counts().plot(kind="bar")

plt.title("Transaction Class Distribution")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(BASE_DIR / "class_distribution.png")
plt.show()
# ==============================
# Dataset Information
# ==============================

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())
# ==============================
# Dataset Information
# ==============================

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())
# ==============================
# Fraud vs Normal Percentage
# ==============================

class_percentage = df["Class"].value_counts(normalize=True) * 100

print("\nClass Percentage:")
print(class_percentage)

print("\nFraud Percentage:")
print(class_percentage[1])
# ==============================
# Transaction Amount Distribution
# ==============================

plt.figure(figsize=(8, 5))

plt.hist(df["Amount"], bins=50)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(BASE_DIR / "amount_distribution.png")
plt.show()
# ==============================
# Normal vs Fraud Transaction Amount
# ==============================

normal_amount = df[df["Class"] == 0]["Amount"]
fraud_amount = df[df["Class"] == 1]["Amount"]

print("\nAverage Normal Transaction Amount:")
print(normal_amount.mean())

print("\nAverage Fraud Transaction Amount:")
print(fraud_amount.mean())

# Box Plot
plt.figure(figsize=(8, 5))

plt.boxplot(
    [normal_amount, fraud_amount],
    tick_labels=["Normal", "Fraud"]
)

plt.title("Transaction Amount: Normal vs Fraud")
plt.xlabel("Transaction Class")
plt.ylabel("Amount")

plt.tight_layout()

plt.savefig(BASE_DIR / "normal_vs_fraud_amount.png")
plt.show()
# ==============================
# Correlation with Class
# ==============================

correlation = df.corr()["Class"].sort_values(ascending=False)

print("\nCorrelation with Class:")
print(correlation)

# Plot top 10 correlated features
top_corr = correlation.drop("Class").sort_values().tail(10)

plt.figure(figsize=(8, 6))

top_corr.plot(kind="barh")

plt.title("Top 10 Features Correlated with Fraud")
plt.xlabel("Correlation with Class")

plt.tight_layout()

plt.savefig(BASE_DIR / "top_correlations.png")
plt.show()
