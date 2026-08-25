import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("Fraud Detection Project Started!")

# Get the folder where this Python file is located
BASE_DIR = Path(__file__).resolve().parent

# Dataset path
DATASET_PATH = BASE_DIR / "dataset" / "creditcard.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)

print("\nDataset Loaded Successfully!")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df["Class"].value_counts())

print("\nClass Percentage:")
print(df["Class"].value_counts(normalize=True) * 100)