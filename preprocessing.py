# =========================================================
# FraudShield AI - Data Preprocessing
# =========================================================

import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "creditcard.csv"
SCALER_PATH = BASE_DIR / "scaler.pkl"


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

print("=" * 60)
print("FraudShield AI - Data Preprocessing")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset loaded successfully!")
print(f"Dataset shape: {df.shape}")


# ---------------------------------------------------------
# FEATURES / TARGET
# ---------------------------------------------------------

X = df.drop("Class", axis=1)
y = df["Class"]

print(f"Features: {X.shape[1]}")
print(f"Samples : {X.shape[0]}")


# ---------------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train):,}")
print(f"Testing samples : {len(X_test):,}")


# ---------------------------------------------------------
# SCALING
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# SAVE SCALER
# ---------------------------------------------------------

joblib.dump(scaler, SCALER_PATH)

print("\nScaler created successfully!")
print(f"Scaler saved at: {SCALER_PATH}")

print("\nPreprocessing completed.")