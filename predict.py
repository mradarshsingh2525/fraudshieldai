# ==========================================
# Credit Card Fraud Detection - Prediction
# ==========================================

import joblib
import pandas as pd
from pathlib import Path


# ==========================================
# 1. Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"


# ==========================================
# 2. Load Model and Scaler
# ==========================================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("=" * 45)
print("   CREDIT CARD FRAUD DETECTION SYSTEM")
print("=" * 45)

print("Random Forest Model Loaded!")
print("Scaler Loaded!")


# ==========================================
# 3. Feature Names
# ==========================================

features = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]


# ==========================================
# 4. Take User Input
# ==========================================

print("\nEnter transaction details:")

values = []

for feature in features:
    value = float(input(f"{feature}: "))
    values.append(value)


# ==========================================
# 5. Create DataFrame
# ==========================================

input_data = pd.DataFrame(
    [values],
    columns=features
)


# ==========================================
# 6. Scale Input
# ==========================================

input_scaled = scaler.transform(input_data)


# ==========================================
# 7. Prediction
# ==========================================

prediction = model.predict(input_scaled)[0]

probabilities = model.predict_proba(input_scaled)[0]

fraud_probability = probabilities[1] * 100


# ==========================================
# 8. Display Result
# ==========================================

print("\n")
print("=" * 30)
print(" TRANSACTION PREDICTION")
print("=" * 30)

if prediction == 1:
    print("Transaction Status :", "⚠️ FRAUD")
else:
    print("Transaction Status :", "✅ NORMAL")

print(f"Fraud Probability   : {fraud_probability:.2f}%")

print("=" * 30)