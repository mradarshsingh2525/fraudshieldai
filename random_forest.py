# =========================================================
# FraudShield AI - Random Forest Training
# =========================================================

import joblib

from preprocessing import (
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("Training FraudShield AI Random Forest Model")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=150,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ---------------------------------------------------------
# TRAIN
# ---------------------------------------------------------

print("\nTraining model...")

model.fit(
    X_train_scaled,
    y_train
)

print("Training completed!")


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

y_pred = model.predict(X_test_scaled)


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

cm = confusion_matrix(y_test, y_pred)


print("\nModel Performance")
print("-" * 40)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------

MODEL_PATH = __file__

from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "random_forest_model.pkl"
)

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print(f"Model location: {MODEL_PATH}")

print("\n" + "=" * 60)
print("FraudShield AI model ready!")
print("=" * 60)