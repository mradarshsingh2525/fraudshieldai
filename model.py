# ==============================
# Credit Card Fraud Detection
# Logistic Regression Model
# ==============================

from preprocessing import X_train_scaled, X_test_scaled, y_train, y_test

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==============================
# 1. Create Model
# ==============================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)



# ==============================
# 2. Train Model
# ==============================

print("Training Logistic Regression Model...")

model.fit(X_train_scaled, y_train)

print("Model Training Completed!")


# ==============================
# 3. Prediction
# ==============================

y_pred = model.predict(X_test_scaled)


# ==============================
# 4. Evaluation
# ==============================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance:")
print("----------------------------")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ==============================
# 5. Confusion Matrix
# ==============================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ==============================
# 6. Classification Report
# ==============================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))