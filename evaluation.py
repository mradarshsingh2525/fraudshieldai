# ==============================
# Model Evaluation
# ==============================

from preprocessing import X_test_scaled, y_test
from random_forest import model

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

import matplotlib.pyplot as plt


# ==============================
# 1. Prediction Probability
# ==============================

y_prob = model.predict_proba(X_test_scaled)[:, 1]


# ==============================
# 2. ROC-AUC Score
# ==============================

roc_auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC Score:")
print(f"{roc_auc:.4f}")


# ==============================
# 3. Precision-Recall Score
# ==============================

average_precision = average_precision_score(y_test, y_prob)

print("\nAverage Precision Score:")
print(f"{average_precision:.4f}")


# ==============================
# 4. ROC Curve
# ==============================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))

plt.plot(fpr, tpr)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Random Forest")

plt.tight_layout()

plt.savefig("roc_curve.png")

plt.show()


# ==============================
# 5. Precision-Recall Curve
# ==============================

precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(7, 5))

plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision-Recall Curve - Random Forest")

plt.tight_layout()

plt.savefig("precision_recall_curve.png")

plt.show()

print("\nEvaluation Completed Successfully!")