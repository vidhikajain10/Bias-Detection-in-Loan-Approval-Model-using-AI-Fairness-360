import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from aif360.datasets import AdultDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

# Create results folder if not exists
os.makedirs("results", exist_ok=True)

# Load dataset
dataset = AdultDataset()
train, test = dataset.split([0.7], shuffle=True)

X_train = train.features
y_train = train.labels.ravel()

X_test = test.features
y_test = test.labels.ravel()

# -----------------------------
# STEP 1 — Train Original Model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

test_pred = test.copy()
test_pred.labels = y_pred.reshape(-1, 1)

metric_before = ClassificationMetric(
    test,
    test_pred,
    unprivileged_groups=[{'sex': 0}],
    privileged_groups=[{'sex': 1}]
)

accuracy_before = accuracy_score(y_test, y_pred)

# -----------------------------
# STEP 2 — Apply Reweighing
# -----------------------------
RW = Reweighing(
    unprivileged_groups=[{'sex': 0}],
    privileged_groups=[{'sex': 1}]
)

train_transf = RW.fit_transform(train)

X_train_rw = train_transf.features
y_train_rw = train_transf.labels.ravel()

model_rw = LogisticRegression(max_iter=1000)
model_rw.fit(X_train_rw, y_train_rw)

y_pred_rw = model_rw.predict(X_test)

test_pred_rw = test.copy()
test_pred_rw.labels = y_pred_rw.reshape(-1, 1)

metric_after = ClassificationMetric(
    test,
    test_pred_rw,
    unprivileged_groups=[{'sex': 0}],
    privileged_groups=[{'sex': 1}]
)

accuracy_after = accuracy_score(y_test, y_pred_rw)

# -----------------------------
# Print Results
# -----------------------------
print("----- BEFORE MITIGATION -----")
print("Accuracy:", accuracy_before)
print("Statistical Parity:", metric_before.statistical_parity_difference())
print("Disparate Impact:", metric_before.disparate_impact())

print("\n----- AFTER MITIGATION -----")
print("Accuracy:", accuracy_after)
print("Statistical Parity:", metric_after.statistical_parity_difference())
print("Disparate Impact:", metric_after.disparate_impact())

# -----------------------------
# Save Comparison Plot
# -----------------------------
metrics = ["Statistical Parity", "Disparate Impact"]
before = [
    metric_before.statistical_parity_difference(),
    metric_before.disparate_impact()
]
after = [
    metric_after.statistical_parity_difference(),
    metric_after.disparate_impact()
]

x = np.arange(len(metrics))

plt.bar(x - 0.2, before, width=0.4, label="Before")
plt.bar(x + 0.2, after, width=0.4, label="After")

plt.xticks(x, metrics)
plt.legend()
plt.title("Fairness Before vs After Mitigation")

plt.savefig("results/fairness_comparison.png")
plt.close()

print("\nFairness comparison plot saved in results/")
