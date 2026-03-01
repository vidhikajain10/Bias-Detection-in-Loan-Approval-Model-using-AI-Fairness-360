from aif360.datasets import AdultDataset
from aif360.metrics import ClassificationMetric
import pickle

# Load data
dataset = AdultDataset()
train, test = dataset.split([0.7], shuffle=True)

X_test = test.features

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)

test_pred = test.copy()
test_pred.labels = y_pred.reshape(-1, 1)

metric = ClassificationMetric(
    test,
    test_pred,
    unprivileged_groups=[{'sex': 0}],
    privileged_groups=[{'sex': 1}]
)

print("Statistical Parity Difference:", metric.statistical_parity_difference())
print("Disparate Impact:", metric.disparate_impact())
print("Equal Opportunity Difference:", metric.equal_opportunity_difference())
