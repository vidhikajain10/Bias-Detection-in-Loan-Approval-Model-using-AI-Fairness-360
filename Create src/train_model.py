from aif360.datasets import AdultDataset
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
dataset = AdultDataset()
train, test = dataset.split([0.7], shuffle=True)

X_train = train.features
y_train = train.labels.ravel()

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully.")
