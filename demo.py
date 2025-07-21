from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from model import MulticlassLogisticRegression
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load and preprocess data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MulticlassLogisticRegression(
    learning_rate=0.1,
    max_iter=1000,
    lambda_reg=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
