import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load the healthcare dataset
df = pd.read_csv("healthcare_dataset.csv")

# Drop unnecessary columns
df = df.drop(columns=[
    "Name",
    "Doctor",
    "Hospital",
    "Date of Admission",
    "Discharge Date"
])

# Convert categorical variables to numeric
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop([
    "Test Results_Inconclusive",
    "Test Results_Normal"
], axis=1)

y = df[[
    "Test Results_Inconclusive",
    "Test Results_Normal"
]]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Set:", X_train.shape)
print("Testing Set:", X_test.shape)

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(f"{accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test.values.argmax(axis=1),
    predictions.argmax(axis=1)
))