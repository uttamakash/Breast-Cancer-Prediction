import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

FEATURES = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']

data = load_breast_cancer()

try:
    df = pd.DataFrame(data.data, columns=data.feature_names)
except ValueError as e:
    print("Error creating DataFrame. Using default column names:", e)
    df = pd.DataFrame(data.data, columns=[f"feature_{i}" for i in range(data.data.shape[1])])

if len(data.target) == df.shape[0]:
    df['target'] = data.target
else:
    raise ValueError("Mismatch between number of rows in data and target!")

missing_features = [f for f in FEATURES if f not in df.columns]
if missing_features:
    raise ValueError(f"The following features are missing in the dataset: {missing_features}")

X = df[FEATURES]
y = df['target']

print("Class distribution:", np.bincount(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")
