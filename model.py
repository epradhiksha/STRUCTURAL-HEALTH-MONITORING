import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
# Load datasetdf = pd.read_excel("large_bridge_sensor_data.xlsx") generate a random dataset
# For demonstration purposes, we will generate a random dataset
# In practice, you would load your dataset from a file
df= pd.read_excel("large_bridge_sensor_data.xlsx")
# Drop missing or non-numeric values
df = df.dropna()
df = df[df[["ST355", "ST356", "ST348"]].applymap(np.isreal).all(axis=1)]
# Encode target labels
label_map = {"Healthy": 0, "Moderate": 1, "Critical": 2}
df["Condition"] = df["Condition"].map(label_map)
# Confirm dataset shape
if df.empty:
    raise ValueError("Dataset is empty after cleaning.")

# Features and target
X = df[["ST355", "ST356", "ST348"]]
y = df["Condition"]
# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest model with tuned params
model = RandomForestClassifier(n_estimators=200, max_depth=25, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(
    y_test, y_pred, target_names=["Healthy", "Moderate", "Critical"])
)

model_filename = "bridge_health_model.pkl"
joblib.dump(model, model_filename)
print(f"\nModel saved as: {model_filename}")
# Sample prediction
sample = pd.DataFrame([[600, 580, 560]], columns=["ST355", "ST356", "ST348"])
prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]
conditions = ["Healthy", "Moderate", "Critical"]

print(f"\nPrediction for sample {sample.values.tolist()[0]}: {conditions[prediction]}")
print("Prediction Probabilities:", dict(zip(conditions, probabilities.round(3))))
