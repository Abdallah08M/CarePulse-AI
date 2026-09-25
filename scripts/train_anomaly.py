
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# Load processed dataset
df = pd.read_csv("data/processed/processed_patient_data.csv")

# Features used for anomaly detection
features = [
    "HeartRate",
    "SpO2",
    "Temperature",
    "SystolicBP",
    "DiastolicBP",
    "Activity"
]

X = df[features]

# Train Isolation Forest
model = IsolationForest(
    contamination=0.03,   # about 3% anomalies expected
    random_state=42
)

model.fit(X)

# Predictions
pred = model.predict(X)

# Convert predictions
# IsolationForest: -1 = anomaly, 1 = normal
df["PredictedAnomaly"] = pred
df["PredictedAnomaly"] = df["PredictedAnomaly"].map({
    1: 0,
    -1: 1
})

# Compare with actual labels
print("\nConfusion Matrix")
print(confusion_matrix(df["StatusLabel"], df["PredictedAnomaly"]))

print("\nClassification Report")
print(classification_report(df["StatusLabel"], df["PredictedAnomaly"]))

# Save model
joblib.dump(model, "models/anomaly_detector.pkl")

print("\nAnomaly detection model saved successfully!")