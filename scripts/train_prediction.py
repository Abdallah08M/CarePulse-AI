
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load processed dataset
df = pd.read_csv("data/processed/processed_patient_data.csv")

# Features
X = df[[
    "HeartRate",
    "SpO2",
    "Temperature",
    "SystolicBP",
    "DiastolicBP",
    "Activity"
]]

# Target
y = df["NextHeartRate"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Save model
joblib.dump(model, "models/heart_rate_prediction.pkl")

print("Prediction model saved successfully!")