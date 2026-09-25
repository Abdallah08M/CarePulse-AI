
import pandas as pd
import numpy as np

# Make results reproducible
np.random.seed(42)

# Create 24 hours of data (one reading every minute)
minutes = 24 * 60
timestamps = pd.date_range("2025-01-01", periods=minutes, freq="min")

# Simulate normal vital signs
heart_rate = np.random.normal(72, 6, minutes).clip(55, 100)
spo2 = np.random.normal(98, 1, minutes).clip(92, 100)
temperature = np.random.normal(36.8, 0.2, minutes).clip(36.0, 37.5)
systolic_bp = np.random.normal(120, 8, minutes).clip(100, 150)
diastolic_bp = np.random.normal(80, 5, minutes).clip(60, 95)

# Simulate activity
activity = np.random.choice(
    ["Resting", "Walking"],
    size=minutes,
    p=[0.7, 0.3]
)

# Inject abnormal events
anomaly_indices = np.random.choice(minutes, 40, replace=False)

heart_rate[anomaly_indices] = np.random.randint(120, 150, len(anomaly_indices))
spo2[anomaly_indices] = np.random.randint(84, 90, len(anomaly_indices))
temperature[anomaly_indices] = np.random.uniform(38.0, 39.5, len(anomaly_indices))
systolic_bp[anomaly_indices] = np.random.randint(150, 180, len(anomaly_indices))
diastolic_bp[anomaly_indices] = np.random.randint(95, 110, len(anomaly_indices))

# Label anomalies
status = np.array(["Normal"] * minutes)
status[anomaly_indices] = "Abnormal"

# Build dataset
df = pd.DataFrame({
    "Timestamp": timestamps,
    "HeartRate": heart_rate.round(1),
    "SpO2": spo2.round(1),
    "Temperature": temperature.round(1),
    "SystolicBP": systolic_bp.round(0).astype(int),
    "DiastolicBP": diastolic_bp.round(0).astype(int),
    "Activity": activity,
    "Status": status
})

# Save dataset
output_path = "data/raw/elderly_patient_data.csv"
df.to_csv(output_path, index=False)

print("Dataset created successfully!")
print(df.head())
print(f"Saved to: {output_path}")