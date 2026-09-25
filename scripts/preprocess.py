
import pandas as pd

# Load raw data
df = pd.read_csv("data/raw/elderly_patient_data.csv")

# Convert timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Encode activity into numbers
df["Activity"] = df["Activity"].map({
    "Resting": 0,
    "Walking": 1
})

# Encode status
df["StatusLabel"] = df["Status"].map({
    "Normal": 0,
    "Abnormal": 1
})

# Create prediction target (next minute heart rate)
df["NextHeartRate"] = df["HeartRate"].shift(-1)

# Remove last row (no next value)
df = df.dropna()

# Save processed data
output = "data/processed/processed_patient_data.csv"
df.to_csv(output, index=False)

print("Processed dataset saved!")
print(df.head())