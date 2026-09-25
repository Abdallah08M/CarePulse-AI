# CarePulse AI

AI for IoT Capstone Project – Elderly Patient Remote Monitoring System

## Overview

CarePulse AI is a healthcare monitoring dashboard that simulates IoMT wearable sensor data for an elderly patient. The system predicts heart rate using machine learning, detects abnormal health conditions, and displays real-time patient information through an interactive Streamlit dashboard.

## Features

- 👤 Patient profile dashboard
- ❤️ Live vital sign monitoring
- 🤖 AI heart-rate prediction (Random Forest)
- 🚨 Anomaly detection (Isolation Forest)
- 📈 Heart-rate trend visualization
- 🎮 Real-time simulation controls

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Joblib

## Project Structure

```text
dashboard/      # Streamlit dashboard
data/           # Raw and processed datasets
models/         # Trained AI models
scripts/        # Data simulation and training scripts
reports/        # Project documentation
```

## Run the Project

```bash
python -m streamlit run dashboard/app.py
```

---
Developed as an AI for IoT Healthcare Capstone Project.
