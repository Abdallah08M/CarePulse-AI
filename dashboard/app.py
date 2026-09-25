import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------
st.set_page_config(
    page_title="CarePulse AI",
    page_icon="🩺",
    layout="wide",
)

DATA_PATH = "data/processed/processed_patient_data.csv"
PREDICTION_MODEL_PATH = "models/heart_rate_prediction.pkl"
ANOMALY_MODEL_PATH = "models/anomaly_detector.pkl"

REQUIRED_COLUMNS = [
    "Timestamp", "HeartRate", "SpO2", "Temperature",
    "SystolicBP", "DiastolicBP", "Activity",
]

# --------------------------------------------------
# STYLE
# --------------------------------------------------
st.markdown("""
<style>
.stApp{
    background:linear-gradient(135deg,#FCE4EC,#FFFFFF,#E3F2FD);
}
.main-title{
    font-size:40px;
    font-weight:bold;
    color:#1E3A5F;
    margin-bottom:0;
}
.card{
    background:white;
    padding:20px 25px;
    border-radius:18px;
    box-shadow:0px 4px 12px rgba(0,0,0,.08);
    height:100%;
}
.prediction{
    background:linear-gradient(135deg,#64B5F6,#F8BBD9);
    padding:18px;
    border-radius:18px;
    text-align:center;
    color:white;
    font-weight:bold;
}
.alert-box{
    background:#FFEBEE;
    border-left:8px solid #E91E63;
    padding:18px 22px;
    border-radius:15px;
}
.status-pill{
    display:inline-block;
    padding:8px 14px;
    border-radius:10px;
    font-weight:bold;
}
.status-ok{ background:#E8F5E9; color:#2E7D32; }
.status-alert{ background:#FFEBEE; color:#C2185B; }
.section-title{
    color:#1E3A5F;
    margin-top:6px;
}
table.med-info{ width:100%; border-collapse:collapse; font-size:16px; }
table.med-info td{ padding:6px 4px; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DATA + MODEL LOADING (cached, with safe fallbacks)
# --------------------------------------------------
def _generate_demo_data(n=400) -> pd.DataFrame:
    """Synthetic vitals so the dashboard still runs if real files are missing."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range(end=datetime.now(), periods=n, freq="min")

    heart_rate = rng.normal(78, 6, n)
    spo2 = rng.normal(97, 1.2, n)
    temperature = rng.normal(36.8, 0.3, n)
    systolic = rng.normal(122, 8, n)
    diastolic = rng.normal(80, 6, n)
    activity = rng.choice(
        ["Resting", "Sleeping", "Walking", "Light Exercise"], size=n
    )

    # Sprinkle in some abnormal events so alerts / trends are visible.
    event_idx = rng.choice(n, size=max(6, n // 40), replace=False)
    for i in event_idx:
        kind = rng.integers(0, 4)
        if kind == 0:
            heart_rate[i] += rng.uniform(30, 45)
        elif kind == 1:
            spo2[i] -= rng.uniform(6, 10)
        elif kind == 2:
            temperature[i] += rng.uniform(1.5, 2.5)
        else:
            systolic[i] += rng.uniform(25, 40)
            diastolic[i] += rng.uniform(15, 25)

    df = pd.DataFrame({
        "Timestamp": timestamps,
        "HeartRate": heart_rate.round(1),
        "SpO2": spo2.clip(70, 100).round(1),
        "Temperature": temperature.round(1),
        "SystolicBP": systolic.round(0),
        "DiastolicBP": diastolic.round(0),
        "Activity": activity,
    })
    df["StatusLabel"] = (
        (df["HeartRate"] > 110) | (df["HeartRate"] < 55)
        | (df["SpO2"] < 92) | (df["Temperature"] > 38)
        | (df["SystolicBP"] > 145) | (df["DiastolicBP"] > 95)
    ).astype(int)
    return df


class FallbackPredictor:
    """Naive next-heart-rate estimate used only if the real model can't be loaded."""
    def predict(self, X: pd.DataFrame):
        base = X["HeartRate"].to_numpy()
        return base + np.random.uniform(-2, 2, size=len(X))


class FallbackAnomalyDetector:
    """Simple threshold-based stand-in for the trained anomaly model."""
    def predict(self, X: pd.DataFrame):
        row = X.iloc[0]
        abnormal = (
            row["HeartRate"] > 110 or row["HeartRate"] < 55
            or row["SpO2"] < 92 or row["Temperature"] > 38
            or row["SystolicBP"] > 145 or row["DiastolicBP"] > 95
        )
        return np.array([-1 if abnormal else 1])


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, bool]:
    try:
        df = pd.read_csv(DATA_PATH)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in data file: {missing}")
        return df, False
    except Exception:
        return _generate_demo_data(), True


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[object, object, bool]:
    try:
        prediction_model = joblib.load(PREDICTION_MODEL_PATH)
        anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
        return prediction_model, anomaly_model, False
    except Exception:
        return FallbackPredictor(), FallbackAnomalyDetector(), True


df, using_demo_data = load_data()
prediction_model, anomaly_model, using_fallback_models = load_models()

if using_demo_data or using_fallback_models:
    missing_bits = []
    if using_demo_data:
        missing_bits.append(f"`{DATA_PATH}`")
    if using_fallback_models:
        missing_bits.append("model files")
    st.info(
        f"⚠️ Couldn't find {', '.join(missing_bits)} — showing demo data / "
        "simplified logic so the dashboard still works. Add the real files to "
        "see live predictions."
    )

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
st.session_state.setdefault("index", 0)
st.session_state.setdefault("alerts", [])
st.session_state.setdefault("alerted_indices", set())
st.session_state.setdefault("last_update", time.time())

row = df.iloc[st.session_state.index]

# --------------------------------------------------
# AI PREDICTION + ALERT LOGIC
# --------------------------------------------------
features = pd.DataFrame([{
    "HeartRate": row["HeartRate"],
    "SpO2": row["SpO2"],
    "Temperature": row["Temperature"],
    "SystolicBP": row["SystolicBP"],
    "DiastolicBP": row["DiastolicBP"],
    "Activity": row["Activity"],
}])

predicted_hr = prediction_model.predict(features)[0]
is_anomaly = anomaly_model.predict(features)[0] == -1

reasons = []
if is_anomaly:
    reasons.append("Unusual vitals pattern flagged by anomaly model")
if row["HeartRate"] > 110:
    reasons.append("Elevated heart rate")
if row["HeartRate"] < 55:
    reasons.append("Low heart rate")
if row["SpO2"] < 92:
    reasons.append("Low oxygen saturation")
if row["Temperature"] > 38:
    reasons.append("Fever")
if row["SystolicBP"] > 145:
    reasons.append("High systolic blood pressure")
if row["DiastolicBP"] > 95:
    reasons.append("High diastolic blood pressure")

is_alert = len(reasons) > 0

# Record alert once per reading (index-based, no more duplicate spam)
if is_alert and st.session_state.index not in st.session_state.alerted_indices:
    st.session_state.alerted_indices.add(st.session_state.index)
    st.session_state.alerts.append({
        "Reading Time": row.get("Timestamp", datetime.now()),
        "Logged At": datetime.now().strftime("%H:%M:%S"),
        "Heart Rate": row["HeartRate"],
        "SpO₂": row["SpO2"],
        "Reason": "; ".join(reasons),
    })

# --------------------------------------------------
# HEALTH SCORE
# --------------------------------------------------
score = 100
if row["HeartRate"] > 110 or row["HeartRate"] < 55:
    score -= 20
if row["SpO2"] < 92:
    score -= 25
if row["Temperature"] > 38:
    score -= 15
if row["SystolicBP"] > 145 or row["DiastolicBP"] > 95:
    score -= 15
if is_anomaly:
    score -= 10
score = max(0, min(100, score))

if score >= 80:
    gauge_color = "#2E7D32"
elif score >= 50:
    gauge_color = "#F9A825"
else:
    gauge_color = "#C2185B"

# --------------------------------------------------
# HEADER
# --------------------------------------------------
left, right = st.columns([3, 1])

with left:
    st.markdown('<div class="main-title">💖 CarePulse AI</div>', unsafe_allow_html=True)
    st.caption("AI-Powered Elderly Patient Remote Monitoring")

with right:
    current = datetime.now()
    st.markdown(f"""
    <div class="card" style="text-align:center;">
        <div style="color:#5C6BC0;">LIVE TIME</div>
        <h3 style="margin:4px 0;">{current.strftime("%I:%M:%S %p")}</h3>
        <div>{current.strftime("%d %b %Y")}</div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# PATIENT PROFILE
# --------------------------------------------------
st.markdown('<h2 class="section-title">👤 Patient Profile</h2>', unsafe_allow_html=True)

left_col, right_col = st.columns([1.2, 2.3])

with left_col:
    st.markdown("""
    <div class="card" style="text-align:center;">
        <div style="font-size:80px;">👵</div>
        <h2 style="margin:10px 0 5px 0;color:#1E3A5F;white-space:nowrap;font-size:28px;">
            Sarah Johnson
        </h2>
        <p style="color:#5C6BC0;font-size:18px;">Patient ID: EP-001</p>
        <div class="status-pill status-ok">🟢 Connected</div>
        <div style="margin-top:15px;font-size:18px;">🔋 Battery: 98%</div>
    </div>
    """, unsafe_allow_html=True)

with right_col:
    st.markdown(f"""
    <div class="card">
        <h3 style="color:#1E3A5F;margin-top:0;">Medical Information</h3>
        <table class="med-info">
            <tr><td><b>Age</b></td><td>74 Years</td><td><b>Gender</b></td><td>Female</td></tr>
            <tr><td><b>Blood Group</b></td><td>O+</td><td><b>Height</b></td><td>160 cm</td></tr>
            <tr><td><b>Weight</b></td><td>62 kg</td><td><b>Allergy</b></td><td>Penicillin</td></tr>
            <tr><td><b>Activity</b></td><td colspan="3">{row["Activity"]}</td></tr>
        </table>
        <hr style="margin:18px 0;">
        <h4 style="color:#1E3A5F;margin-bottom:8px;">Emergency Contact</h4>
        <p style="margin:4px 0;">👤 John Johnson</p>
        <p style="margin:4px 0;">📞 +1 555-123-4567</p>
        <p style="margin:4px 0;">📍 Home Monitoring Enabled</p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# AI PREDICTION + HEALTH SCORE
# --------------------------------------------------
pred_col, score_col = st.columns([2, 1])

with pred_col:
    st.markdown(f"""
    <div class="prediction">
        Predicted Next Heart Rate
        <h1 style="margin:6px 0 0 0;">{predicted_hr:.1f} BPM</h1>
    </div>
    """, unsafe_allow_html=True)

with score_col:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Health Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": gauge_color},
        },
    ))
    gauge.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20), paper_bgcolor="white")
    st.plotly_chart(gauge, use_container_width=True)

# --------------------------------------------------
# PATIENT STATUS
# --------------------------------------------------
st.markdown('<h2 class="section-title">🚨 Patient Status</h2>', unsafe_allow_html=True)

if is_alert:
    st.markdown(f"""
    <div class="alert-box">
        <h3 style="margin-top:0;color:#C2185B;">Emergency Alert</h3>
        <p><b>Patient:</b> Sarah Johnson</p>
        <p><b>Heart Rate:</b> {row["HeartRate"]} BPM &nbsp;|&nbsp;
           <b>SpO₂:</b> {row["SpO2"]}% &nbsp;|&nbsp;
           <b>Temperature:</b> {row["Temperature"]}°C</p>
        <p><b>Reason:</b> {"; ".join(reasons)}</p>
        <p><b>Caregiver:</b> Notified (Simulation)</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.success("Patient condition is stable.")

# --------------------------------------------------
# HEART RATE TREND
# --------------------------------------------------
st.markdown('<h2 class="section-title">📈 Heart Rate Trend</h2>', unsafe_allow_html=True)

history = df.iloc[max(0, st.session_state.index - 30): st.session_state.index + 1]

fig = px.line(history, x="Timestamp", y="HeartRate", markers=True)
fig.update_traces(line=dict(color="#64B5F6", width=4))
fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=20, b=20))
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# CURRENT READING
# --------------------------------------------------
st.markdown('<h2 class="section-title">📋 Current Reading</h2>', unsafe_allow_html=True)
st.dataframe(pd.DataFrame([row]), use_container_width=True)

# --------------------------------------------------
# ALERT HISTORY
# --------------------------------------------------
st.markdown('<h2 class="section-title">🚨 Alert History</h2>', unsafe_allow_html=True)

if len(st.session_state.alerts) == 0:
    st.info("No alerts yet.")
else:
    st.dataframe(pd.DataFrame(st.session_state.alerts[::-1]), use_container_width=True)

# --------------------------------------------------
# CONTROLS
# --------------------------------------------------
st.markdown('<h2 class="section-title">🎮 Simulation Controls</h2>', unsafe_allow_html=True)

b1, b2, b3, b4 = st.columns(4)

with b1:
    if st.button("Next Reading"):
        st.session_state.index = (st.session_state.index + 1) % len(df)
        st.rerun()

with b2:
    if st.button("Jump to Emergency"):
        if "StatusLabel" in df.columns:
            abnormal = df[df["StatusLabel"] == 1]
            if not abnormal.empty:
                # jump to the next abnormal reading after the current index
                after = abnormal[abnormal.index > st.session_state.index]
                target = after.index[0] if not after.empty else abnormal.index[0]
                st.session_state.index = target
                st.rerun()
            else:
                st.warning("No emergency readings found in the data.")
        else:
            st.warning("Data has no 'StatusLabel' column to jump to.")

with b3:
    if st.button("Reset"):
        st.session_state.index = 0
        st.session_state.alerts = []
        st.session_state.alerted_indices = set()
        st.rerun()

with b4:
    auto = st.checkbox("Live Simulation")

# --------------------------------------------------
# LIVE SIMULATION
# --------------------------------------------------
if auto:
    time.sleep(0.5)
    st.session_state.index = (st.session_state.index + 1) % len(df)
    st.session_state.last_update = time.time()
    st.rerun()

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#5C6BC0;">
CarePulse AI • AI for IoT Capstone Project • Elderly Patient Remote Monitoring
</div>
""", unsafe_allow_html=True)
