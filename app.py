import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Industrial Fault Diagnosis",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND = "http://127.0.0.1:8520"
FEATURES = ["Vibration", "Temperature", "Current", "Noise", "RPM"]
UNITS = ["mm/s", "°C", "A", "dB", "RPM"]

MACHINES = {
    "M-01": {"name": "Industrial Motor M-01", "type": "Motor"},
    "P-02": {"name": "Industrial Pump P-02", "type": "Pump"},
    "C-03": {"name": "Air Compressor C-03", "type": "Compressor"},
    "F-04": {"name": "Cooling Fan F-04", "type": "Fan"},
}

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(0, 176, 255, .20), transparent 32%),
            radial-gradient(circle at 100% 20%, rgba(30, 80, 180, .24), transparent 35%),
            linear-gradient(135deg, #04152f 0%, #062b55 48%, #063b69 100%);
        color: #eaf6ff;
    }

    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #03152f 0%, #052b52 100%);
        border-right: 1px solid rgba(96, 196, 255, .25);
    }
    [data-testid="stSidebar"] * { color: #eaf6ff !important; }

    .block-container { padding-top: 2rem; max-width: 1450px; }

    .hero {
        padding: 26px 30px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(8, 61, 110, .96), rgba(4, 31, 65, .96));
        border: 1px solid rgba(94, 204, 255, .32);
        box-shadow: 0 18px 45px rgba(0, 0, 0, .30);
        margin-bottom: 20px;
    }
    .hero h1 { color: #ffffff; margin: 0; font-size: 2.1rem; }
    .hero p { color: #bcecff; margin: 8px 0 0; }

    .panel {
        padding: 20px;
        border-radius: 17px;
        background: rgba(4, 28, 56, .78);
        border: 1px solid rgba(97, 204, 255, .20);
        box-shadow: 0 14px 35px rgba(0, 0, 0, .20);
        margin-bottom: 18px;
    }

    .sensor-card {
        padding: 17px 12px;
        border-radius: 15px;
        text-align: center;
        background: linear-gradient(160deg, #0a4778, #062b53);
        border: 1px solid rgba(110, 218, 255, .25);
        min-height: 115px;
    }
    .sensor-name { color: #9bdfff; font-size: .82rem; font-weight: 700; }
    .sensor-value { color: white; font-size: 1.45rem; font-weight: 850; margin-top: 8px; }
    .sensor-unit { color: #7fc8ed; font-size: .76rem; }

    .status {
        padding: 15px 18px;
        border-radius: 14px;
        background: rgba(0, 190, 255, .10);
        border: 1px solid rgba(89, 208, 255, .28);
        color: #dff8ff;
        margin-bottom: 15px;
    }

    .accuracy-card {
        padding: 18px;
        border-radius: 15px;
        background: linear-gradient(160deg, #0b4c7d, #062c52);
        border: 1px solid rgba(117, 221, 255, .22);
        text-align: center;
    }
    .accuracy-title { color: #a9e8ff; font-weight: 750; }
    .accuracy-value { color: #ffffff; font-size: 2rem; font-weight: 900; margin-top: 4px; }

    .backend-note {
        padding: 12px 15px;
        border-radius: 12px;
        background: rgba(0, 130, 210, .12);
        border: 1px dashed rgba(110, 210, 255, .28);
        color: #bfeaff;
        font-size: .88rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(4, 40, 75, .80);
        border: 1px solid rgba(110, 214, 255, .20);
        padding: 12px;
        border-radius: 14px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 750;
        border: 1px solid rgba(120, 220, 255, .30);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "running" not in st.session_state:
    st.session_state.running = False
if "active_machine" not in st.session_state:
    st.session_state.active_machine = None
if "sensor_values" not in st.session_state:
    st.session_state.sensor_values = {}
if "diagnosis" not in st.session_state:
    st.session_state.diagnosis = {}
if "accuracy" not in st.session_state:
    st.session_state.accuracy = None


def api_get(path, timeout=8):
    try:
        r = requests.get(f"{BACKEND}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(path, payload=None, timeout=12):
    try:
        r = requests.post(
            f"{BACKEND}{path}",
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def refresh_reading(machine):
    result = api_post(f"/read/{machine}")
    if result:
        st.session_state.sensor_values[machine] = result["values"]
        diagnosis = api_post(
            "/diagnose",
            {"machine": machine, "values": result["values"]},
        )
        if diagnosis:
            st.session_state.diagnosis[machine] = diagnosis
        return result
    return None


@st.cache_data(ttl=300, show_spinner=False)
def get_accuracy_cached():
    return api_get("/model/accuracy", timeout=5)

def get_accuracy():
    if st.session_state.accuracy is None:
        st.session_state.accuracy = get_accuracy_cached()
    return st.session_state.accuracy


st.markdown(
    """
    <div class="hero">
        <h1>🏭 Intelligent Industrial Fault Diagnosis</h1>
        <p>Multi-Sensor Monitoring • Decision Tree • Random Forest • Automatic Fault Diagnosis</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    machine = st.selectbox(
        "Select machine",
        list(MACHINES.keys()),
        format_func=lambda x: f"{x} — {MACHINES[x]['name']}",
    )

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ Run", use_container_width=True):
            start = api_post(f"/start/{machine}")
            if start:
                st.session_state.active_machine = machine
                st.session_state.running = True
                refresh_reading(machine)
                st.rerun()
            else:
                st.error("Backend is not running.")
    with c2:
        if st.button("⏸ Pause", use_container_width=True):
            st.session_state.running = False
            st.rerun()

    if st.button("🔄 Refresh accuracy", use_container_width=True):
        st.session_state.accuracy = None
        st.rerun()

    st.markdown("---")
    st.markdown("### Backend")
    health = api_get("/health")
    if health:
        st.success("Backend connected")
        st.caption(f"Dataset: {health['rows']:,} rows")
    else:
        st.error("Backend unavailable")
        st.caption("Start FastAPI on port 8520.")

active = st.session_state.active_machine
if active is None:
    active = machine

if st.session_state.running and active == machine:
    live = refresh_reading(machine)
    status_text = "🟢 LIVE — sensor values are being read from the supplied dataset."
else:
    live = None
    status_text = "⚪ READY — press Run to acquire the selected machine reading."

st.markdown(f'<div class="status">{status_text}</div>', unsafe_allow_html=True)

info1, info2, info3, info4 = st.columns(4)
info1.metric("Selected machine", machine)
info2.metric("Machine type", MACHINES[machine]["type"])
info3.metric("System status", "RUNNING" if st.session_state.running else "PAUSED")
info4.metric(
    "Current diagnosis",
    st.session_state.diagnosis.get(machine, {}).get("fault", "Not analyzed"),
)

values = st.session_state.sensor_values.get(machine)
if values is None:
    values = [0, 0, 0, 0, 0]

st.markdown("## 📡 Multi-Sensor Data")
sensor_cols = st.columns(5)
for i, (name, value, unit) in enumerate(zip(FEATURES, values, UNITS)):
    with sensor_cols[i]:
        display = f"{value:.0f}" if name == "RPM" else f"{value:.2f}"
        st.markdown(
            f"""
            <div class="sensor-card">
                <div class="sensor-name">{name}</div>
                <div class="sensor-value">{display}</div>
                <div class="sensor-unit">{unit}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("## 🥧 Multi-Sensor Pie Chart")
st.caption(
    "The pie chart shows the relative normalized intensity of the five sensor channels. "
    "Normalization prevents RPM from dominating the chart because it has a different unit and scale."
)

# Feature-wise ranges from the supplied dataset are used for normalization.
sensor_max = np.array([20.0, 100.0, 200.0, 100.0, 5000.0])
sensor_min = np.array([0.0, 20.0, 0.0, 40.0, 500.0])
raw = np.asarray(values, dtype=float)
normalized = np.clip((raw - sensor_min) / (sensor_max - sensor_min), 0, 1)
if normalized.sum() == 0:
    normalized = np.ones(5)

fig = go.Figure(
    data=[
        go.Pie(
            labels=FEATURES,
            values=normalized,
            hole=0.42,
            textinfo="label+percent",
            hovertemplate="%{label}<br>Relative intensity: %{percent}<extra></extra>",
        )
    ]
)
fig.update_layout(
    title="Current Multi-Sensor Distribution",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#eaf6ff"),
    legend=dict(orientation="h", y=-0.08),
    margin=dict(l=10, r=10, t=55, b=10),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def safe_percent(value, default="N/A", decimals=1):
    """Format an optional numeric percentage without crashing on None/invalid data."""
    if value is None or value == "":
        return default
    try:
        number = float(value)
        if not np.isfinite(number):
            return default
        return f"{number:.{decimals}f}%"
    except (TypeError, ValueError):
        return default


diag = st.session_state.diagnosis.get(machine, {})
if diag:
    st.markdown("## 🧠 Automatic Diagnosis")
    a, b, c, d = st.columns(4)
    a.metric("Fault", diag.get("fault") or "Unknown")
    b.metric("Status", diag.get("status") or "Unknown")
    c.metric("Health", safe_percent(diag.get("health"), default="N/A", decimals=1))
    d.metric("Failure probability", safe_percent(diag.get("failure_probability"), default="N/A", decimals=1))

    model_cols = st.columns(2)
    for col, item in zip(model_cols, diag.get("models", [])):
        with col:
            st.markdown(
                f"""
                <div class="accuracy-card">
                    <div class="accuracy-title">{item['model']}</div>
                    <div class="accuracy-value">{item['confidence']:.1f}%</div>
                    <div>{item['prediction']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("## 🤖 Model Accuracy")
st.caption(
    "Validated on the uploaded equipment_anomaly_data.csv dataset using a held-out 80/20 test split. "
    "Training data, test data and confusion matrices remain backend-only."
)

accuracy = get_accuracy()
if accuracy:
    fault_models = accuracy.get("algorithms", [])
    acc_cols = st.columns(2)
    for col, item in zip(acc_cols, fault_models):
        with col:
            st.markdown(
                f"""
                <div class="accuracy-card">
                    <div class="accuracy-title">{item['model']} — Equipment Anomaly Accuracy</div>
                    <div class="accuracy-value">{item['accuracy']:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="backend-note">🔒 Backend retained: 80/20 train-test split, test predictions and confusion matrices. The uploaded anomaly dataset is used for the validated accuracy cards; the original 5-class fault-type benchmark remains backend-only.</div>', unsafe_allow_html=True)
else:
    st.warning("Model accuracy is unavailable until the FastAPI backend is running.")

st.markdown("## 🛠️ Maintenance Recommendation")
fault = diag.get("fault", "Normal")
recommendations = {
    "Normal": "Continue normal monitoring and follow the planned preventive-maintenance schedule.",
    "Overheating": "Inspect cooling, airflow, temperature rise and thermal protection.",
    "Bearing Failure": "Inspect bearing vibration, lubrication and mechanical alignment.",
    "Motor Failure": "Inspect motor current, winding condition, insulation and mechanical load.",
    "Power Failure": "Check supply voltage, current, connections and protection devices.",
    "Sensor Fault": "Inspect sensor wiring, sensor health and calibration.",
}
st.info(recommendations.get(fault, "Inspect the machine and review the backend diagnostic record."))

st.markdown(
    '<div class="backend-note">Model training: Decision Tree + Random Forest. '
    'The models are trained on the supplied Industrial IoT dataset and use reproducible tuning parameters. '
    'No synthetic training labels are created.</div>',
    unsafe_allow_html=True,
)

# No automatic polling loop: removing the sleep/rerun cycle keeps the app responsive.
# Press Run when a fresh reading is required.
