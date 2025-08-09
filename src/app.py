import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from src.anomaly_model import detect_anomalies

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="SmartEnergyWatch Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# LOAD & PREPROCESS DATA
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/sample_energy.csv", parse_dates=["timestamp"])
    df = detect_anomalies(df)
    df["date"] = df["timestamp"].dt.date
    return df

df = load_data()

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
st.sidebar.title("🔍 Filters")
selected_machine = st.sidebar.selectbox("Select Machine ID", df["machine_id"].unique())
date_range = st.sidebar.date_input("Date Range", [df["date"].min(), df["date"].max()])

filtered_df = df[
    (df["machine_id"] == selected_machine) &
    (df["date"] >= date_range[0]) &
    (df["date"] <= date_range[1])
]

# ----------------------------
# KPI METRICS
# ----------------------------
st.title("⚡ SmartEnergyWatch Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🔌 kWh per Cycle", f"{filtered_df['energy_kwh'].mean():.2f}")

with col2:
    efficiency = (filtered_df['is_operational'] * filtered_df['energy_kwh']).sum() / (filtered_df['energy_kwh'].sum() + 1e-9)
    st.metric("⚙️ Efficiency Score", f"{efficiency:.2%}")

with col3:
    idle_energy = filtered_df[filtered_df["is_operational"] == 0]["energy_kwh"].sum()
    st.metric("🛑 Idle Energy Loss (kWh)", f"{idle_energy:.2f}")

with col4:
    anomaly_rate = (filtered_df["anomaly"] == -1).mean()
    st.metric("🚨 Anomaly Rate", f"{anomaly_rate:.2%}")

# ----------------------------
# MAIN VISUALIZATION
# ----------------------------
st.markdown("### 📈 Energy Consumption & Anomalies")

import plotly.express as px

fig = px.line(
    filtered_df,
    x="timestamp",
    y="energy_kwh",
    title="Energy Consumption Over Time",
    labels={"energy_kwh": "kWh", "timestamp": "Time"},
    color_discrete_sequence=["#00bfff"]
)

# Add anomaly points
anomalies = filtered_df[filtered_df["anomaly"] == -1]
fig.add_scatter(
    x=anomalies["timestamp"],
    y=anomalies["energy_kwh"],
    mode="markers",
    name="Anomalies",
    marker=dict(color="red", size=6)
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# TABS FOR DETAILS
# ----------------------------
tab1, tab2 = st.tabs(["📋 Raw Data", "🚨 Anomalies Only"])

with tab1:
    st.dataframe(filtered_df, use_container_width=True)

with tab2:
    st.dataframe(anomalies, use_container_width=True)
