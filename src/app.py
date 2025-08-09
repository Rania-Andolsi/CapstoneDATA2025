import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from src.anomaly_model import detect_anomalies

st.set_page_config(layout="wide")
st.title("⚡ SmartEnergyWatch Dashboard")

df = pd.read_csv("data/sample_energy.csv", parse_dates=["timestamp"])
df = detect_anomalies(df)

st.subheader("Energy Consumption Over Time")
st.line_chart(df.set_index("timestamp")["energy_kwh"])

st.subheader("Detected Anomalies")
st.write(df[df["anomaly"] == -1])
