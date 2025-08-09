import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from src.anomaly_model import detect_anomalies

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="SmartEnergyWatch Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
if st.button("ℹ️ Learn More About Metrics"):
    st.switch_page("pages/_about.py")

# ----------------------------
# MACHINE DESCRIPTIONS
# ----------------------------
MACHINE_DESCRIPTIONS = {
    "Machine_A1": "Precision CNC mill used for aerospace parts.",
    "Machine_B2": "Industrial 3D printer for rapid prototyping.",
    "Machine_C3": "Automated welding robot in chassis assembly.",
    "Machine_D4": "High-efficiency laser cutter for stainless steel."
}

# ----------------------------
# LOAD & PREPROCESS DATA
# ----------------------------
anomaly_method = st.sidebar.selectbox(
    "Anomaly Detection Method",
    ["isolation_forest", "z_score", "rolling_median"]
)

@st.cache_data
def load_data(anomaly_method):
    df = pd.read_csv("data/sample_energy.csv", parse_dates=["timestamp"])

    # Simulate more machines by duplicating A1
    machine_ids = list(MACHINE_DESCRIPTIONS.keys())
    all_data = []
    for i, machine_id in enumerate(machine_ids):
        df_copy = df.copy()
        df_copy["machine_id"] = machine_id
        df_copy["energy_kwh"] += i * 0.7  # Add variation
        all_data.append(df_copy)
    df = pd.concat(all_data, ignore_index=True)

    df = detect_anomalies(df, method=anomaly_method)
    df["date"] = df["timestamp"].dt.date
    return df

df = load_data(anomaly_method)

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
st.sidebar.title("🔍 Filters")

selected_machine = st.sidebar.selectbox("Select Machine ID", df["machine_id"].unique())

# Show description
desc = MACHINE_DESCRIPTIONS.get(selected_machine, "No description available.")
st.sidebar.markdown(f"**ℹ️ {selected_machine}**: {desc}")

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
    anomaly_count = (filtered_df["anomaly"] == -1).sum()
    total_points = len(filtered_df)
    anomaly_rate = anomaly_count / total_points if total_points else 0
    st.metric("🚨 Anomaly Rate", f"{anomaly_rate:.2%}")
    st.caption(f"Detected: {anomaly_count} anomalies using **{anomaly_method}**")

# ----------------------------
# MAIN CHART
# ----------------------------
st.markdown(f"### 📈 Energy Consumption Over Time ({anomaly_method.replace('_', ' ').title()})")

show_anomalies = st.checkbox("Show Anomalies on Chart", value=True)

fig = px.line(
    filtered_df,
    x="timestamp",
    y="energy_kwh",
    labels={"energy_kwh": "kWh", "timestamp": "Time"},
    color_discrete_sequence=["#00bfff"]
)

anomalies = filtered_df[filtered_df["anomaly"] == -1]
if show_anomalies and not anomalies.empty:
    fig.add_scatter(
        x=anomalies["timestamp"],
        y=anomalies["energy_kwh"],
        mode="markers",
        name="Anomalies",
        marker=dict(color="red", size=8, symbol="x"),
    )

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# ANOMALY SUMMARY
# ----------------------------
st.markdown("### 📊 Anomaly Summary")
st.dataframe(pd.DataFrame({
    "Method": [anomaly_method],
    "Total Points": [total_points],
    "Anomalies": [anomaly_count],
    "Anomaly Rate": [f"{anomaly_rate:.2%}"]
}))

# ----------------------------
# TABS
# ----------------------------
tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "🚨 Anomalies Only", "🔮 Forecast"])

with tab1:
    st.dataframe(filtered_df, use_container_width=True)

with tab2:
    st.dataframe(anomalies, use_container_width=True)

with tab3:
    st.subheader("📈 Energy Consumption Forecast")
    forecast_days = st.slider("Select Forecast Horizon (days)", 7, 60, 30)

    forecast_df = filtered_df[["timestamp", "energy_kwh"]].rename(
        columns={"timestamp": "ds", "energy_kwh": "y"}
    )

    if len(forecast_df) < 20:
        st.warning("Not enough data to generate a reliable forecast.")
    else:
        model = Prophet()
        model.fit(forecast_df)

        future = model.make_future_dataframe(periods=forecast_days, freq='D')
        forecast = model.predict(future)

        fig_forecast = plot_plotly(model, forecast)
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.markdown("#### 🔍 Forecast Summary")
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_days), use_container_width=True)
# ----------------------------
# EXPORT DATA
# ----------------------------
st.markdown("### 💾 Export Data")

csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Filtered Data as CSV",
    data=csv,
    file_name=f"{selected_machine}_energy_data.csv",
    mime="text/csv",
)
# ----------------------------
# COMPARISON OF METHODS
# ----------------------------
st.markdown("### 🧪 Anomaly Detection Comparison")

comparison_methods = ["isolation_forest", "z_score", "rolling_median"]
comparison_data = []

for method in comparison_methods:
    temp_df = detect_anomalies(filtered_df.copy(), method=method)
    count = (temp_df["anomaly"] == -1).sum()
    rate = count / len(temp_df)
    comparison_data.append({
        "Method": method.replace("_", " ").title(),
        "Anomalies": count,
        "Anomaly Rate": f"{rate:.2%}"
    })

st.dataframe(pd.DataFrame(comparison_data))
# ----------------------------
# ALERT SYSTEM
# ----------------------------
st.markdown("### 🚨 Alerts")

if anomaly_rate > 0.15:
    st.error("⚠️ High anomaly rate detected!")
elif idle_energy > 50:
    st.warning("⚠️ Significant idle energy loss observed.")
else:
    st.success("✅ All systems operating normally.")
