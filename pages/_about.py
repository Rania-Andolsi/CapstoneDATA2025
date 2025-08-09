import streamlit as st

st.set_page_config(page_title="About SmartEnergyWatch", layout="wide")

st.title("📘 About SmartEnergyWatch")
st.markdown("""
Welcome to **SmartEnergyWatch**, your energy monitoring and anomaly detection assistant.

---

## 🔍 Purpose

This app helps:
- Track machine-level energy usage
- Analyze performance and downtime
- Detect unusual behavior automatically

---

## 📊 Metric Glossary

**🔌 kWh per Cycle**: Energy consumed on average per usage cycle  
**⚙️ Efficiency Score**: How much energy is used when machine is *actually operational*  
**🛑 Idle Energy Loss**: Wasted energy when machine is idle  
**🚨 Anomaly Rate**: Percentage of readings flagged as anomalies

---

## 🧪 Anomaly Detection Methods

### Isolation Forest
- Machine learning-based
- Detects rare or isolated behaviors
- Great for unknown patterns

### Z-Score
- Statistical deviation from mean
- Good for normally distributed data

### Rolling Median Deviation
- Compares values to recent historical trends
- Ideal for real-time drift detection

---

## 🏁 How to Use

1. Go to the main dashboard
2. Select a machine and detection method
3. View metrics, anomalies, and forecast
4. Export filtered data

---

""")
