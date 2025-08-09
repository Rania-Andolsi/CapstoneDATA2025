# Capstone_DATA_SCIENCE2025
DATASCIENCE_Oct2025

🧠 Project Title: SmartEnergyWatch – Anomaly Detection & Performance Impact Analysis in Energy Consumption

📌 Project Description

This capstone project, conducted within the framework of the Rome Business School’s Master in Data Science, aims to develop a data-driven solution to detect anomalies in energy consumption and analyze the impact of specific equipment issues on overall performance.

Using modern data science techniques—such as time series analysis, anomaly detection algorithms, and performance prediction models—the project seeks to support preventive maintenance, improve energy efficiency, and reduce operational risks in smart industrial or commercial environments.

The proposed solution will simulate or ingest real-world datasets from equipment such as HVAC systems, manufacturing machinery, or server infrastructure, and apply machine learning models to identify unusual patterns and correlate them with system faults or degradation.

🎯 Objectives

Problem Framing: Define clear Key Performance Indicators (KPIs) such as kWh per cycle, operational efficiency, idle energy loss, and anomaly rate.

Data Pipeline: Collect, clean, and process energy consumption data and maintenance logs (real or synthetic).

Anomaly Detection: Apply statistical or machine learning techniques (Isolation Forest, LSTM, Prophet, etc.) to detect irregular energy patterns.

Performance Impact Analysis: Assess how detected anomalies correlate with system failures, downtime, or efficiency losses.

Visualization & Reporting: Build an interactive dashboard (e.g., Streamlit, Plotly Dash, or Grafana) to monitor equipment health and anomaly alerts.

Documentation & Architecture: Design a modular, scalable architecture with complete documentation, code, and technical design.

🧩 Team Members

Mohamed Seddik Kamoun – Project Lead 

Rania Andolsi – Model Optimization & Data Engineerin, Architecture & Research

Mootaz Bachtouli – Anomaly Detection & KPIs Analyst

Kousey Azouzi – Data Acquisition & EDA Lead

Hamza Bou Dokhane – Visualization & Dashboard Developer

💡 Technologies & Tools

Languages: Python 3.10 (Pandas, NumPy, Scikit-learn, TensorFlow, Statsmodels)

Data Viz: Streamlit, Plotly, Matplotlib, Grafana

Version Control: Git, GitHub

Collaboration: Notion, Trello, Google Docs

Deployment (optional): Docker, FastAPI, Heroku/Render

🧭 Methodology

We follow a 6-week Agile Sprint Model:

Week 1 – Business Understanding, KPIs, Architecture Design

Week 2 – Data Collection & Cleaning

Week 3 – Exploratory Data Analysis & Feature Engineering

Week 4 – Model Selection & Training

Week 5 – Evaluation, Optimization & Dashboard

Week 6 – Final Report, Presentation, and Deployment


Devolopment-Phase
(venv) PS C:\Users\fedy\CapstoneDATA2025> pip list
Package                   Version
------------------------- -----------
altair                    5.5.0
attrs                     25.3.0
blinker                   1.9.0
cachetools                6.1.0
certifi                   2025.8.3
charset-normalizer        3.4.3
click                     8.2.1
colorama                  0.4.6
contourpy                 1.3.3
cycler                    0.12.1
fonttools                 4.59.0
gitdb                     4.0.12
GitPython                 3.1.45
idna                      3.10
Jinja2                    3.1.6
joblib                    1.5.1
jsonschema                4.25.0
jsonschema-specifications 2025.4.1
kiwisolver                1.4.8
MarkupSafe                3.0.2
matplotlib                3.10.5
narwhals                  2.0.1
numpy                     2.3.2
packaging                 25.0
pandas                    2.3.1
pillow                    11.3.0
pip                       25.2
plotly                    6.2.0
protobuf                  6.31.1
pyarrow                   21.0.0
pydeck                    0.9.1
pyparsing                 3.2.3
python-dateutil           2.9.0.post0
pytz                      2025.2
referencing               0.36.2
requests                  2.32.4
rpds-py                   0.27.0
scikit-learn              1.7.1
scipy                     1.16.1
setuptools                80.9.0
six                       1.17.0
smmap                     5.0.2
streamlit                 1.48.0
tenacity                  9.1.2
threadpoolctl             3.6.0
toml                      0.10.2
tornado                   6.5.2
typing_extensions         4.14.1
tzdata                    2025.2
urllib3                   2.5.0
watchdog                  6.0.0
wheel                     0.45.1
(venv) PS C:\Users\fedy\CapstoneDATA2025> 
✅ Step 3: Anomaly Detection Module
MVP: Isolation Forest
Later: Upgrade to LSTM or Prophet for temporal anomalies.

python
Copier le code
from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.05)
data['anomaly'] = model.fit_predict(data[['energy_kwh']])
We can also use:

Prophet for seasonality & trend

LSTM for sequence-based detection

✅ Step 4: KPI & Impact Analysis
Use Pandas to:

Aggregate KPIs by day/week

Compare anomaly timestamps with maintenance_flag or downtime

Correlate anomaly patterns with system degradation

python
Copier le code
anomaly_days = data[data['anomaly'] == -1].groupby('date').size()
✅ Step 5: Streamlit Dashboard
Why Streamlit?

Lightweight

Fast to build

Interactive

Basic dashboard features:

Date range selector

Time series plot of energy vs anomalies

KPI summary cards

Machine-wise anomaly map

python
Copier le code
import streamlit as st
import matplotlib.pyplot as plt

st.title("SmartEnergyWatch Dashboard")

selected_machine = st.selectbox("Choose machine", data['machine_id'].unique())
filtered = data[data['machine_id'] == selected_machine]

st.line_chart(filtered.set_index('timestamp')['energy_kwh'])

anomalies = filtered[filtered['anomaly'] == -1]
st.write("Anomalies Detected:", anomalies.shape[0])