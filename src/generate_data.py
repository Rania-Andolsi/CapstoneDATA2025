# src/generate_data.py
import pandas as pd
import numpy as np

np.random.seed(42)

timestamps = pd.date_range(start="2025-08-01", periods=500, freq="H")
energy = np.random.normal(loc=5, scale=1, size=500)
# Inject anomalies
energy[100] = 15
energy[300] = 0.2

df = pd.DataFrame({
    "timestamp": timestamps,
    "machine_id": "A1",
    "energy_kwh": energy,
    "is_operational": np.random.choice([0, 1], size=500),
    "maintenance_flag": 0
})

df.to_csv("data/sample_energy.csv", index=False)
