from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

def detect_anomalies(df, method="isolation_forest"):
    df = df.copy()
    
    if method == "isolation_forest":
        model = IsolationForest(contamination=0.03, random_state=42)
        df["anomaly"] = model.fit_predict(df[["energy_kwh"]])
        
    elif method == "z_score":
        mean = df["energy_kwh"].mean()
        std = df["energy_kwh"].std()
        z_scores = (df["energy_kwh"] - mean) / std
        df["anomaly"] = (np.abs(z_scores) > 3).astype(int)
        df["anomaly"] = df["anomaly"].replace({0: 1, 1: -1})  # keep -1 as anomaly

    elif method == "rolling_median":
        rolling_median = df["energy_kwh"].rolling(window=12, center=True).median()
        diff = np.abs(df["energy_kwh"] - rolling_median)
        threshold = diff.mean() + 3 * diff.std()
        df["anomaly"] = (diff > threshold).astype(int)
        df["anomaly"] = df["anomaly"].replace({0: 1, 1: -1})

    else:
        raise ValueError(f"Unknown method: {method}")
    
    return df
