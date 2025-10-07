from __future__ import annotations
import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05):
    num = df.select_dtypes(include="number")
    if num.empty:
        return None
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(num)
    scores = iso.decision_function(num)
    labels = iso.predict(num)
    res = num.copy()
    res["_anomaly_score"] = scores
    res["_is_anomaly"] = (labels == -1).astype(int)
    return res
