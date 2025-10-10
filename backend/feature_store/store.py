from __future__ import annotations
import pandas as pd, pathlib, time
class FeatureStore:
    def __init__(self, root: str = "feature_store"):
        self.root = pathlib.Path(root); self.root.mkdir(exist_ok=True)
    def write(self, df: pd.DataFrame, name: str)->str:
        ts = int(time.time()); path = self.root/f"{name}_{ts}.parquet"
        df.to_parquet(path, index=False)
        latest = self.root/f"{name}_latest.parquet"
        df.to_parquet(latest, index=False)
        return str(path)
    def read_latest(self, name: str)->pd.DataFrame:
        return pd.read_parquet(self.root/f"{name}_latest.parquet")
