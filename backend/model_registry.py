from __future__ import annotations
import pathlib, joblib, time
from typing import Any, Dict

REG_DIR = pathlib.Path("models")
REG_DIR.mkdir(exist_ok=True)

def save_model(pipeline: Any, name: str) -> str:
    ts = int(time.time())
    path = REG_DIR / f"{ts}_{name}.joblib"
    joblib.dump(pipeline, path)
    return str(path)
