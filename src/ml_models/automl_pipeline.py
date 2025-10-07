from __future__ import annotations
import pathlib, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, f1_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import joblib

MODELS_DIR = pathlib.Path(__file__).resolve().parents[2] / "models" / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY = MODELS_DIR / "registry.json"

def _save_registry(entry: dict):
    if REGISTRY.exists():
        try:
            data = json.loads(REGISTRY.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    else:
        data = []
    data.append(entry)
    REGISTRY.write_text(json.dumps(data, indent=2), encoding="utf-8")

def train_automl(df: pd.DataFrame, target: str, random_state: int = 42):
    y = df[target]
    X = df.drop(columns=[target])

    is_classif = (y.nunique() <= max(20, int(0.05 * len(y)))) and (y.dtype == "object" or pd.api.types.is_integer_dtype(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y if is_classif else None)

    model = None
    metrics = {}

    if is_classif:
        try:
            model = LGBMClassifier(random_state=random_state)
        except Exception:
            try:
                model = XGBClassifier(random_state=random_state, eval_metric="logloss")
            except Exception:
                model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1_weighted": float(f1_score(y_test, pred, average="weighted"))
        }
        problem_type = "classification"
    else:
        try:
            model = LGBMRegressor(random_state=random_state)
        except Exception:
            try:
                model = XGBRegressor(random_state=random_state)
            except Exception:
                model = RandomForestRegressor(random_state=random_state)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = float(mean_squared_error(y_test, pred, squared=False))
        r2 = float(r2_score(y_test, pred))
        metrics = {"rmse": rmse, "r2": r2}
        problem_type = "regression"

    # Save model
    model_path = MODELS_DIR / f"model_{problem_type}_{random_state}.joblib"
    joblib.dump({"model": model, "columns": list(X.columns), "target": target, "problem_type": problem_type}, model_path)

    # registry
    _save_registry({
        "path": str(model_path.name),
        "target": target,
        "problem_type": problem_type,
        "metrics": metrics
    })

    return model, metrics, problem_type
