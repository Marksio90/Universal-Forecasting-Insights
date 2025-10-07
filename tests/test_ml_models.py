# tests/test_ml_models.py
from __future__ import annotations
import os
import math
import joblib
import pytest
import pandas as pd
import numpy as np


# =========================================================
# Helpers do danych syntetycznych
# =========================================================
def make_regression_df(n=220, noise=5.0, seed=42):
    rng = np.random.default_rng(seed)
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(2, 1.5, n)
    cat = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    y = 10 + 3.5 * X1 - 1.7 * X2 + (cat == "B") * 2.0 + rng.normal(0, noise, n)
    df = pd.DataFrame(
        {
            "f1": X1,
            "f2": X2,
            "category": cat,
            "y": y,
        }
    )
    return df


def make_classification_df(n=240, seed=42):
    rng = np.random.default_rng(seed)
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    linear = 0.8 * X1 - 1.2 * X2 + rng.normal(0, 0.5, n)
    # probit → klasy 0/1
    p = 1 / (1 + np.exp(-linear))
    y = (p > 0.5).astype(int)
    seg = rng.choice(["S1", "S2"], size=n)  # dodatkowa kategoria
    df = pd.DataFrame({"x1": X1, "x2": X2, "seg": seg, "target": y})
    return df


# =========================================================
# AutoML: regresja
# =========================================================
def test_train_automl_regression_saves_and_reports(tmp_path, monkeypatch):
    # przestaw katalog modeli/registry -> tmp
    import src.ml_models.automl_pipeline as ap
    monkeypatch.setattr(ap, "MODELS_DIR", tmp_path, raising=True)
    monkeypatch.setattr(ap, "REGISTRY", tmp_path / "registry.json", raising=True)
    (tmp_path).mkdir(parents=True, exist_ok=True)

    df = make_regression_df()
    model, metrics, ptype = ap.train_automl(df, target="y", random_state=42)

    assert ptype == "regression"
    # podstawowe metryki obecne
    assert set(metrics.keys()) >= {"rmse", "mae", "r2", "mape"}
    # sanity bounds
    assert metrics["rmse"] >= 0
    assert -1.0 <= metrics["r2"] <= 1.0

    # model działa
    y_hat = model.predict(df.drop(columns=["y"]))
    assert len(y_hat) == len(df)

    # wpis w registry został dodany
    import src.ml_models.model_registry as reg
    # ustaw ten sam rejestr w module registry
    monkeypatch.setattr(reg, "MODELS_DIR", tmp_path, raising=True)
    monkeypatch.setattr(reg, "REGISTRY", tmp_path / "registry.json", raising=True)

    items = reg.list_models()
    assert len(items) >= 1
    # plik istnieje
    p = tmp_path / items[-1]["path"]
    assert p.exists()


# =========================================================
# AutoML: klasyfikacja
# =========================================================
def test_train_automl_classification_metrics(tmp_path, monkeypatch):
    import src.ml_models.automl_pipeline as ap
    monkeypatch.setattr(ap, "MODELS_DIR", tmp_path, raising=True)
    monkeypatch.setattr(ap, "REGISTRY", tmp_path / "registry.json", raising=True)
    (tmp_path).mkdir(parents=True, exist_ok=True)

    df = make_classification_df()
    model, metrics, ptype = ap.train_automl(df, target="target", random_state=7)

    assert ptype == "classification"
    # obecność metryk
    assert "accuracy" in metrics and "f1_weighted" in metrics
    # sanity zakresy
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_weighted"] <= 1.0

    # predict i (opcjonalnie) predict_proba działają
    X = df.drop(columns=["target"])
    y_pred = model.predict(X)
    assert len(y_pred) == len(df)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        assert proba.shape[0] == len(df)


# =========================================================
# Forecasting (Prophet) – skip jeśli brak
# =========================================================
@pytest.mark.skipif(pytest.importorskip("prophet", reason="prophet not installed") is None, reason="prophet not installed")
def test_forecasting_prophet_basic():
    from src.ml_models.forecasting import forecast

    # dzienna seria z trendem i sezonowością tygodniową
    idx = pd.date_range("2024-01-01", periods=90, freq="D")
    rng = np.random.default_rng(42)
    y = 50 + 0.2 * np.arange(len(idx)) + 5 * np.sin(2 * np.pi * (idx.dayofyear / 7.0)) + rng.normal(0, 1.0, len(idx))
    df = pd.DataFrame({"ds": idx, "y": y})

    model, fcst = forecast(df, target="y", horizon=14)
    assert {"ds", "yhat", "yhat_lower", "yhat_upper"} <= set(fcst.columns)
    assert len(fcst) >= len(df)  # zawiera historię + przyszłość
    # metadane
    meta = fcst.attrs.get("forecast_meta", {})
    assert meta.get("horizon") == 14


@pytest.mark.skipif(pytest.importorskip("prophet", reason="prophet not installed") is None, reason="prophet not installed")
def test_time_series_fit_prophet_interface():
    # Ten test utrzymuje zgodność API fit_prophet(df, target, horizon)
    from src.ml_models.time_series import fit_prophet

    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    y = np.linspace(10, 30, len(idx)) + np.random.default_rng(0).normal(0, 0.5, len(idx))
    df = pd.DataFrame({"date": idx, "y": y})

    m, fcst = fit_prophet(df, target="y", horizon=7)
    assert {"ds", "yhat", "yhat_lower", "yhat_upper"} <= set(fcst.columns)
    assert len(fcst) == 7  # w Twojej implementacji: include_history=False


# =========================================================
# Anomaly detection
# =========================================================
def test_anomaly_detection_flags_and_meta():
    from src.ml_models.anomaly_detection import detect_anomalies

    rng = np.random.default_rng(123)
    normal = pd.DataFrame({"a": rng.normal(0, 1, 400), "b": rng.normal(0, 1, 400)})
    outliers = pd.DataFrame({"a": rng.normal(8, 0.2, 12), "b": rng.normal(8, 0.2, 12)})
    df = pd.concat([normal, outliers], ignore_index=True)

    scored = detect_anomalies(df, contamination=0.02, method="iforest", scale=False)
    assert "_is_anomaly" in scored.columns and "_anomaly_score" in scored.columns
    # spodziewamy się wykryć przynajmniej część z 12 outlierów
    assert int(scored["_is_anomaly"].sum()) >= 5
    # metadane
    meta = getattr(scored, "attrs", {}).get("anomaly_meta")
    assert isinstance(meta, dict) and meta.get("method") in {"iforest", "lof", "elliptic", "ocsvm"}


# =========================================================
# Model registry (API PRO) – opcjonalnie
# =========================================================
def test_model_registry_pro_roundtrip(tmp_path, monkeypatch):
    import src.ml_models.model_registry as reg

    # jeżeli moduł nie ma PRO-API (register_model), pomiń test
    if not hasattr(reg, "register_model"):
        pytest.skip("Legacy registry without PRO API")

    # ustaw katalogi na tmp
    monkeypatch.setattr(reg, "MODELS_DIR", tmp_path, raising=True)
    monkeypatch.setattr(reg, "REGISTRY", tmp_path / "registry.json", raising=True)
    tmp_path.mkdir(parents=True, exist_ok=True)

    # utwórz fikcyjny model
    mp = tmp_path / "dummy.joblib"
    joblib.dump({"hello": "world"}, mp)

    entry = reg.register_model(
        model_path=mp,
        target="y",
        problem_type="regression",
        metrics={"rmse": 1.23, "r2": 0.9},
        columns=["f1", "f2"],
        best_estimator="rf",
        tags=["test"],
        extra={"note": "unit"},
    )
    assert entry["path"] == mp.name

    items = reg.list_models()
    assert any(it["path"] == mp.name for it in items)

    loaded = reg.load_model(entry["id"])
    assert isinstance(loaded, dict) and loaded.get("hello") == "world"

    # get_best_model
    best = reg.get_best_model(problem_type="regression")
    assert best is not None and "metrics" in best

    # delete
    ok = reg.delete_model(entry["id"], remove_file=True)
    assert ok is True
    assert not (tmp_path / mp.name).exists()
