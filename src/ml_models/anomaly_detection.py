from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

# --- logger (bezpieczny, bez duplikatów handlerów) ---
_LOGGER = logging.getLogger("anomalies")
if not _LOGGER.handlers:
    _LOGGER.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    _LOGGER.addHandler(_h)
    _LOGGER.propagate = False


def _mk_empty_result(df: pd.DataFrame, reason: str) -> pd.DataFrame:
    res = df.copy()
    res["_anomaly_score"] = np.nan
    res["_is_anomaly"] = pd.Series(np.zeros(len(df), dtype="int8"), index=df.index).astype("Int8")
    res.attrs["anomaly_meta"] = {"error": reason, "n_rows": int(len(df))}
    _LOGGER.warning("anomaly: %s", reason)
    return res


def _minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    mn, mx = np.nanmin(x), np.nanmax(x)
    rng = mx - mn
    if rng <= 1e-12:
        return np.zeros_like(x)
    return (x - mn) / rng


def _sanitize_params(n_rows: int,
                     contamination: float | str,
                     method: str,
                     n_neighbors: int,
                     nu: float) -> Tuple[float | str, int, float, list[str]]:
    """Koryguje parametry do bezpiecznych zakresów; zwraca (contam, n_neighbors_eff, nu_eff, warnings)."""
    warns: list[str] = []
    m = method.lower().strip()

    # contamination
    contam: float | str = contamination
    if isinstance(contam, float):
        if contam <= 0 or contam >= 0.5:
            warns.append(f"contamination={contam} poza (0,0.5); skorygowałem do 0.05")
            contam = 0.05
    else:
        if m in ("elliptic", "iforest") and contam == "auto":
            # ok dla IF (>=1.4), dla EE brak 'auto' – użyjemy 0.1
            if m == "elliptic":
                warns.append("EllipticEnvelope nie wspiera contamination='auto' → używam 0.1")
                contam = 0.1

    # n_neighbors
    nn_eff = max(2, min(n_neighbors, max(2, n_rows - 1)))
    if m == "lof" and nn_eff != n_neighbors:
        warns.append(f"n_neighbors skorygowany do {nn_eff} (n_rows={n_rows})")

    # nu
    nu_eff = nu
    if m == "ocsvm" and (nu <= 0 or nu >= 1):
        warns.append(f"nu={nu} poza (0,1); skorygowałem do 0.1")
        nu_eff = 0.1

    return contam, nn_eff, nu_eff, warns


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float | str = 0.05,
    method: str = "iforest",            # "iforest" | "lof" | "elliptic" | "ocsvm"
    scale: bool = False,                # skalowanie RobustScaler (przydatne dla LOF/OCSVM)
    random_state: int = 42,
    n_neighbors: int = 20,              # dla LOF
    assume_centered: bool = False,      # dla EllipticEnvelope
    nu: float = 0.1,                    # dla OCSVM
) -> pd.DataFrame:
    """
    Wykrywa anomalie w danych (kolumny numeryczne) czterema klasykami:
    IsolationForest | LOF | EllipticEnvelope | OneClassSVM.

    Zwraca kopię DataFrame z kolumnami:
      - `_anomaly_score`  (float; **wyższy = bardziej normalny**;
         LOF: min-max do [0,1]; inne: surowy decision_function)
      - `_is_anomaly`     (0/1; Int8)

    Dodatkowo zapisuje metadane w `res.attrs["anomaly_meta"]`:
      - method, contamination, n_rows, n_num_cols, dropped_constant_cols,
        n_anomalies, feature_influence (ranking Spearmana), scaled, imputer,
        params (dla danej metody), score_range, warnings, fallback_used.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return _mk_empty_result(pd.DataFrame(), "invalid_dataframe")

    if df.empty:
        return _mk_empty_result(df, "empty_dataframe")

    # --- kolumny numeryczne + sanity ---
    num = df.select_dtypes(include=np.number).copy()
    if num.empty:
        return _mk_empty_result(df, "no_numeric_columns")

    # Inf → NaN (potem imputacja)
    num = num.replace([np.inf, -np.inf], np.nan)

    # Usuń kolumny stałe
    constant_cols = [c for c in num.columns if num[c].nunique(dropna=True) <= 1]
    if constant_cols:
        num = num.drop(columns=constant_cols)

    if num.empty:
        return _mk_empty_result(df, "only_constant_columns")

    n_rows = len(num)
    if n_rows < 5:
        # Zbyt mało obserwacji dla sensownej detekcji
        return _mk_empty_result(df, f"too_few_rows({n_rows})")

    # --- imputacja + skalowanie ---
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(num)

    use_scaler = bool(scale or method.lower().strip() in ("lof", "ocsvm"))
    scaler = None
    if use_scaler:
        scaler = RobustScaler()
        X = scaler.fit_transform(X)

    # --- param sanity & warnings ---
    contam_eff, nn_eff, nu_eff, warns = _sanitize_params(n_rows, contamination, method, n_neighbors, nu)

    # --- trenowanie modeli + fallback ---
    m = method.lower().strip()
    decision_scores: np.ndarray | None = None
    is_anom: np.ndarray | None = None
    fallback_used = None

    def _fit_iforest() -> Tuple[np.ndarray, np.ndarray]:
        clf = IsolationForest(
            contamination=contam_eff,  # float lub 'auto'
            random_state=random_state,
            n_estimators=300,
            max_samples="auto",
            n_jobs=-1,
            bootstrap=False,
        )
        clf.fit(X)
        scores = clf.decision_function(X)  # większy = normalny
        labels = clf.predict(X)            # -1 anomalia, 1 normal
        return scores, (labels == -1).astype(int)

    try:
        if m == "iforest":
            decision_scores, is_anom = _fit_iforest()

        elif m == "lof":
            lof = LocalOutlierFactor(
                n_neighbors=nn_eff,
                contamination=contam_eff if isinstance(contam_eff, float) else "auto",
                novelty=False,
                n_jobs=-1,
            )
            labels = lof.fit_predict(X)  # -1 anomalia
            is_anom = (labels == -1).astype(int)
            nof = lof.negative_outlier_factor_.astype(float)
            decision_scores = _minmax01(nof)  # większy = normalny (skalowane)

        elif m == "elliptic":
            ee = EllipticEnvelope(
                contamination=contam_eff if isinstance(contam_eff, float) else 0.1,
                support_fraction=None,
                assume_centered=assume_centered,
                random_state=random_state,
            )
            ee.fit(X)
            decision_scores = ee.decision_function(X)
            labels = ee.predict(X)  # -1 anomalia
            is_anom = (labels == -1).astype(int)

        elif m == "ocsvm":
            oc = OneClassSVM(kernel="rbf", nu=nu_eff, gamma="scale")
            oc.fit(X)
            decision_scores = oc.decision_function(X)
            labels = oc.predict(X)
            is_anom = (labels == -1).astype(int)

        else:
            raise ValueError(f"Nieznana metoda: {method}. Użyj: 'iforest' | 'lof' | 'elliptic' | 'ocsvm'.")

    except Exception as e:
        # Fallback do IsolationForest (najbardziej odporny)
        _LOGGER.exception("anomaly: metoda '%s' nie powiodła się, fallback do IsolationForest", method)
        warns.append(f"fallback_from_{m}_to_iforest: {e}")
        decision_scores, is_anom = _fit_iforest()
        fallback_used = "iforest"

    # --- złożenie wyniku ---
    res = df.copy()
    res["_anomaly_score"] = pd.Series(decision_scores, index=num.index)
    res["_is_anomaly"] = pd.Series(is_anom, index=num.index).astype("Int8")

    # --- wpływ cech (Spearman), na próbce gdy duże dane ---
    try:
        view = num.copy()
        view["_score"] = res["_anomaly_score"].values
        if len(view) > 50_000:
            view = view.sample(50_000, random_state=42)
        corr = view.corr(method="spearman")["_score"].drop("_score", errors="ignore").abs().sort_values(ascending=False)
        feature_influence: Dict[str, float] = corr.to_dict()
    except Exception:
        feature_influence = {}

    # --- meta ---
    score_arr = res["_anomaly_score"].to_numpy(dtype=float)
    score_range = (
        float(np.nanmin(score_arr)) if score_arr.size else np.nan,
        float(np.nanmax(score_arr)) if score_arr.size else np.nan,
    )
    meta: Dict[str, Any] = {
        "method": m if fallback_used is None else f"{m} (fallback→iforest)",
        "contamination": contamination,
        "n_rows": int(len(df)),
        "n_num_cols": int(num.shape[1]),
        "dropped_constant_cols": constant_cols,
        "n_anomalies": int(int(res["_is_anomaly"].sum())),
        "feature_influence": feature_influence,  # dict: kolumna -> |rho|
        "scaled": use_scaler,
        "imputer": "median",
        "random_state": random_state,
        "score_range": score_range,
        "params": {
            "n_neighbors": nn_eff if m == "lof" else None,
            "assume_centered": assume_centered if m == "elliptic" else None,
            "nu": nu_eff if m == "ocsvm" else None,
        },
        "warnings": warns or None,
        "fallback_used": fallback_used,
    }
    res.attrs["anomaly_meta"] = meta

    return res
