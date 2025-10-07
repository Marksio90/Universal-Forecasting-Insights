from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer


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
    Wykrywa anomalie w danych (kolumny numeryczne).
    Zwraca kopię DataFrame z kolumnami:
      - `_anomaly_score`  (float; wyższy = normalniejszy dla IF/EE/OCSVM; dla LOF przeskalowane)
      - `_is_anomaly`     (0/1)

    Dodatkowo zapisuje metadane do `res.attrs["anomaly_meta"]`:
      - method, contamination, n_rows, n_num_cols, n_anomalies, feature_influence (ranking)
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    # -------- Wybór + sanity-check kolumn --------
    num = df.select_dtypes(include=np.number).copy()
    if num.empty:
        # brak cech numerycznych — zwróć DataFrame z brakowymi kolumnami wynikowymi
        res = df.copy()
        res["_anomaly_score"] = np.nan
        res["_is_anomaly"] = 0
        res.attrs["anomaly_meta"] = {"error": "no_numeric_columns"}
        return res

    # Usuń kolumny stałe (zerowa wariancja)
    constant_cols = [c for c in num.columns if num[c].nunique(dropna=True) <= 1]
    if constant_cols:
        num = num.drop(columns=constant_cols)

    if num.empty:
        res = df.copy()
        res["_anomaly_score"] = np.nan
        res["_is_anomaly"] = 0
        res.attrs["anomaly_meta"] = {"error": "only_constant_columns"}
        return res

    # -------- Imputacja + (opcjonalnie) skalowanie --------
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(num)

    scaler = None
    if scale or method in ("lof", "ocsvm"):
        scaler = RobustScaler()
        X = scaler.fit_transform(X)

    # -------- Model --------
    method = method.lower().strip()
    clf = None
    decision_scores = None
    labels = None

    if method == "iforest":
        clf = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=300,
            max_samples="auto",
            n_jobs=-1,
            bootstrap=False,
        )
        clf.fit(X)
        # decision_function: wyższy = bardziej "normal"
        decision_scores = clf.decision_function(X)
        labels = clf.predict(X)  # -1 anomalie, 1 normal
        is_anom = (labels == -1).astype(int)

    elif method == "lof":
        # LOF nie ma .decision_function; użyjemy -negative_outlier_factor_ znormalizowanego do ~[0,1]
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination if isinstance(contamination, float) else "auto",
            novelty=False,  # klasyczny LOF tylko do fit_predict
            n_jobs=-1,
        )
        labels = lof.fit_predict(X)  # -1 anomalie
        is_anom = (labels == -1).astype(int)
        # Skala wyników: im mniejszy NOF, tym większa anomalia → przekształć na "im wyżej tym normalniejszy"
        nof = lof.negative_outlier_factor_
        # przeskaluj do [0, 1] (większy = bardziej normalny)
        min_nof, max_nof = float(np.min(nof)), float(np.max(nof))
        rng = max(max_nof - min_nof, 1e-12)
        decision_scores = (nof - min_nof) / rng

        clf = lof  # do metadanych

    elif method == "elliptic":
        ee = EllipticEnvelope(
            contamination=contamination if isinstance(contamination, float) else 0.1,
            support_fraction=None,
            assume_centered=assume_centered,
            random_state=random_state,
        )
        ee.fit(X)
        # decision_function: wyższy = bardziej "normal"
        decision_scores = ee.decision_function(X)
        labels = ee.predict(X)  # -1 anomalie, 1 normal
        is_anom = (labels == -1).astype(int)
        clf = ee

    elif method == "ocsvm":
        oc = OneClassSVM(
            kernel="rbf",
            nu=nu,    # frakcja outlierów upper-bound
            gamma="scale",
        )
        oc.fit(X)
        decision_scores = oc.decision_function(X)  # wyższy = normalniejszy
        labels = oc.predict(X)  # -1 anomalie, 1 normal
        is_anom = (labels == -1).astype(int)
        clf = oc

    else:
        raise ValueError(f"Nieznana metoda: {method}. Użyj: 'iforest' | 'lof' | 'elliptic' | 'ocsvm'.")

    # -------- Złożenie wyniku --------
    res = df.copy()
    res["_anomaly_score"] = pd.Series(decision_scores, index=num.index)
    res["_is_anomaly"] = pd.Series(is_anom, index=num.index).astype("Int8")

    # -------- Ranking cech (wpływ ≈ korelacja Spearmana ze score) --------
    try:
        # Korelacja tylko dla numerycznych, na próbce dla dużych zbiorów
        view = num.copy()
        view["_score"] = res["_anomaly_score"].values
        if len(view) > 50000:
            view = view.sample(50000, random_state=42)
        corr = view.corr(method="spearman")["_score"].drop("_score", errors="ignore").abs().sort_values(ascending=False)
        feature_influence = corr.to_dict()
    except Exception:
        feature_influence = None

    # -------- Metadane --------
    meta = {
        "method": method,
        "contamination": contamination,
        "n_rows": int(len(df)),
        "n_num_cols": int(num.shape[1]),
        "dropped_constant_cols": constant_cols,
        "n_anomalies": int(int(res["_is_anomaly"].sum())),
        "feature_influence": feature_influence,  # dict: kolumna -> |rho|
        "scaled": bool(scale or method in ("lof", "ocsvm")),
        "imputer": "median",
        "random_state": random_state,
        "params": {
            "n_neighbors": n_neighbors if method == "lof" else None,
            "assume_centered": assume_centered if method == "elliptic" else None,
            "nu": nu if method == "ocsvm" else None,
        },
    }
    res.attrs["anomaly_meta"] = meta

    return res
