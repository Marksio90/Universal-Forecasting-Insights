from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import os, json, time, pathlib, warnings
import pandas as pd

# === NAZWA_SEKCJI === IMPORTY KOMPATYBILNE
# Loader danych – preferuj nasz smart_read, fallback na pandas
try:
    from src.utils.helpers import smart_read  # PRO loader
except Exception:  # pragma: no cover
    smart_read = None  # type: ignore

# AutoML Fusion (Twoja funkcja)
from backend.automl_fusion import train_fusion

# Zapis modelu – preferuj rejestr PRO, fallback na prostą wersję
try:
    from backend.persistence.model_registry import save_model  # PRO+++
except Exception:  # pragma: no cover
    try:
        from backend.persistence.model_store import save_model  # wcześniejsza nazwa
    except Exception:  # pragma: no cover
        from backend.model_io import save_model  # Twój minimalny zapis

# Raportowanie – użyj tego, co masz w repo
try:
    from backend.reports.report_builder import build_html_summary, build_pdf_from_html
except Exception:  # pragma: no cover
    # Alternatywnie nasz PRO pdf_builder (jeśli używasz nowszego modułu)
    from backend.reports.pdf_builder import build_html as build_html_summary  # type: ignore
    from backend.reports.pdf_builder import build_pdf as build_pdf_from_html   # type: ignore

# MLflow (opcjonalnie)
try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore

warnings.filterwarnings("ignore")


# === NAZWA_SEKCJI === DANE WYJŚCIOWE
@dataclass
class TrainJobResult:
    model_path: str
    problem_type: str
    metric_name: str
    best_score: float
    html_report: Optional[str] = None
    pdf_report: Optional[str] = None
    mlflow_run_id: Optional[str] = None


# === NAZWA_SEKCJI === POMOCNICZE
def _read_dataframe(csv_path: str) -> pd.DataFrame:
    """Czytaj dane z pliku. Jeśli dostępny `smart_read`, użyj go; inaczej pandas."""
    if smart_read is not None:
        df = smart_read(csv_path)  # obsłuży CSV/Parquet/Excel/JSON itd.
    else:
        df = pd.read_csv(csv_path)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Źródło danych puste lub niepoprawne.")
    return df


def _ensure_dirs() -> None:
    for d in ("reports", "artifacts", "models"):
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def _log_to_mlflow_safe(params: Dict[str, Any], metrics: Dict[str, float], artifacts: Dict[str, str]) -> Optional[str]:
    """Bezpieczne logowanie do MLflow (jeśli dostępne). Zwraca run_id lub None."""
    if mlflow is None:
        return None
    try:
        # Użyj aktywnego runu z train_fusion jeśli istnieje, inaczej stwórz nowy
        active = mlflow.active_run()
        ctx = mlflow.start_run(nested=True) if active is not None else mlflow.start_run()
        run_id = (ctx or active).info.run_id  # type: ignore
        # Parametry
        for k, v in (params or {}).items():
            try: mlflow.log_param(k, v)
            except Exception: pass
        # Metryki
        for k, v in (metrics or {}).items():
            try: mlflow.log_metric(k, float(v))
            except Exception: pass
        # Artefakty
        for name, path in (artifacts or {}).items():
            if path and os.path.exists(path):
                try: mlflow.log_artifact(path, artifact_path=name)
                except Exception: pass
        # Jeśli sami otwieraliśmy run – zamkniemy go
        if active is None:
            mlflow.end_run()
        return run_id
    except Exception:
        return None


def _save_drift_reference(df: pd.DataFrame) -> None:
    """Zapisz referencję do monitoringu driftu (lekki CSV z numeric)."""
    try:
        ref = df.select_dtypes(include="number")
        if not ref.empty:
            out = pathlib.Path("reports") / "tmp_train.csv"
            ref.to_csv(out, index=False)
    except Exception:
        pass


# === NAZWA_SEKCJI === GŁÓWNA FUNKCJA (API KOMPATYBILNE)
def train_job(csv_path: str, target: str, trials: int = 30) -> str:
    """
    PRO+++ Training Orchestrator:
    - wczytuje dane (smart_read/pandas),
    - uruchamia AutoML Fusion (Optuna + XGB/LGBM/CAT) z `trials`,
    - zapisuje model wersjonowany (`models/<name>_<ts>.joblib` + meta, jeśli używasz PRO registry),
    - generuje HTML & PDF raport (EDA + metryka z AutoML),
    - loguje wszystko do MLflow (jeśli dostępne),
    - zapisuje referencję do drift monitoringu (`reports/tmp_train.csv`).

    Zwraca: ścieżkę do zapisanego modelu (zachowuje Twój dotychczasowy kontrakt).
    """
    t0 = time.time()
    _ensure_dirs()

    # 1) Dane
    df = _read_dataframe(csv_path)
    if target not in df.columns:
        raise KeyError(f"Brak kolumny target '{target}' w danych. Dostępne: {list(df.columns)[:20]}{' ...' if df.shape[1]>20 else ''}")

    # 2) Trening (AutoML Fusion)
    res = train_fusion(df, target=target, trials=trials)  # -> FusionResult

    # 3) Zapis modelu (nazewnictwo: FUSION_<type>)
    model_name = f"FUSION_{res.problem_type}"
    model_path = save_model(res.model, model_name)

    # 4) Artefakty: raport HTML + PDF
    #    (build_html_summary przyjmuje df + dict metryk; PDF z HTML)
    html_path = build_html_summary(df, {res.metric_name: res.best_score})
    try:
        pdf_path = build_pdf_from_html(html_path)
    except Exception:
        pdf_path = None  # jeżeli brak zależności do PDF – nie przerywaj

    # 5) Drift reference (numeric subset)
    _save_drift_reference(df)

    # 6) MLflow (bezpiecznie, jeśli dostępne)
    run_id = _log_to_mlflow_safe(
        params={
            "target": target,
            "trials": trials,
            "problem_type": res.problem_type,
            "top_models": ",".join([n for n, _ in (res.leaderboard or [])]),
        },
        metrics={res.metric_name: float(res.best_score)},
        artifacts={
            "reports/html": html_path,
            "reports/pdf": pdf_path or "",
            "models": model_path,
        }
    )

    # 7) Log pomocniczy (JSON) – ułatwia parsowanie w CI/workerze
    summary = TrainJobResult(
        model_path=model_path,
        problem_type=res.problem_type,
        metric_name=res.metric_name,
        best_score=float(res.best_score),
        html_report=html_path,
        pdf_report=pdf_path,
        mlflow_run_id=run_id,
    )
    try:
        pathlib.Path("artifacts").mkdir(exist_ok=True, parents=True)
        with open("artifacts/last_train_summary.json", "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # (opcjonalnie) wydruk do stdout
    elapsed = time.time() - t0
    print(f"[train_job] OK in {elapsed:.2f}s | model={model_path} | {res.metric_name}={res.best_score:.4f}")

    # === KOMPAT: zwróć tylko ścieżkę modelu (jak w Twojej wersji) ===
    return model_path
