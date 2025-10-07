"""
Moduł Reports & Export PRO++ - Zaawansowany eksport i generowanie raportów.

Funkcjonalności:
- Generowanie raportów HTML z pełnym kontekstem
- Export ZIP z wieloma artefaktami
- Anonimizacja danych (hash/drop)
- Sampling dla dużych zbiorów
- Manifest z checksumami SHA-256
- Atomic writes dla bezpieczeństwa
- Historia operacji (JSONL)
- Compact metadata dla wydajności
- Multi-format CSV (separator, encoding)
- Walidacja bezpieczeństwa
"""

from __future__ import annotations

import io
import os
import json
import time
import pathlib
import zipfile
import logging
import hashlib
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Literal, Tuple, Callable

import pandas as pd
import streamlit as st

from src.ai_engine.report_generator import build_report_html

# ========================================================================================
# KONFIGURACJA
# ========================================================================================

TITLE = "📋 Reports & Export — PRO++"

# Ścieżki i limity
BASE_EXPORT_DIR = pathlib.Path("data/exports")
MAX_PREVIEW_ROWS = 500
MAX_EXPORT_ROWS = 1_000_000
DEFAULT_SAMPLE_ROWS = 100_000
MAX_ZIP_SIZE_MB = 500
MAX_CSV_SIZE_MB = 250

# Meta
META_TRUNCATE_CHARS = 50_000
HISTORY_KEY = "reports_history"

# CSV defaults
CSV_DEFAULTS = {
    "index": False,
    "sep": ",",
    "encoding": "utf-8",
    "na_rep": "",
    "lineterminator": "\n"
}

# ========================================================================================
# LOGGING
# ========================================================================================

def get_logger(name: str = "reports", level: int = logging.INFO) -> logging.Logger:
    """
    Konfiguruje i zwraca logger.
    
    Args:
        name: Nazwa loggera
        level: Poziom logowania
        
    Returns:
        Skonfigurowany logger
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
    
    return logger


LOGGER = get_logger()


# ========================================================================================
# DECORATORS
# ========================================================================================

def safe_operation(default: Any = None, log_error: bool = True):
    """
    Dekorator łapiący wyjątki i zwracający domyślną wartość.
    
    Args:
        default: Wartość zwracana przy błędzie
        log_error: Czy logować błąd
    """
    def wrapper(func: Callable) -> Callable:
        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    LOGGER.exception(f"Operation failed: {func.__name__}")
                    st.error(f"❌ Błąd w {func.__name__}: {e}")
                return default
        return inner
    return wrapper


# ========================================================================================
# DATACLASSES
# ========================================================================================

@dataclass(frozen=True)
class ReportOptions:
    """Opcje generowania raportu."""
    include_data_dictionary: bool = True
    include_df_preview: bool = False
    include_kpis: bool = True
    include_tags: bool = True
    include_forecast: bool = True
    include_anomalies: bool = True
    compact_meta: bool = True
    sample_export: bool = False
    sample_rows: int = DEFAULT_SAMPLE_ROWS
    anonymize: bool = False
    anonymize_mode: Literal["hash", "drop"] = "hash"
    anonymize_columns: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExportArtifact:
    """Artefakt eksportu."""
    path: str
    kind: Literal["html", "zip", "csv", "json", "jsonl", "manifest"]
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None


@dataclass(frozen=True)
class ExportResult:
    """Wynik operacji eksportu."""
    ok: bool
    message: str
    artifacts: List[ExportArtifact]
    run_id: Optional[str] = None


@dataclass(frozen=True)
class ReportContext:
    """Kontekst raportu."""
    title: str
    subtitle: str
    run_meta: Dict[str, Any]
    metrics: Dict[str, Any]
    notes: str
    kpis: Optional[List[Dict[str, Any]]]
    tags: Optional[List[str]]
    insights: Optional[Any]
    recommendations: Optional[Any]
    forecast: Optional[Dict[str, Any]]
    anomalies: Optional[Dict[str, Any]]
    data_dictionary: Optional[Dict[str, Any]]
    df_preview: Optional[Dict[str, Any]]
    model_card: Optional[Dict[str, Any]]


@dataclass(frozen=True)
class Manifest:
    """Manifest eksportu."""
    run_id: str
    created_at: str
    artifacts: List[Dict[str, Any]]
    options: Dict[str, Any]
    context_meta: Dict[str, Any]


# ========================================================================================
# PROGRESS HELPER
# ========================================================================================

@contextmanager
def staged_progress(stages: List[Tuple[str, int]]):
    """
    Context manager dla progress bar z etapami.
    
    Args:
        stages: Lista tuple (nazwa_etapu, procent)
        
    Yields:
        Funkcja step(nazwa) do aktualizacji progress
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    stage_map = {name: pct for name, pct in stages}
    
    def step(name: str) -> None:
        status_text.text(f"**{name}...**")
        progress_bar.progress(stage_map.get(name, 0))
    
    try:
        yield step
    finally:
        status_text.empty()
        progress_bar.progress(100)
        time.sleep(0.1)  # Krótka pauza dla UX
        progress_bar.empty()


# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def now_timestamp() -> str:
    """Zwraca aktualny timestamp."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def make_run_id(prefix: str = "report") -> str:
    """
    Generuje unikalny ID dla runu.
    
    Args:
        prefix: Prefiks ID
        
    Returns:
        Unikalny run ID
    """
    base = time.strftime("%Y%m%d-%H%M%S")
    salt = hashlib.sha256(str(time.time()).encode()).hexdigest()[:6]
    return f"{prefix}-{base}-{salt}"


def compute_sha256(data: bytes) -> str:
    """
    Oblicza SHA-256 hash.
    
    Args:
        data: Dane binarne
        
    Returns:
        Hash hex
    """
    return hashlib.sha256(data).hexdigest()


def validate_path(path: pathlib.Path, base_dir: pathlib.Path = BASE_EXPORT_DIR) -> pathlib.Path:
    """
    Waliduje ścieżkę (security - prevent path traversal).
    
    Args:
        path: Ścieżka do walidacji
        base_dir: Bazowy katalog
        
    Returns:
        Zwalidowana ścieżka
        
    Raises:
        ValueError: Jeśli ścieżka jest nieprawidłowa
    """
    # Resolve to absolute path
    resolved = path.resolve()
    base_resolved = base_dir.resolve()
    
    # Check if path is within base_dir
    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        raise ValueError(
            f"Invalid path: {path} is outside base directory {base_dir}"
        )
    
    return resolved


def atomic_write(path: pathlib.Path, data: bytes) -> None:
    """
    Atomowy zapis danych do pliku.
    
    Args:
        path: Ścieżka docelowa
        data: Dane binarne
    """
    # Walidacja ścieżki
    path = validate_path(path)
    
    # Upewnij się że katalog istnieje
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Atomic write przez temp file
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        delete=False,
        mode='wb'
    ) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    
    # Replace atomically
    os.replace(tmp_name, path)
    
    LOGGER.info(f"Atomically wrote {len(data)} bytes to {path}")


def write_text_atomic(path: pathlib.Path, text: str, encoding: str = "utf-8") -> None:
    """
    Atomowy zapis tekstu.
    
    Args:
        path: Ścieżka docelowa
        text: Tekst do zapisu
        encoding: Kodowanie
    """
    atomic_write(path, text.encode(encoding))


# ========================================================================================
# DATAFRAME VALIDATION
# ========================================================================================

def validate_dataframe(df: Optional[pd.DataFrame]) -> List[str]:
    """
    Waliduje DataFrame przed eksportem.
    
    Args:
        df: DataFrame do walidacji
        
    Returns:
        Lista komunikatów o problemach (pusta = OK)
    """
    issues: List[str] = []
    
    if df is None:
        issues.append("Brak danych (df=None)")
        return issues
    
    if not isinstance(df, pd.DataFrame):
        issues.append(f"Obiekt nie jest DataFrame (typ: {type(df)})")
        return issues
    
    rows, cols = df.shape
    
    if rows == 0:
        issues.append("DataFrame jest pusty (0 wierszy)")
    
    if rows > MAX_EXPORT_ROWS:
        issues.append(
            f"Za dużo wierszy ({rows:,} > {MAX_EXPORT_ROWS:,}). "
            "Użyj 'Sample export'"
        )
    
    if cols == 0:
        issues.append("DataFrame nie ma kolumn")
    
    # Check memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    if memory_mb > 1000:  # 1GB
        issues.append(
            f"DataFrame zużywa dużo pamięci ({memory_mb:.0f} MB). "
            "Rozważ sampling"
        )
    
    return issues


@st.cache_data(show_spinner=False, ttl=600)
def df_to_preview_dict(
    df: pd.DataFrame,
    max_rows: int = MAX_PREVIEW_ROWS
) -> Dict[str, Any]:
    """
    Konwertuje DataFrame do słownika dla preview (cachowane).
    
    Args:
        df: DataFrame
        max_rows: Maksymalna liczba wierszy
        
    Returns:
        Słownik z kolumnami i wierszami
    """
    df_preview = df.head(max_rows).copy()
    
    return {
        "columns": [str(col) for col in df_preview.columns],
        "rows": df_preview.astype(object).to_dict(orient="records")
    }


# ========================================================================================
# KPI & DATA DICTIONARY
# ========================================================================================

@st.cache_data(show_spinner=False, ttl=600)
def build_kpis(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Buduje KPI danych (cachowane).
    
    Args:
        df: DataFrame
        
    Returns:
        Lista KPI
    """
    if df.empty:
        return []
    
    # Metryki
    missing_count = df.isna().sum().sum()
    total_cells = df.size
    missing_pct = (missing_count / total_cells * 100.0) if total_cells > 0 else 0.0
    
    dupes = int(df.duplicated().sum())
    
    return [
        {
            "label": "Wiersze",
            "value": f"{len(df):,}",
            "status": "ok"
        },
        {
            "label": "Kolumny",
            "value": f"{df.shape[1]:,}",
            "status": "ok"
        },
        {
            "label": "Braki",
            "value": f"{missing_pct:.2f}%",
            "status": "warn" if missing_pct > 1.0 else "ok"
        },
        {
            "label": "Duplikaty",
            "value": f"{dupes:,}",
            "status": "warn" if dupes > 0 else "ok"
        }
    ]


@st.cache_data(show_spinner=False, ttl=600)
def build_data_dictionary(
    df: pd.DataFrame,
    top_k: int = 30
) -> pd.DataFrame:
    """
    Buduje słownik danych (cachowane).
    
    Args:
        df: DataFrame
        top_k: Liczba top kolumn
        
    Returns:
        DataFrame ze słownikiem
    """
    if df.empty:
        return pd.DataFrame()
    
    rows = []
    
    for col in df.columns:
        series = df[col]
        
        # Przykładowa wartość
        example = ""
        try:
            non_na = series.dropna()
            if not non_na.empty:
                example = str(non_na.iloc[0])[:60]
        except Exception:
            example = ""
        
        rows.append({
            "column": str(col),
            "dtype": str(series.dtype),
            "missing_pct": round(series.isna().mean() * 100.0, 2),
            "nunique": int(series.nunique(dropna=True)),
            "example": example
        })
    
    dict_df = pd.DataFrame(rows)
    dict_df = dict_df.sort_values(
        ["missing_pct", "nunique"],
        ascending=[False, True]
    ).head(top_k)
    
    return dict_df


def make_tags(
    opts: ReportOptions,
    df: Optional[pd.DataFrame],
    goal: Optional[str],
    problem_type: Optional[str],
    target: Optional[str]
) -> Optional[List[str]]:
    """
    Generuje tagi dla raportu.
    
    Args:
        opts: Opcje raportu
        df: DataFrame
        goal: Cel biznesowy
        problem_type: Typ problemu
        target: Zmienna celu
        
    Returns:
        Lista tagów lub None
    """
    if not opts.include_tags:
        return None
    
    tags: List[str] = []
    
    if goal:
        tags.append("goal")
    if problem_type:
        tags.append(problem_type)
    if target:
        tags.append(f"y:{target}")
    
    if isinstance(df, pd.DataFrame) and not df.empty:
        tags.append(f"{len(df)}x{df.shape[1]}")
    
    return tags if tags else None


def build_minimal_model_card(
    model: Any,
    problem_type: Optional[str],
    metrics: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Buduje minimalną kartę modelu.
    
    Args:
        model: Model ML
        problem_type: Typ problemu
        metrics: Metryki
        
    Returns:
        Słownik z kartą modelu
    """
    model_name = "Unknown"
    try:
        model_name = model.__class__.__name__
    except Exception:
        model_name = str(type(model))
    
    return {
        "name": model_name,
        "type": problem_type or "—",
        "version": time.strftime("%Y.%m.%d"),
        "dataset": "session_dataframe",
        "split": "—",
        "hparams": None,
        "metrics": metrics
    }


# ========================================================================================
# ANONYMIZATION & SAMPLING
# ========================================================================================

def anonymize_dataframe(
    df: pd.DataFrame,
    columns: Tuple[str, ...],
    mode: Literal["hash", "drop"]
) -> pd.DataFrame:
    """
    Anonimizuje DataFrame.
    
    Args:
        df: DataFrame
        columns: Kolumny do anonimizacji
        mode: Tryb (hash lub drop)
        
    Returns:
        Zanonimizowany DataFrame
    """
    if not columns:
        return df
    
    # Filtruj tylko istniejące kolumny
    valid_cols = [col for col in columns if col in df.columns]
    
    if not valid_cols:
        return df
    
    LOGGER.info(f"Anonymizing {len(valid_cols)} columns with mode={mode}")
    
    if mode == "drop":
        return df.drop(columns=valid_cols)
    
    # Hash mode
    df_anon = df.copy()
    
    for col in valid_cols:
        df_anon[col] = (
            df_anon[col]
            .astype(str)
            .apply(lambda v: compute_sha256(v.encode("utf-8")))
        )
    
    return df_anon


def sample_dataframe(
    df: pd.DataFrame,
    enable: bool,
    n_rows: int
) -> pd.DataFrame:
    """
    Próbkuje DataFrame.
    
    Args:
        df: DataFrame
        enable: Czy włączyć sampling
        n_rows: Liczba wierszy
        
    Returns:
        Sampelowany DataFrame
    """
    if not enable or len(df) <= n_rows:
        return df
    
    n_rows = max(1, int(n_rows))
    
    LOGGER.info(f"Sampling DataFrame: {len(df)} -> {n_rows} rows")
    
    return df.head(n_rows)


# ========================================================================================
# HISTORY
# ========================================================================================

def get_history() -> List[Dict[str, Any]]:
    """
    Pobiera historię z session state.
    
    Returns:
        Lista historii
    """
    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []
    
    return st.session_state[HISTORY_KEY]


def add_to_history(
    module: str,
    action: str,
    params: Dict[str, Any],
    artifacts: List[ExportArtifact],
    summary: str = ""
) -> None:
    """
    Dodaje wpis do historii.
    
    Args:
        module: Nazwa modułu
        action: Akcja
        params: Parametry
        artifacts: Artefakty
        summary: Podsumowanie
    """
    entry = {
        "timestamp": now_timestamp(),
        "module": module,
        "action": action,
        "params": params,
        "summary": summary,
        "artifacts": [asdict(a) for a in artifacts]
    }
    
    history = get_history()
    history.insert(0, entry)
    
    # Ogranicz do 20
    st.session_state[HISTORY_KEY] = history[:20]


def append_history_jsonl(run_dir: pathlib.Path, entry: Dict[str, Any]) -> None:
    """
    Dodaje wpis do pliku history.jsonl.
    
    Args:
        run_dir: Katalog runu
        entry: Wpis do zapisania
    """
    try:
        history_file = run_dir / "history.jsonl"
        line = json.dumps(entry, ensure_ascii=False)
        
        with open(history_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        
        LOGGER.info(f"Appended to history.jsonl: {history_file}")
        
    except Exception as e:
        LOGGER.error(f"Failed to append history: {e}")


# ========================================================================================
# CONTEXT BUILDING
# ========================================================================================

def compact_object(obj: Any, char_limit: int = META_TRUNCATE_CHARS) -> Any:
    """
    Ucina duże obiekty do rozsądnego rozmiaru.
    
    Args:
        obj: Obiekt do ucięcia
        char_limit: Limit znaków
        
    Returns:
        Ucięty obiekt lub oryginalny
    """
    try:
        json_str = json.dumps(obj, ensure_ascii=False)
        
        if len(json_str) <= char_limit:
            return obj
        
        # Ucięcie z obu stron
        half = char_limit // 2
        head = json_str[:half]
        tail = json_str[-half:]
        
        return f"{head}...<truncated>...{tail}"
        
    except Exception:
        # Fallback dla nietypowych obiektów
        str_repr = str(obj)
        if len(str_repr) <= char_limit:
            return obj
        
        return str_repr[:char_limit] + "...<truncated>"


def build_meta_payload(context: ReportContext, compact: bool) -> Dict[str, Any]:
    """
    Buduje payload meta z opcjonalnym compactem.
    
    Args:
        context: Kontekst raportu
        compact: Czy kompaktować
        
    Returns:
        Słownik z meta
    """
    payload = asdict(context)
    
    if not compact:
        return payload
    
    # Compact wybranych pól
    compact_fields = [
        "insights", "recommendations", "forecast", "anomalies",
        "data_dictionary", "df_preview", "metrics", "model_card"
    ]
    
    for field in compact_fields:
        if field in payload and payload[field] is not None:
            payload[field] = compact_object(payload[field], META_TRUNCATE_CHARS)
    
    return payload


@safe_operation(None)
def build_report_context(
    opts: ReportOptions,
    df: Optional[pd.DataFrame],
    goal: Optional[str],
    problem_type: Optional[str],
    target: Optional[str],
    model: Optional[Any],
    forecast_df: Optional[pd.DataFrame],
    anomalies_df: Optional[pd.DataFrame],
    automl_metrics: Optional[Dict[str, Any]],
    notes: str
) -> Optional[ReportContext]:
    """
    Buduje kontekst raportu.
    
    Args:
        opts: Opcje raportu
        df: DataFrame
        goal: Cel biznesowy
        problem_type: Typ problemu
        target: Zmienna celu
        model: Model ML
        forecast_df: Prognoza
        anomalies_df: Anomalie
        automl_metrics: Metryki AutoML
        notes: Notatki
        
    Returns:
        ReportContext lub None przy błędzie
    """
    # Metryki
    metrics: Dict[str, Any] = {}
    
    if problem_type:
        metrics["problem_type"] = problem_type
    if target:
        metrics["target"] = target
    if isinstance(automl_metrics, dict):
        metrics["automl"] = automl_metrics
    
    # KPIs
    kpis = None
    if opts.include_kpis and isinstance(df, pd.DataFrame) and not df.empty:
        kpis = build_kpis(df)
        if not kpis:
            kpis = None
    
    # Data dictionary
    dd_table = None
    if opts.include_data_dictionary and isinstance(df, pd.DataFrame) and not df.empty:
        dd_df = build_data_dictionary(df)
        if not dd_df.empty:
            dd_table = df_to_preview_dict(dd_df)
    
    # DataFrame preview
    df_preview = None
    if opts.include_df_preview and isinstance(df, pd.DataFrame) and not df.empty:
        df_preview = df_to_preview_dict(df.head(50))
    
    # Forecast
    forecast_blob = None
    if opts.include_forecast and isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
        forecast_blob = df_to_preview_dict(forecast_df)
    
    # Anomalies
    anomalies_blob = None
    if opts.include_anomalies and isinstance(anomalies_df, pd.DataFrame) and not anomalies_df.empty:
        anomalies_blob = df_to_preview_dict(anomalies_df)
    
    # Model card
    model_card = st.session_state.get("model_card")
    if model_card is None and model is not None:
        model_card = build_minimal_model_card(model, problem_type, automl_metrics)
    
    # Tags
    tags = make_tags(opts, df, goal, problem_type, target)
    
    return ReportContext(
        title="Raport Biznesowy",
        subtitle=goal or "",
        run_meta={"timestamp": time.strftime("%Y-%m-%d %H:%M")},
        metrics=metrics,
        notes=notes,
        kpis=kpis,
        tags=tags,
        insights=st.session_state.get("ai_top_insights"),
        recommendations=st.session_state.get("ai_recommendations"),
        forecast=forecast_blob,
        anomalies=anomalies_blob,
        data_dictionary=dd_table,
        df_preview=df_preview,
        model_card=model_card
    )


# ========================================================================================
# CSV HELPERS
# ========================================================================================

def dataframe_to_csv_bytes(df: pd.DataFrame, **csv_kwargs) -> bytes:
    """
    Konwertuje DataFrame do CSV bytes.
    
    Args:
        df: DataFrame
        **csv_kwargs: Argumenty dla to_csv
        
    Returns:
        CSV jako bytes
    """
    buffer = io.StringIO()
    merged_kwargs = {**CSV_DEFAULTS, **csv_kwargs}
    df.to_csv(buffer, **merged_kwargs)
    
    encoding = csv_kwargs.get("encoding", "utf-8")
    return buffer.getvalue().encode(encoding)


def estimate_csv_size(df: pd.DataFrame, sample_size: int = 10_000) -> int:
    """
    Szacuje rozmiar CSV w bytes.
    
    Args:
        df: DataFrame
        sample_size: Rozmiar próbki
        
    Returns:
        Szacowany rozmiar w bytes
    """
    if df.empty:
        return 0
    
    # Próbka
    n = len(df)
    sample = df.head(min(sample_size, n))
    
    # Rozmiar próbki
    sample_bytes = len(dataframe_to_csv_bytes(sample))
    
    # Skalowanie
    scale_factor = n / len(sample)
    estimated = int(sample_bytes * scale_factor)
    
    LOGGER.info(f"CSV size estimate: {estimated / 1024 / 1024:.2f} MB")
    
    return estimated


# ========================================================================================
# EXPORT FUNCTIONS
# ========================================================================================

def ensure_run_directory(run_id: str) -> pathlib.Path:
    """
    Zapewnia istnienie katalogu runu.
    
    Args:
        run_id: ID runu
        
    Returns:
        Ścieżka do katalogu
    """
    run_dir = BASE_EXPORT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir


@safe_operation(ExportResult(ok=False, message="Render failed", artifacts=[]))
def render_html_report(context: ReportContext, run_id: str) -> ExportResult:
    """
    Renderuje raport HTML.
    
    Args:
        context: Kontekst raportu
        run_id: ID runu
        
    Returns:
        ExportResult
    """
    LOGGER.info(f"Rendering HTML report for run {run_id}")
    
    # Generuj HTML
    html_content = build_report_html(asdict(context))
    
    # Zapisz
    run_dir = ensure_run_directory(run_id)
    html_path = run_dir / "report.html"
    
    html_bytes = html_content.encode("utf-8")
    atomic_write(html_path, html_bytes)
    
    # Artifact
    artifact = ExportArtifact(
        path=str(html_path),
        kind="html",
        sha256=compute_sha256(html_bytes),
        size_bytes=len(html_bytes)
    )
    
    LOGGER.info(f"HTML report saved: {html_path} ({len(html_bytes)} bytes)")
    
    return ExportResult(
        ok=True,
        message="HTML wygenerowano pomyślnie",
        artifacts=[artifact],
        run_id=run_id
    )


@safe_operation(ExportResult(ok=False, message="ZIP failed", artifacts=[]))
def build_zip_export(
    context: ReportContext,
    df: Optional[pd.DataFrame],
    dd_df: Optional[pd.DataFrame],
    forecast_df: Optional[pd.DataFrame],
    anomalies_df: Optional[pd.DataFrame],
    run_id: str,
    opts: ReportOptions,
    csv_kwargs: Dict[str, Any]
) -> ExportResult:
    """
    Buduje eksport ZIP.
    
    Args:
        context: Kontekst raportu
        df: DataFrame główny
        dd_df: Data dictionary DataFrame
        forecast_df: Prognoza
        anomalies_df: Anomalie
        run_id: ID runu
        opts: Opcje
        csv_kwargs: Argumenty CSV
        
    Returns:
        ExportResult
    """
    LOGGER.info(f"Building ZIP export for run {run_id}")
    
    run_dir = ensure_run_directory(run_id)
    zip_path = run_dir / "export.zip"
    
    # Przygotuj DataFrame do eksportu
    df_export = None
    if isinstance(df, pd.DataFrame) and not df.empty:
        df_tmp = df
        
        # Anonimizacja
        if opts.anonymize:
            LOGGER.info("Applying anonymization")
            df_tmp = anonymize_dataframe(
                df_tmp,
                opts.anonymize_columns,
                opts.anonymize_mode
            )
        
        # Sampling
        df_export = sample_dataframe(df_tmp, opts.sample_export, opts.sample_rows)
        
        LOGGER.info(f"Export DataFrame: {len(df_export)} rows")
    
    # Build ZIP in memory
    zip_buffer = io.BytesIO()
    artifacts: List[ExportArtifact] = []
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # HTML report
        html_content = build_report_html(asdict(context))
        zf.writestr("report.html", html_content)
        LOGGER.info("Added report.html to ZIP")
        
        # Metadata
        meta_full = asdict(context)
        meta_compact = build_meta_payload(context, opts.compact_meta)
        
        zf.writestr(
            "meta.json",
            json.dumps(meta_full, ensure_ascii=False, indent=2)
        )
        zf.writestr(
            "context_compact.json",
            json.dumps(meta_compact, ensure_ascii=False, indent=2)
        )
        LOGGER.info("Added metadata to ZIP")
        
        # Main data
        if df_export is not None and not df_export.empty:
            # Size check
            estimated_size = estimate_csv_size(df_export)
            size_mb = estimated_size / 1024 / 1024
            
            if size_mb > MAX_CSV_SIZE_MB:
                st.warning(
                    f"⚠️ CSV szacowany na {size_mb:.1f} MB. "
                    "Rozważ mniejszą próbkę."
                )
            
            csv_data = dataframe_to_csv_bytes(df_export, **csv_kwargs)
            zf.writestr("data.csv", csv_data)
            LOGGER.info(f"Added data.csv to ZIP ({len(csv_data)} bytes)")
        
        # Data dictionary
        if dd_df is not None and not dd_df.empty:
            dd_csv = dataframe_to_csv_bytes(dd_df, index=False)
            zf.writestr("data_dictionary.csv", dd_csv)
            LOGGER.info("Added data_dictionary.csv to ZIP")
        
        # Forecast
        if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
            fc_csv = dataframe_to_csv_bytes(forecast_df, **csv_kwargs)
            zf.writestr("forecast.csv", fc_csv)
            LOGGER.info("Added forecast.csv to ZIP")
        
        # Anomalies
        if isinstance(anomalies_df, pd.DataFrame) and not anomalies_df.empty:
            anom_csv = dataframe_to_csv_bytes(anomalies_df, **csv_kwargs)
            zf.writestr("anomalies.csv", anom_csv)
            LOGGER.info("Added anomalies.csv to ZIP")
        
        # ZIP comment
        zf.comment = f"run_id={run_id}".encode("utf-8")
    
    # Write ZIP to disk
    zip_buffer.seek(0)
    zip_bytes = zip_buffer.read()
    
    # Size check
    zip_size_mb = len(zip_bytes) / 1024 / 1024
    if zip_size_mb > MAX_ZIP_SIZE_MB:
        raise ValueError(
            f"ZIP zbyt duży ({zip_size_mb:.1f} MB > {MAX_ZIP_SIZE_MB} MB)"
        )
    
    atomic_write(zip_path, zip_bytes)
    
    LOGGER.info(f"ZIP saved: {zip_path} ({len(zip_bytes)} bytes)")
    
    # Compute checksums for all artifacts
    artifacts_info = []
    
    for file_path in [run_dir / "report.html", zip_path]:
        if file_path.exists():
            file_bytes = file_path.read_bytes()
            artifacts_info.append({
                "path": str(file_path),
                "kind": "html" if file_path.suffix == ".html" else "zip",
                "sha256": compute_sha256(file_bytes),
                "bytes": len(file_bytes)
            })
    
    # Manifest
    manifest = Manifest(
        run_id=run_id,
        created_at=now_timestamp(),
        artifacts=artifacts_info,
        options=asdict(opts),
        context_meta={
            "title": context.title,
            "subtitle": context.subtitle,
            "timestamp": context.run_meta.get("timestamp")
        }
    )
    
    manifest_path = run_dir / "manifest.json"
    manifest_json = json.dumps(asdict(manifest), ensure_ascii=False, indent=2)
    write_text_atomic(manifest_path, manifest_json)
    
    LOGGER.info(f"Manifest saved: {manifest_path}")
    
    # Build artifacts list
    for info in artifacts_info:
        artifacts.append(ExportArtifact(
            path=info["path"],
            kind=info["kind"],  # type: ignore
            sha256=info["sha256"],
            size_bytes=info["bytes"]
        ))
    
    artifacts.append(ExportArtifact(
        path=str(manifest_path),
        kind="manifest",
        sha256=compute_sha256(manifest_json.encode("utf-8")),
        size_bytes=len(manifest_json)
    ))
    
    return ExportResult(
        ok=True,
        message="ZIP wygenerowano pomyślnie",
        artifacts=artifacts,
        run_id=run_id
    )


# ========================================================================================
# STREAMLIT UI
# ========================================================================================

st.title(TITLE)

# ========================================================================================
# DANE Z SESSION STATE
# ========================================================================================

df = st.session_state.get("df") or st.session_state.get("df_raw")
goal = st.session_state.get("goal")
problem_type = st.session_state.get("problem_type")
target = st.session_state.get("target")
model = st.session_state.get("model")
forecast_df = st.session_state.get("forecast_df")
anomalies_df = st.session_state.get("anomalies_df")
automl_metrics = st.session_state.get("last_metrics")

# ========================================================================================
# SIDEBAR: OPCJE
# ========================================================================================

with st.sidebar:
    st.subheader("⚙️ Opcje raportu")
    
    # Podstawowe
    st.markdown("**Zawartość raportu:**")
    
    include_data_dictionary = st.checkbox(
        "📚 Słownik danych",
        value=True,
        help="Tabela z opisem kolumn"
    )
    
    include_df_preview = st.checkbox(
        "📄 Podgląd danych",
        value=False,
        help="Pierwsze 50 wierszy"
    )
    
    include_kpis = st.checkbox(
        "📊 KPI danych",
        value=True,
        help="Metryki jakości"
    )
    
    include_tags = st.checkbox(
        "🏷️ Tagi",
        value=True,
        help="Kontekstowe tagi"
    )
    
    include_forecast = st.checkbox(
        "📈 Prognoza",
        value=True,
        help="Jeśli dostępna"
    )
    
    include_anomalies = st.checkbox(
        "⚠️ Anomalie",
        value=True,
        help="Jeśli dostępne"
    )
    
    st.divider()
    
    # PRO++
    st.markdown("**PRO++ Features:**")
    
    compact_meta = st.checkbox(
        "🗜️ Kompaktuj metadata",
        value=True,
        help="Ucina duże pola do 50K znaków"
    )
    
    sample_export = st.checkbox(
        "🔬 Export próbki",
        value=False,
        help="Eksportuj tylko N pierwszych wierszy"
    )
    
    if sample_export:
        sample_rows = st.number_input(
            "Wiersze w próbce",
            min_value=100,
            max_value=MAX_EXPORT_ROWS,
            value=DEFAULT_SAMPLE_ROWS,
            step=1000
        )
    else:
        sample_rows = DEFAULT_SAMPLE_ROWS
    
    st.divider()
    
    # Anonimizacja
    st.markdown("**Anonimizacja:**")
    
    anonymize = st.checkbox(
        "🔒 Włącz anonimizację",
        value=False,
        help="Hash lub usuń wybrane kolumny"
    )
    
    if anonymize:
        anonymize_mode = st.selectbox(
            "Tryb",
            options=["hash", "drop"],
            index=0,
            help="hash=SHA-256, drop=usuń kolumny"
        )
        
        possible_cols = list(df.columns) if isinstance(df, pd.DataFrame) else []
        anonymize_columns = st.multiselect(
            "Kolumny",
            options=possible_cols,
            default=[],
            help="Kolumny do anonimizacji"
        )
    else:
        anonymize_mode = "hash"
        anonymize_columns = []
    
    st.divider()
    
    # CSV
    st.markdown("**CSV Options:**")
    
    csv_sep = st.selectbox(
        "Separator",
        options=[",", ";", "\t", "|"],
        index=0
    )
    
    csv_encoding = st.selectbox(
        "Encoding",
        options=["utf-8", "utf-8-sig", "cp1250", "latin1"],
        index=0
    )
    
    csv_na_rep = st.text_input(
        "NA reprezentacja",
        value=""
    )

# Konfiguracja
opts = ReportOptions(
    include_data_dictionary=include_data_dictionary,
    include_df_preview=include_df_preview,
    include_kpis=include_kpis,
    include_tags=include_tags,
    include_forecast=include_forecast,
    include_anomalies=include_anomalies,
    compact_meta=compact_meta,
    sample_export=sample_export,
    sample_rows=int(sample_rows),
    anonymize=anonymize,
    anonymize_mode=anonymize_mode,  # type: ignore
    anonymize_columns=tuple(anonymize_columns)
)

csv_kwargs = {
    "sep": csv_sep,
    "encoding": csv_encoding,
    "na_rep": csv_na_rep
}

# ========================================================================================
# NOTATKI
# ========================================================================================

st.subheader("📝 Notatki i wnioski")
notes = st.text_area(
    "Dodaj notatki do raportu",
    placeholder="Np. rekomendacje biznesowe, kluczowe obserwacje...",
    height=100,
    label_visibility="collapsed"
)

# ========================================================================================
# WALIDACJA
# ========================================================================================

st.divider()
st.subheader("🔍 Walidacja danych")

issues = validate_dataframe(df)

if issues:
    with st.expander("⚠️ Wykryte problemy", expanded=True):
        for issue in issues:
            st.warning(f"• {issue}")
else:
    st.success("✅ Dane przeszły walidację")

# Podstawowe info
if isinstance(df, pd.DataFrame) and not df.empty:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Wiersze", f"{len(df):,}")
    col2.metric("Kolumny", f"{df.shape[1]}")
    
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    col3.metric("Pamięć", f"{memory_mb:.1f} MB")
    
    if opts.sample_export:
        actual_rows = min(len(df), opts.sample_rows)
        col4.metric("Export", f"{actual_rows:,} wierszy")
    else:
        col4.metric("Export", "Pełny zbiór")

# ========================================================================================
# KONTEKST (DEBUG)
# ========================================================================================

st.divider()

with st.expander("🧪 Podgląd kontekstu raportu", expanded=False):
    with st.spinner("Budowanie kontekstu..."):
        context = build_report_context(
            opts=opts,
            df=df,
            goal=goal,
            problem_type=problem_type,
            target=target,
            model=model,
            forecast_df=forecast_df,
            anomalies_df=anomalies_df,
            automl_metrics=automl_metrics if isinstance(automl_metrics, dict) else None,
            notes=notes
        )
    
    if context:
        # Skrócony podgląd
        def truncate(v: Any, max_len: int = 600) -> str:
            s = str(v)
            return s if len(s) <= max_len else s[:max_len] + "...<truncated>"
        
        preview = {k: truncate(v) for k, v in asdict(context).items()}
        st.json(preview)
    else:
        st.error("Nie udało się zbudować kontekstu")

# ========================================================================================
# AKCJE EKSPORTU
# ========================================================================================

st.divider()
st.subheader("💾 Eksport")

tab1, tab2 = st.tabs(["🧾 Raport HTML", "📦 ZIP Export"])

# ============================================================================
# TAB 1: HTML REPORT
# ============================================================================

with tab1:
    st.markdown("**Generuj raport HTML z pełnym kontekstem**")
    
    html_button = st.button(
        "🧾 Generuj raport HTML",
        type="primary",
        use_container_width=True,
        key="html_btn"
    )
    
    if html_button:
        # Walidacja
        if issues:
            st.error("❌ Napraw problemy z danymi przed eksportem")
        else:
            run_id = make_run_id("report")
            
            with staged_progress([
                ("Przygotowanie", 15),
                ("Budowanie kontekstu", 40),
                ("Renderowanie HTML", 70),
                ("Zapisywanie", 90),
                ("Finalizacja", 100)
            ]) as step:
                step("Przygotowanie")
                
                # Build context
                step("Budowanie kontekstu")
                context = build_report_context(
                    opts=opts,
                    df=df,
                    goal=goal,
                    problem_type=problem_type,
                    target=target,
                    model=model,
                    forecast_df=forecast_df,
                    anomalies_df=anomalies_df,
                    automl_metrics=automl_metrics if isinstance(automl_metrics, dict) else None,
                    notes=notes
                )
                
                if context is None:
                    st.error("❌ Błąd budowania kontekstu")
                else:
                    # Render
                    step("Renderowanie HTML")
                    result = render_html_report(context, run_id)
                    
                    step("Zapisywanie")
                    
                    if result.ok:
                        step("Finalizacja")
                        
                        st.success(f"✅ {result.message}")
                        
                        # Download button
                        html_path = pathlib.Path(result.artifacts[0].path)
                        html_content = html_path.read_text(encoding="utf-8")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.download_button(
                                "⬇️ Pobierz HTML",
                                data=html_content,
                                file_name="report.html",
                                mime="text/html",
                                use_container_width=True
                            )
                        
                        with col2:
                            st.info(f"📁 Zapisano: `{html_path}`")
                        
                        # Preview
                        with st.expander("👁️ Podgląd raportu", expanded=False):
                            st.components.v1.html(html_content, height=680, scrolling=True)
                        
                        # Historia
                        add_to_history(
                            module="reports",
                            action="render_html",
                            params=asdict(opts),
                            artifacts=result.artifacts,
                            summary=f"HTML report ({run_id})"
                        )
                        
                        # JSONL
                        run_dir = ensure_run_directory(run_id)
                        history = get_history()
                        if history:
                            append_history_jsonl(run_dir, history[0])
                    else:
                        st.error(f"❌ {result.message}")

# ============================================================================
# TAB 2: ZIP EXPORT
# ============================================================================

with tab2:
    st.markdown("**Eksport kompletny: raport + dane + metadata + manifest**")
    
    # Info o zawartości
    with st.expander("ℹ️ Co będzie w ZIP?", expanded=False):
        st.markdown("""
        **Pliki w archiwum:**
        - `report.html` - raport HTML
        - `data.csv` - dane (z opcjonalną anonimizacją/samplingiem)
        - `data_dictionary.csv` - słownik danych
        - `forecast.csv` - prognoza (jeśli dostępna)
        - `anomalies.csv` - anomalie (jeśli dostępne)
        - `meta.json` - pełne metadata
        - `context_compact.json` - skompaktowane metadata
        - `manifest.json` - manifest z checksumami SHA-256
        """)
    
    zip_button = st.button(
        "📦 Generuj ZIP",
        type="primary",
        use_container_width=True,
        key="zip_btn"
    )
    
    if zip_button:
        # Walidacja
        if issues:
            st.error("❌ Napraw problemy z danymi przed eksportem")
        else:
            run_id = make_run_id("export")
            
            with staged_progress([
                ("Przygotowanie", 10),
                ("Budowanie kontekstu", 25),
                ("Przygotowanie danych", 40),
                ("Renderowanie inline", 55),
                ("Budowanie ZIP", 75),
                ("Zapisywanie", 90),
                ("Finalizacja", 100)
            ]) as step:
                step("Przygotowanie")
                
                # Build context
                step("Budowanie kontekstu")
                context = build_report_context(
                    opts=opts,
                    df=df,
                    goal=goal,
                    problem_type=problem_type,
                    target=target,
                    model=model,
                    forecast_df=forecast_df,
                    anomalies_df=anomalies_df,
                    automl_metrics=automl_metrics if isinstance(automl_metrics, dict) else None,
                    notes=notes
                )
                
                if context is None:
                    st.error("❌ Błąd budowania kontekstu")
                else:
                    # Data dictionary
                    step("Przygotowanie danych")
                    dd_df = None
                    if opts.include_data_dictionary and isinstance(df, pd.DataFrame):
                        dd_df = build_data_dictionary(df)
                    
                    # Build ZIP
                    step("Renderowanie inline")
                    step("Budowanie ZIP")
                    
                    result = build_zip_export(
                        context=context,
                        df=df if isinstance(df, pd.DataFrame) else None,
                        dd_df=dd_df,
                        forecast_df=forecast_df if isinstance(forecast_df, pd.DataFrame) else None,
                        anomalies_df=anomalies_df if isinstance(anomalies_df, pd.DataFrame) else None,
                        run_id=run_id,
                        opts=opts,
                        csv_kwargs=csv_kwargs
                    )
                    
                    step("Zapisywanie")
                    
                    if result.ok:
                        step("Finalizacja")
                        
                        st.success(f"✅ {result.message}")
                        
                        # Find ZIP artifact
                        zip_artifact = next(
                            (a for a in result.artifacts if a.kind == "zip"),
                            None
                        )
                        
                        if zip_artifact:
                            zip_path = pathlib.Path(zip_artifact.path)
                            zip_bytes = zip_path.read_bytes()
                            
                            # Download button
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.download_button(
                                    "⬇️ Pobierz ZIP",
                                    data=zip_bytes,
                                    file_name=f"export_{run_id}.zip",
                                    mime="application/zip",
                                    use_container_width=True
                                )
                            
                            with col2:
                                size_mb = len(zip_bytes) / 1024 / 1024
                                st.metric("Rozmiar", f"{size_mb:.2f} MB")
                        
                        # Artifacts info
                        with st.expander("📋 Szczegóły artefaktów", expanded=False):
                            artifacts_data = []
                            for art in result.artifacts:
                                artifacts_data.append({
                                    "Typ": art.kind,
                                    "Ścieżka": art.path,
                                    "Rozmiar": f"{art.size_bytes / 1024:.1f} KB" if art.size_bytes else "N/A",
                                    "SHA-256": art.sha256[:16] + "..." if art.sha256 else "N/A"
                                })
                            
                            st.dataframe(artifacts_data, use_container_width=True)
                        
                        # Historia
                        add_to_history(
                            module="reports",
                            action="export_zip",
                            params=asdict(opts),
                            artifacts=result.artifacts,
                            summary=f"ZIP export ({run_id})"
                        )
                        
                        # JSONL
                        run_dir = ensure_run_directory(run_id)
                        history = get_history()
                        if history:
                            append_history_jsonl(run_dir, history[0])
                    else:
                        st.error(f"❌ {result.message}")

# ========================================================================================
# HISTORIA
# ========================================================================================

history = get_history()

if history:
    st.divider()
    st.subheader("📚 Historia eksportów")
    
    for idx, entry in enumerate(history[:10]):  # Ostatnie 10
        timestamp = entry.get("timestamp", "Unknown")
        action = entry.get("action", "N/A")
        summary = entry.get("summary", "")
        
        with st.expander(
            f"🕒 {timestamp} | {action}",
            expanded=(idx == 0)
        ):
            if summary:
                st.info(summary)
            
            # Artifacts
            artifacts = entry.get("artifacts", [])
            if artifacts:
                st.caption(f"**Artefakty:** {len(artifacts)}")
                for art in artifacts[:5]:  # Max 5
                    st.caption(f"• {art.get('kind', 'N/A')}: `{art.get('path', 'N/A')}`")

# ========================================================================================
# WSKAZÓWKI
# ========================================================================================

st.divider()
st.info(
    "💡 **Wskazówki:**\n\n"
    "- **HTML** - szybki raport do podglądu\n"
    "- **ZIP** - kompletny eksport z danymi i checksumami\n"
    "- **Anonimizacja** - hash (SHA-256) lub drop kolumn\n"
    "- **Sampling** - dla dużych zbiorów (> 100K wierszy)\n"
    "- **Manifest** - zawiera SHA-256 wszystkich plików\n"
    "- **Historia** - JSONL w katalogu każdego runu"
)

st.success(
    "✨ **PRO++ Features:**\n\n"
    "Manifest z checksumami • Atomic writes • "
    "Anonimizacja • Compact metadata • Historie JSONL"
)