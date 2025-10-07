# reports_export.py — Reports & Export — TURBO PRO
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
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Literal, Tuple

import pandas as pd
import streamlit as st
from src.ai_engine.report_generator import build_report_html


# =========================
# Konfiguracja i stałe
# =========================
TITLE = "📋 Reports & Export — PRO++"
BASE_EXPORT_DIR = pathlib.Path("data/exports")
MAX_PREVIEW_ROWS = 500
MAX_EXPORT_ROWS = 1_000_000
DEFAULT_SAMPLE_ROWS = 100_000
HISTORY_KEY = ("history", "reports")
META_TRUNCATE_CHARS = 50_000  # limit znaków na pole przy compact meta
CSV_DEFAULTS = dict(index=False, sep=",", encoding="utf-8", na_rep="", lineterminator="\n")

# =========================
# Logger + dekoratory
# =========================
def get_logger(name: str = "reports", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        _h = logging.StreamHandler()
        _h.setLevel(level)
        _h.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(_h)
        logger.propagate = False
    return logger

LOGGER = get_logger()

def safe_op(default=None):
    """Dekorator łapiący wyjątki i zwracający domyślną wartość (nie wywala UI)."""
    def _wrap(fn):
        def _inner(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                LOGGER.exception("Operation failed: %s", fn.__name__)
                st.error(f"❌ Błąd: {e}")
                return default
        return _inner
    return _wrap

# =========================
# Typy / dataclasses
# =========================
@dataclass(frozen=True)
class ReportOptions:
    include_data_dictionary: bool = True
    include_df_preview: bool = False
    include_kpis: bool = True
    include_tags: bool = True
    include_forecast: bool = True
    include_anomalies: bool = True
    # PRO++
    compact_meta: bool = True
    sample_export: bool = False
    sample_rows: int = DEFAULT_SAMPLE_ROWS
    anonymize: bool = False
    anonymize_mode: Literal["hash", "drop"] = "hash"
    anonymize_columns: Tuple[str, ...] = field(default_factory=tuple)  # set at runtime

@dataclass(frozen=True)
class ExportArtifact:
    path: str
    kind: Literal["html", "zip", "csv", "json", "jsonl", "manifest"]

@dataclass(frozen=True)
class ExportResult:
    ok: bool
    message: str
    artifacts: List[ExportArtifact]

@dataclass(frozen=True)
class ReportContext:
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
    run_id: str
    created_at: str
    artifacts: List[Dict[str, Any]]
    options: Dict[str, Any]
    context_meta: Dict[str, Any]


# =========================
# Progress helper
# =========================
from contextlib import contextmanager
@contextmanager
def staged_progress(stages: List[Tuple[str, int]]):
    bar = st.progress(0)
    msg = st.empty()
    lut = {name: pct for name, pct in stages}
    def step(name: str):
        msg.write(f"**{name}…**")
        bar.progress(lut.get(name, 0))
    try:
        yield step
    finally:
        msg.empty()
        bar.progress(100)

# =========================
# Utils: pathy, atomic, checksum
# =========================
def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def make_run_id(prefix: str = "report") -> str:
    base = time.strftime("%Y%m%d-%H%M%S")
    salt = hashlib.sha256(str(time.time()).encode()).hexdigest()[:6]
    return f"{prefix}-{base}-{salt}"

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def atomic_write(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)

def write_text_atomic(path: pathlib.Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write(path, text.encode(encoding))

# =========================
# Walidacje / DF helpers
# =========================
def validate_df(df: Optional[pd.DataFrame]) -> List[str]:
    issues: List[str] = []
    if df is None:
        issues.append("Brak danych (df=None).")
        return issues
    if not isinstance(df, pd.DataFrame):
        issues.append("Obiekt nie jest pandas.DataFrame.")
        return issues
    r, c = df.shape
    if r == 0:
        issues.append("DataFrame jest pusty (0 wierszy).")
    if r > MAX_EXPORT_ROWS:
        issues.append(f"Zbyt dużo wierszy do eksportu ({r:,} > {MAX_EXPORT_ROWS:,}).")
    if c == 0:
        issues.append("DataFrame nie ma kolumn.")
    return issues

@st.cache_data(show_spinner=False)
def df_head_table(d: Optional[pd.DataFrame], max_rows: int = MAX_PREVIEW_ROWS) -> Optional[Dict[str, Any]]:
    if d is None or not isinstance(d, pd.DataFrame) or d.empty:
        return None
    d2 = d.head(max_rows).copy()
    return {"columns": list(map(str, d2.columns)), "rows": d2.astype(object).to_dict(orient="records")}

@st.cache_data(show_spinner=False)
def build_kpis(d: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    if d is None or not isinstance(d, pd.DataFrame) or d.empty:
        return []
    missing_pct = float((d.isna().sum().sum()) / (d.size or 1)) * 100.0
    dupes = int(d.duplicated().sum())
    return [
        {"label": "Wiersze", "value": f"{len(d):,}", "status": "ok"},
        {"label": "Kolumny", "value": f"{d.shape[1]:,}", "status": "ok"},
        {"label": "Braki", "value": f"{missing_pct:.2f}%", "status": "warn" if missing_pct > 1.0 else "ok"},
        {"label": "Duplikaty", "value": f"{dupes:,}", "status": "warn" if dupes > 0 else "ok"},
    ]

@st.cache_data(show_spinner=False)
def build_data_dictionary(d: Optional[pd.DataFrame], top_k: int = 30) -> pd.DataFrame:
    if d is None or not isinstance(d, pd.DataFrame) or d.empty:
        return pd.DataFrame()
    rows = []
    for c in d.columns:
        s = d[c]
        example = ""
        try:
            non_na = s.dropna()
            if not non_na.empty:
                example = str(non_na.iloc[0])[:60]
        except Exception:
            example = ""
        rows.append(
            {
                "column": str(c),
                "dtype": str(s.dtype),
                "missing_pct": round(float(s.isna().mean()) * 100.0, 2),
                "nunique": int(s.nunique(dropna=True)),
                "example": example,
            }
        )
    out = pd.DataFrame(rows).sort_values(["missing_pct", "nunique"], ascending=[False, True]).head(top_k)
    return out

def make_tags(opts: ReportOptions, df: Optional[pd.DataFrame], goal: Optional[str],
              problem_type: Optional[str], target: Optional[str]) -> Optional[List[str]]:
    if not opts.include_tags:
        return None
    tags: List[str] = []
    if goal:
        tags.append("goal")
    if problem_type:
        tags.append(problem_type)
    if target:
        tags.append(f"y:{target}")
    if isinstance(df, pd.DataFrame):
        tags.append(f"{len(df)}x{df.shape[1]}")
    return tags or None

def build_minimal_model_card(model: Any, problem_type: Optional[str], metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "name": getattr(getattr(model, "__class__", None), "__name__", str(type(model))),
        "type": problem_type or "—",
        "version": time.strftime("%Y.%m.%d"),
        "dataset": "session_dataframe",
        "split": "—",
        "hparams": None,
        "metrics": metrics or None,
    }

# =========================
# Anonimizacja / sampling
# =========================
def mask_df(df: pd.DataFrame, cols: Tuple[str, ...], mode: Literal["hash", "drop"]) -> pd.DataFrame:
    if not cols:
        return df
    cols = tuple(c for c in cols if c in df.columns)
    if not cols:
        return df
    if mode == "drop":
        return df.drop(columns=list(cols))
    # hash mode
    out = df.copy()
    for c in cols:
        out[c] = (
            out[c]
            .astype(str, copy=False)
            .map(lambda v: hashlib.sha256(v.encode("utf-8")).hexdigest())
        )
    return out

def maybe_sample(df: pd.DataFrame, enable: bool, n_rows: int) -> pd.DataFrame:
    if not enable:
        return df
    n_rows = max(1, int(n_rows))
    return df.head(n_rows)

# =========================
# Historia działań
# =========================
def _get_history_list() -> List[Dict[str, Any]]:
    root_key, sub_key = HISTORY_KEY
    if root_key not in st.session_state:
        st.session_state[root_key] = {}
    if sub_key not in st.session_state[root_key]:
        st.session_state[root_key][sub_key] = []
    return st.session_state[root_key][sub_key]

def push_history(module: str, action: str, params: Dict[str, Any], artifacts: List[ExportArtifact], summary: str = "") -> None:
    item = {
        "timestamp": now_ts(),
        "module": module,
        "action": action,
        "params": params,
        "summary": summary,
        "artifacts": [asdict(a) for a in artifacts],
    }
    _get_history_list().append(item)

def append_history_jsonl(dirpath: pathlib.Path, item: Dict[str, Any]) -> None:
    f = dirpath / "history.jsonl"
    line = json.dumps(item, ensure_ascii=False)
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")

# =========================
# Budowanie kontekstu raportu
# =========================
@safe_op()
def build_context(
    opts: ReportOptions,
    df: Optional[pd.DataFrame],
    goal: Optional[str],
    problem_type: Optional[str],
    target: Optional[str],
    model: Optional[Any],
    forecast_df: Optional[pd.DataFrame],
    anomalies_df: Optional[pd.DataFrame],
    automl_metrics: Optional[Dict[str, Any]],
    notes: str,
) -> ReportContext:
    metrics: Dict[str, Any] = {}
    if problem_type:
        metrics["problem_type"] = problem_type
    if target:
        metrics["target"] = target
    if isinstance(automl_metrics, dict):
        metrics["automl"] = automl_metrics

    kpis: Optional[List[Dict[str, Any]]] = build_kpis(df) if opts.include_kpis else None
    if kpis == []:
        kpis = None

    dd_df = build_data_dictionary(df)
    dd_table = df_head_table(dd_df) if (opts.include_data_dictionary and not dd_df.empty) else None

    df_preview_table = df_head_table(df.head(50) if isinstance(df, pd.DataFrame) else None) if opts.include_df_preview else None

    forecast_blob = df_head_table(forecast_df) if (opts.include_forecast and isinstance(forecast_df, pd.DataFrame)) else None
    anomalies_blob = df_head_table(anomalies_df) if (opts.include_anomalies and isinstance(anomalies_df, pd.DataFrame)) else None

    model_card = st.session_state.get("model_card")
    if model_card is None and model is not None:
        model_card = build_minimal_model_card(model, problem_type, automl_metrics)

    return ReportContext(
        title="Raport Biznesowy",
        subtitle=goal or "",
        run_meta={"timestamp": time.strftime("%Y-%m-%d %H:%M")},
        metrics=metrics,
        notes=notes,
        kpis=kpis,
        tags=make_tags(opts, df, goal, problem_type, target),
        insights=st.session_state.get("ai_top_insights"),
        recommendations=st.session_state.get("ai_recommendations"),
        forecast=forecast_blob,
        anomalies=anomalies_blob,
        data_dictionary=dd_table,
        df_preview=df_preview_table,
        model_card=model_card,
    )

def compact_obj(obj: Any, char_limit: int = META_TRUNCATE_CHARS) -> Any:
    """Ucina duże pola tekstowe / listy / dict do rozsądnego rozmiaru (dla compact_meta)."""
    try:
        s = json.dumps(obj, ensure_ascii=False)
        if len(s) <= char_limit:
            return obj
        # prosta strategia: zostaw początek i końcówkę
        head = s[: char_limit // 2]
        tail = s[-char_limit // 2 :]
        return f"{head}…<truncated>…{tail}"
    except Exception:
        # fallback dla nietypowych obiektów
        t = str(obj)
        return t[:char_limit] + ("…<truncated>" if len(t) > char_limit else "")

def build_meta_payload(context: ReportContext, compact: bool) -> Dict[str, Any]:
    payload = asdict(context)
    if not compact:
        return payload
    # compact wybranych pól mogących „puchnąć”
    for k in ["insights", "recommendations", "forecast", "anomalies", "data_dictionary", "df_preview", "metrics", "model_card"]:
        if k in payload and payload[k] is not None:
            payload[k] = compact_obj(payload[k], META_TRUNCATE_CHARS)
    return payload

# =========================
# Render / Export + Manifest
# =========================
def ensure_run_directory(run_id: str) -> pathlib.Path:
    d = BASE_EXPORT_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d

@safe_op(ExportResult(ok=False, message="Render failed", artifacts=[]))
def render_html(context: ReportContext, run_id: str) -> ExportResult:
    html = build_report_html(asdict(context))
    out_dir = ensure_run_directory(run_id)
    path = out_dir / "report.html"
    write_text_atomic(path, html, encoding="utf-8")
    return ExportResult(ok=True, message="HTML wygenerowano", artifacts=[ExportArtifact(str(path), "html")])

def dataframe_to_csv_bytes(df: pd.DataFrame, **csv_kwargs) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, **{**CSV_DEFAULTS, **csv_kwargs})
    return buf.getvalue().encode(csv_kwargs.get("encoding", "utf-8"))

def estimate_csv_size_bytes(df: pd.DataFrame, sample_rows: int = 10_000) -> int:
    n = len(df)
    if n == 0:
        return 0
    sample = df.head(min(sample_rows, n))
    sample_bytes = len(dataframe_to_csv_bytes(sample))
    scale = n / len(sample)
    return int(sample_bytes * scale)

@safe_op(ExportResult(ok=False, message="ZIP failed", artifacts=[]))
def build_zip(
    context: ReportContext,
    df: Optional[pd.DataFrame],
    dd_csv_source: Optional[pd.DataFrame],
    forecast_df: Optional[pd.DataFrame],
    anomalies_df: Optional[pd.DataFrame],
    run_id: str,
    opts: ReportOptions,
    csv_kwargs: Optional[Dict[str, Any]] = None,
) -> ExportResult:
    csv_kwargs = csv_kwargs or {}
    out_dir = ensure_run_directory(run_id)
    zip_path = out_dir / "export.zip"

    # potencjalna anonimizacja + sampling
    df_export = None
    if isinstance(df, pd.DataFrame) and not df.empty:
        df_tmp = df
        if opts.anonymize:
            df_tmp = mask_df(df_tmp, opts.anonymize_columns, opts.anonymize_mode)
        df_export = maybe_sample(df_tmp, opts.sample_export, opts.sample_rows)

    # zip in memory
    mem = io.BytesIO()
    artifacts: List[ExportArtifact] = []

    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        # HTML
        html = build_report_html(asdict(context))
        z.writestr("report.html", html)

        # META pełne + compact
        meta_full = asdict(context)
        meta_compact = build_meta_payload(context, compact=opts.compact_meta)
        z.writestr("meta.json", json.dumps(meta_full, ensure_ascii=False, indent=2))
        z.writestr("context_compact.json", json.dumps(meta_compact, ensure_ascii=False, indent=2))

        # dane
        if isinstance(df_export, pd.DataFrame) and not df_export.empty:
            # guard rozmiaru (szacowany)
            est = estimate_csv_size_bytes(df_export)
            if est > 250 * 1024 * 1024:  # ~250MB
                st.warning(f"⚠️ Szacowany rozmiar CSV ≈ {est/1024/1024:.1f} MB. Rozważ Sample export lub kompaktowe kolumny.")
            z.writestr("data.csv", dataframe_to_csv_bytes(df_export, **csv_kwargs))

        # słownik danych
        if dd_csv_source is not None and not dd_csv_source.empty:
            dd_buf = io.StringIO()
            dd_csv_source.to_csv(dd_buf, index=False)
            z.writestr("data_dictionary.csv", dd_buf.getvalue())

        # forecast / anomalies
        if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
            z.writestr("forecast.csv", dataframe_to_csv_bytes(forecast_df, **csv_kwargs))
        if isinstance(anomalies_df, pd.DataFrame) and not anomalies_df.empty:
            z.writestr("anomalies.csv", dataframe_to_csv_bytes(anomalies_df, **csv_kwargs))

        # manifest (tymczasowo bez checksum — dodamy po zapisaniu ZIP na dysk)
        z.comment = f"run_id={run_id}".encode("utf-8")

    mem.seek(0)
    # zapisz ZIP atomowo na dysk
    atomic_write(zip_path, mem.read())
    artifacts.append(ExportArtifact(str(zip_path), "zip"))

    # policz checksumy poszczególnych artefaktów na dysku (report.html, export.zip)
    artifacts_on_disk = []
    for p in [out_dir / "report.html", zip_path]:
        if p.exists():
            b = p.read_bytes()
            artifacts_on_disk.append({
                "path": str(p),
                "kind": "html" if p.suffix == ".html" else "zip",
                "sha256": sha256_bytes(b),
                "bytes": len(b),
            })

    # manifest.json
    manifest = Manifest(
        run_id=run_id,
        created_at=now_ts(),
        artifacts=artifacts_on_disk,
        options=asdict(opts),
        context_meta={"title": context.title, "subtitle": context.subtitle, "timestamp": context.run_meta.get("timestamp")},
    )
    manifest_path = out_dir / "manifest.json"
    write_text_atomic(manifest_path, json.dumps(asdict(manifest), ensure_ascii=False, indent=2))
    artifacts.append(ExportArtifact(str(manifest_path), "manifest"))

    # przycisk pobrania
    st.download_button("⬇️ Pobierz export.zip", data=zip_path.read_bytes(), file_name="export.zip")

    return ExportResult(ok=True, message="ZIP zbudowano", artifacts=artifacts)

# =========================
# UI — główny widok
# =========================
st.title(TITLE)

# Dane z sesji
df = st.session_state.get("df") or st.session_state.get("df_raw")
goal = st.session_state.get("goal")
problem_type = st.session_state.get("problem_type")
target = st.session_state.get("target")
model = st.session_state.get("model")
forecast_df = st.session_state.get("forecast_df")
anomalies_df = st.session_state.get("anomalies_df")
automl_metrics = st.session_state.get("last_metrics")

# Opcje (sidebar)
with st.sidebar:
    st.subheader("⚙️ Opcje raportu")
    # podstawowe
    include_data_dictionary = st.checkbox("Dołącz skrócony słownik danych", value=True)
    include_df_preview = st.checkbox("Dołącz podgląd danych (tabela)", value=False)
    include_kpis = st.checkbox("Dołącz KPI danych", value=True)
    include_tags = st.checkbox("Dołącz tagi kontekstu", value=True)
    include_forecast = st.checkbox("Dołącz prognozę (jeśli dostępna)", value=True)
    include_anomalies = st.checkbox("Dołącz anomalie (jeśli dostępne)", value=True)

    st.markdown("---")
    st.caption("PRO++")
    compact_meta = st.checkbox("Kompaktuj duże pola meta (compact)", value=True)
    sample_export = st.checkbox("Eksport próbki danych (head)", value=False)
    sample_rows = st.number_input("Wiersze w próbce", min_value=100, max_value=MAX_EXPORT_ROWS, value=DEFAULT_SAMPLE_ROWS, step=1000)

    anonymize = st.checkbox("Anonimizuj kolumny", value=False)
    anonymize_mode = st.selectbox("Tryb anonimizacji", options=["hash", "drop"], index=0)
    possible_cols = tuple(df.columns) if isinstance(df, pd.DataFrame) else tuple()
    anonymize_columns = st.multiselect("Kolumny do anonimizacji", options=possible_cols, default=[])

    st.markdown("---")
    st.caption("CSV")
    csv_sep = st.selectbox("Separator", options=[",", ";", "\t", "|"], index=0)
    csv_encoding = st.selectbox("Kodowanie", options=["utf-8", "utf-8-sig", "cp1250"], index=0)
    na_rep = st.text_input("NA reprezentacja", value=CSV_DEFAULTS["na_rep"])

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
        anonymize_columns=tuple(anonymize_columns),
    )
    csv_kwargs = dict(sep=csv_sep, encoding=csv_encoding, na_rep=na_rep)

# Notatki
notes = st.text_area("Notatki / Wnioski", placeholder="Np. rekomendacje biznesowe…")

# Debug i walidacja danych
st.divider()
st.subheader("🧪 Podgląd kontekstu (debug)")
issues = validate_df(df)
if issues:
    with st.expander("⚠️ Ostrzeżenia dot. danych"):
        for msg in issues:
            st.warning(msg)

# Budowa kontekstu
context = build_context(
    opts=opts,
    df=df,
    goal=goal,
    problem_type=problem_type,
    target=target,
    model=model,
    forecast_df=forecast_df,
    anomalies_df=anomalies_df,
    automl_metrics=automl_metrics if isinstance(automl_metrics, dict) else None,
    notes=notes,
)

with st.expander("JSON kontekstu (compact preview)"):
    payload = asdict(context)
    def _short(v: Any):
        s = str(v)
        return s if len(s) <= 600 else s[:600] + " … <truncated>"
    st.json({k: _short(v) for k, v in payload.items()})

# Akcje
st.divider()
col_a, col_b = st.columns(2)

with col_a:
    if st.button("🧾 Generuj raport HTML", type="primary"):
        run_id = make_run_id("report")
        with staged_progress([("Prepare", 15), ("Build context", 40), ("Render HTML", 80), ("Save", 100)]) as step:
            step("Prepare")
            step("Build context")  # context już jest
            step("Render HTML")
            res_html = render_html(context, run_id=run_id)
            step("Save")
        if res_html.ok:
            out_dir = ensure_run_directory(run_id)
            html_path = out_dir / "report.html"
            html_content = html_path.read_text(encoding="utf-8")
            st.success(f"Zapisano: {html_path}")
            st.download_button("⬇️ Pobierz raport (HTML)", data=html_content, file_name="report.html", mime="text/html")
            with st.expander("Podgląd raportu"):
                st.components.v1.html(html_content, height=680, scrolling=True)
            push_history(
                module="reports",
                action="render_html",
                params={"options": asdict(opts)},
                artifacts=res_html.artifacts,
                summary=f"HTML report generated ({run_id})",
            )
            append_history_jsonl(out_dir, _get_history_list()[-1])
        else:
            st.error(res_html.message)

with col_b:
    if st.button("📦 Eksport ZIP (raport + dane + meta)"):
        run_id = make_run_id("report")
        with staged_progress([("Prepare", 15), ("Ensure meta", 35), ("Render inline", 65), ("Build ZIP", 95), ("Save", 100)]) as step:
            step("Prepare")
            # meta / dd
            dd_df = build_data_dictionary(df) if opts.include_data_dictionary else pd.DataFrame()
            step("Ensure meta")
            # inline render do ZIP
            _ = build_report_html(asdict(context))
            step("Render inline")
            res_zip = build_zip(
                context=context,
                df=df if isinstance(df, pd.DataFrame) else None,
                dd_csv_source=dd_df if not dd_df.empty else None,
                forecast_df=forecast_df if isinstance(forecast_df, pd.DataFrame) else None,
                anomalies_df=anomalies_df if isinstance(anomalies_df, pd.DataFrame) else None,
                run_id=run_id,
                opts=opts,
                csv_kwargs=csv_kwargs,
            )
            step("Build ZIP")
            step("Save")
        if res_zip.ok:
            st.success("Export ZIP gotowy.")
            push_history(
                module="reports",
                action="export_zip",
                params={"options": asdict(opts)},
                artifacts=res_zip.artifacts,
                summary=f"ZIP export with report+meta+data ({run_id})",
            )
            out_dir = ensure_run_directory(run_id)
            append_history_jsonl(out_dir, _get_history_list()[-1])
        else:
            st.error(res_zip.message)

st.caption(
    "Tip: Sekcje **Wizualizacje/Forecast/Anomalie/Model Card** pojawią się po zapisaniu ich do `st.session_state`. "
    "Tryb PRO++ dodaje manifest, checksumy, anonimizację i compact meta dla stabilnych, zgodnych eksportów."
)
