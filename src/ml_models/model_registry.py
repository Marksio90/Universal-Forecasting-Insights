# src/ml_models/model_registry.py — TURBO PRO (back-compat API)
from __future__ import annotations
import json
import uuid
import hashlib
import pathlib
import datetime as dt
import logging
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib

# =========================
# Logger
# =========================
LOGGER = logging.getLogger("model_registry")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    LOGGER.addHandler(_h)
    LOGGER.propagate = False

# =========================
# Ścieżki i stałe
# =========================
MODELS_DIR = pathlib.Path(__file__).resolve().parents[2] / "models" / "trained_models"
REGISTRY = MODELS_DIR / "registry.json"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

REG_SCHEMA_VERSION = 2  # zachowujemy zgodność z Twoim numerowaniem

# =========================
# Dataclass wpisu
# =========================
@dataclass
class ModelEntry:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    schema_version: int = REG_SCHEMA_VERSION
    path: str = ""                     # względna względem MODELS_DIR lub absolutna
    target: Optional[str] = None
    problem_type: Optional[str] = None
    created_at: str = field(default_factory=lambda: dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    metrics: Dict[str, float] = field(default_factory=dict)
    columns: Optional[List[str]] = None
    columns_hash: Optional[str] = None
    best_estimator: Optional[str] = None
    model_format: str = "joblib"
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

# =========================
# I/O pomocnicze
# =========================
def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # kopia zapasowa (ostatni stan)
    if path.exists():
        try:
            backup = path.with_suffix(path.suffix + ".bak")
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

def _read_registry() -> List[dict]:
    if not REGISTRY.exists():
        return []
    try:
        data = json.loads(REGISTRY.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # stary format – znormalizuj do listy
            return [data]
    except Exception as e:
        LOGGER.warning("Nie udało się odczytać rejestru (%s). Zwracam pustą listę.", e)
        return []
    return []

def _write_registry(entries: List[dict]) -> None:
    payload = json.dumps(entries, indent=2, ensure_ascii=False)
    _atomic_write_text(REGISTRY, payload)

def _now_iso() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _rel_or_abs(p: Union[str, pathlib.Path]) -> str:
    """Zapisz ścieżkę względną do MODELS_DIR jeśli się da (czytelniejszy rejestr)."""
    p = pathlib.Path(p)
    try:
        return str(p.relative_to(MODELS_DIR))
    except Exception:
        return str(p)

def _resolve_path(p: Union[str, pathlib.Path]) -> pathlib.Path:
    p = pathlib.Path(p)
    if not p.is_absolute():
        p = MODELS_DIR / p
    return p

def _hash_columns(cols: Optional[List[str]]) -> Optional[str]:
    if not cols:
        return None
    h = hashlib.sha256()
    for c in cols:
        h.update(str(c).encode("utf-8"))
    return h.hexdigest()[:16]

def _normalize_entry(e: dict) -> dict:
    """Upewnij się, że wpis ma kluczowe pola (forward-compat)."""
    defaults = asdict(ModelEntry(path=""))
    # created_at i id zostawimy jeśli już są
    out = {**defaults, **(e or {})}
    # kolumny → hash
    if out.get("columns") and not out.get("columns_hash"):
        out["columns_hash"] = _hash_columns(out.get("columns"))
    # schema_version fallback
    out["schema_version"] = out.get("schema_version") or REG_SCHEMA_VERSION
    return out

# =========================
# Publiczne API
# =========================
def list_models(
    filters: Optional[Dict[str, Any]] = None,
    sort_by: str = "created_at",
    reverse: bool = True,
) -> List[dict]:
    """
    Zwraca listę wpisów rejestru (filtrowanie i sortowanie opcjonalne).
    filters wspiera:
      - proste equality: {"target": "y"}
      - zbiór wartości: {"problem_type": {"classification","regression"}}
      - zagnieżdżone metryki: {"metrics": {"rmse": ("<", 1.0)}}
    """
    items = [_normalize_entry(e) for e in _read_registry()]
    if filters:
        def ok(e: dict) -> bool:
            for k, v in filters.items():
                if k not in e:
                    return False
                if isinstance(v, (list, tuple, set)):
                    if e[k] not in v:
                        return False
                elif isinstance(v, dict):
                    # np. {'metrics': {'rmse': ('<', 1.2)}}
                    for mk, cond in v.items():
                        mv = (e.get(k) or {}).get(mk)
                        if isinstance(cond, tuple) and len(cond) == 2:
                            op, thr = cond
                            if op == "<" and not (mv is not None and mv < thr): return False
                            if op == ">" and not (mv is not None and mv > thr): return False
                            if op == "<=" and not (mv is not None and mv <= thr): return False
                            if op == ">=" and not (mv is not None and mv >= thr): return False
                            if op == "==" and not (mv == thr): return False
                        else:
                            if mv != cond:
                                return False
                else:
                    if e[k] != v:
                        return False
            return True
        items = [e for e in items if ok(e)]

    try:
        items.sort(key=lambda e: e.get(sort_by) or "", reverse=reverse)
    except Exception:
        pass
    return items

def clear_registry() -> None:
    """Czyści rejestr do pustej listy."""
    _write_registry([])

def register_model(
    *,
    model_path: Union[str, pathlib.Path],
    target: str,
    problem_type: str,
    metrics: Optional[Dict[str, float]] = None,
    columns: Optional[List[str]] = None,
    best_estimator: Optional[str] = None,
    tags: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Dodaje lub aktualizuje wpis rejestru dla podanej ścieżki modelu.
    Zwraca pełny, znormalizowany wpis.
    """
    mp = _rel_or_abs(model_path)
    entry = ModelEntry(
        path=mp,
        target=target,
        problem_type=problem_type,
        metrics=metrics or {},
        columns=columns or None,
        columns_hash=_hash_columns(columns) if columns else None,
        best_estimator=best_estimator,
        tags=tags or [],
        extra=extra or {},
    )
    # odczyt + idempotentny update po ścieżce
    items = _read_registry()
    replaced = False
    for i, it in enumerate(items):
        if str(it.get("path")) == mp:
            # zachowaj stare id/created_at gdy aktualizujemy
            old = _normalize_entry(it)
            d = asdict(entry)
            d["id"] = old.get("id") or d["id"]
            d["created_at"] = old.get("created_at") or d["created_at"]
            items[i] = d
            replaced = True
            break
    if not replaced:
        items.append(asdict(entry))

    _write_registry(items)
    return _normalize_entry(asdict(entry))

def load_model(ref: str) -> Any:
    """
    Ładuje model po:
      - id wpisu z rejestru,
      - nazwie pliku w MODELS_DIR,
      - pełnej ścieżce.
    Zwraca obiekt modelu. Jeśli plik przechowuje payload dict z kluczem 'model',
    zwraca payload['model'] (fallback do całego obiektu w przeciwnym razie).
    """
    # 1) po id
    for e in _read_registry():
        if e.get("id") == ref:
            p = _resolve_path(e.get("path"))
            obj = joblib.load(p)
            if isinstance(obj, dict) and "model" in obj:
                return obj["model"]
            return obj

    # 2) po nazwie/ścieżce
    p = _resolve_path(ref)
    if p.exists():
        obj = joblib.load(p)
        if isinstance(obj, dict) and "model" in obj:
            return obj["model"]
        return obj

    raise FileNotFoundError(f"Nie znaleziono modelu dla ref='{ref}'.")

def delete_model(ref: str, remove_file: bool = True) -> bool:
    """
    Usuwa wpis z rejestru i opcjonalnie plik modelu.
    `ref` jak w load_model().
    """
    items = _read_registry()
    kept: List[dict] = []
    removed = False
    removed_path: Optional[pathlib.Path] = None

    # dopasowanie po id/ścieżce (wpisu)
    for e in items:
        if e.get("id") == ref or str(e.get("path")) == ref:
            removed = True
            removed_path = _resolve_path(e.get("path"))
            continue
        kept.append(e)

    # jeśli jeszcze nie – spróbuj potraktować `ref` jako nazwę pliku
    if not removed:
        ref_path = _resolve_path(ref)
        kept2 = []
        for e in kept:
            if _resolve_path(e.get("path")) == ref_path:
                removed = True
                removed_path = ref_path
                continue
            kept2.append(e)
        kept = kept2

    if removed:
        _write_registry(kept)
        if remove_file and removed_path and removed_path.exists():
            try:
                removed_path.unlink()
            except Exception as e:
                LOGGER.warning("Nie udało się usunąć pliku modelu %s (%s)", removed_path, e)
    return removed

def get_best_model(
    *,
    target: Optional[str] = None,
    problem_type: Optional[str] = None,
    metric: Optional[str] = None,
) -> Optional[dict]:
    """
    Zwraca najlepszy wpis wg metryki.
    Domyślne metryki:
      - regresja: rmse (min)
      - klasyfikacja: f1_weighted (max)
    """
    items = list_models(filters={k: v for k, v in {"target": target, "problem_type": problem_type}.items() if v})
    if not items:
        return None

    # domyślna metryka + kierunek
    if metric is None:
        if (problem_type or "").startswith("regress"):
            metric, maximize = "rmse", False
        else:
            metric, maximize = "f1_weighted", True
    else:
        maximize = not any(metric.lower().startswith(m) for m in ("rmse", "mae", "mape"))

    scored: List[Tuple[float, dict]] = []
    for e in items:
        val = (e.get("metrics") or {}).get(metric)
        if val is None:
            continue
        try:
            scored.append((float(val), e))
        except Exception:
            continue

    if not scored:
        return None
    return (max(scored, key=lambda t: t[0])[1] if maximize else min(scored, key=lambda t: t[0])[1])

def stats() -> dict:
    """Zwraca szybkie statystyki rejestru."""
    items = _read_registry()
    by_type: Dict[str, int] = {}
    by_target: Dict[str, int] = {}
    for e in items:
        by_type[e.get("problem_type", "unknown")] = by_type.get(e.get("problem_type", "unknown"), 0) + 1
        by_target[e.get("target", "unknown")] = by_target.get(e.get("target", "unknown"), 0) + 1

    return {
        "count": len(items),
        "by_problem_type": by_type,
        "by_target": by_target,
        "last_created_at": max((e.get("created_at", "") for e in items), default=None),
    }

def prune_orphans() -> dict:
    """
    Porządki:
      - usuwa wpisy wskazujące na nieistniejące pliki,
      - zwraca listę plików bez wpisów.
    """
    items = _read_registry()
    kept, removed = [], []
    for e in items:
        p = _resolve_path(e.get("path"))
        if p.exists():
            kept.append(e)
        else:
            removed.append(e)

    _write_registry(kept)

    files = {p.name for p in MODELS_DIR.glob("*.joblib")}
    paths = {str(_resolve_path(e.get("path")).name) for e in kept}
    orphans = sorted(list(files - paths))

    return {
        "removed_entries": len(removed),
        "orphan_files": orphans,
        "kept_entries": len(kept),
    }

# =========================
# Wstecznie kompatybilne API
# =========================
def clear_registry_legacy():
    clear_registry()

def list_models_legacy():
    return list_models()
