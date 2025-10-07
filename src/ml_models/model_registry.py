# src/ml_models/model_registry.py
from __future__ import annotations
import json
import uuid
import hashlib
import pathlib
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import pandas as pd  # tylko do typu annotacyjnego i ewent. metadanych (nie jest wymagane w runtime)

# ----------------------------
# Ścieżki i stałe
# ----------------------------
MODELS_DIR = pathlib.Path(__file__).resolve().parents[2] / "models" / "trained_models"
REGISTRY = MODELS_DIR / "registry.json"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

REG_SCHEMA_VERSION = 2

# ----------------------------
# I/O pomocnicze
# ----------------------------
def _atomic_write_text(path: pathlib.Path, text: str) -> None:
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
        # stary format (dict) → zignoruj/napraw
        if isinstance(data, dict):
            return [data]
    except Exception:
        # uszkodzony plik rejestru – zachowaj ostrożność i nie wywalaj aplikacji
        return []
    return []

def _write_registry(entries: List[dict]) -> None:
    payload = json.dumps(entries, indent=2, ensure_ascii=False)
    _atomic_write_text(REGISTRY, payload)

def _now_iso() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _rel_or_abs(p: Union[str, pathlib.Path]) -> str:
    """Zapisuj ścieżkę względną do MODELS_DIR jeśli to możliwe (czytelniejsze repo)."""
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
    """Upewnij się, że wpis ma podstawowe pola (kompatybilność do przodu)."""
    defaults = {
        "id": None,
        "schema_version": REG_SCHEMA_VERSION,
        "path": None,
        "target": None,
        "problem_type": None,
        "created_at": _now_iso(),
        "metrics": {},
        "columns": None,
        "columns_hash": None,
        "best_estimator": None,
        "model_format": "joblib",
        "tags": [],
        "extra": {},
    }
    out = {**defaults, **(e or {})}
    if out["columns"] and not out.get("columns_hash"):
        out["columns_hash"] = _hash_columns(out["columns"])
    return out

# ----------------------------
# Publiczne API
# ----------------------------
def list_models(
    filters: Optional[Dict[str, Any]] = None,
    sort_by: str = "created_at",
    reverse: bool = True,
) -> List[dict]:
    """
    Zwraca listę wpisów rejestru (można filtrować i sortować).
    Kompatybilność: wywołanie bez argumentów działa jak wcześniej.
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
                    # porównanie częściowe/metryk: {'metrics': {'rmse': ('<', 1.0)}}
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

    # sortowanie
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
    Dodaje wpis do rejestru (jeśli istnieje wpis o tej samej ścieżce – aktualizuje).
    Zwraca pełny wpis.
    """
    mp = _rel_or_abs(model_path)
    entry = {
        "id": uuid.uuid4().hex[:16],
        "schema_version": REG_SCHEMA_VERSION,
        "path": mp,
        "target": target,
        "problem_type": problem_type,
        "created_at": _now_iso(),
        "metrics": metrics or {},
        "columns": columns or None,
        "columns_hash": _hash_columns(columns) if columns else None,
        "best_estimator": best_estimator,
        "model_format": "joblib",
        "tags": tags or [],
        "extra": extra or {},
    }

    items = _read_registry()
    # jeśli już istnieje wpis z tą ścieżką – podmień (pozwala nadpisać metryki)
    replaced = False
    for i, it in enumerate(items):
        if str(it.get("path")) == mp:
            items[i] = entry
            replaced = True
            break
    if not replaced:
        items.append(entry)

    _write_registry(items)
    return entry

def load_model(ref: str) -> Any:
    """
    Ładuje model po:
      - id wpisu z rejestru,
      - nazwie pliku w MODELS_DIR,
      - pełnej ścieżce.
    Zwraca obiekt (np. sklearn Pipeline) lub podnosi wyjątek, jeśli brak.
    """
    # 1) po id
    for e in _read_registry():
        if e.get("id") == ref:
            p = _resolve_path(e.get("path"))
            return joblib.load(p)

    # 2) po nazwie pliku
    p = _resolve_path(ref)
    if p.exists():
        return joblib.load(p)

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

    for e in items:
        if e.get("id") == ref or str(e.get("path")) == ref:
            removed = True
            removed_path = _resolve_path(e.get("path"))
            continue
        kept.append(e)

    # Jeżeli nie znaleziono po id/ścieżce, spróbuj potraktować `ref` jako nazwę pliku
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
            except Exception:
                pass
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

    # domyślny wybór metryki
    if metric is None:
        if (problem_type or "").startswith("regress"):
            metric, maximize = "rmse", False
        else:
            metric, maximize = "f1_weighted", True
    else:
        # heurystyka kierunku: mniejsze lepsze dla rmse/mae/mape, inaczej większe
        maximize = not any(metric.lower().startswith(m) for m in ("rmse", "mae", "mape"))

    # wybór najlepszego
    scored: List[Tuple[float, dict]] = []
    for e in items:
        val = (e.get("metrics") or {}).get(metric)
        if val is None:
            continue
        score = float(val)
        scored.append((score, e))

    if not scored:
        return None

    if maximize:
        return max(scored, key=lambda t: t[0])[1]
    else:
        return min(scored, key=lambda t: t[0])[1]

def stats() -> dict:
    """
    Zwraca szybkie statystyki rejestru.
    """
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
    Porządkuje rejestr:
      - usuwa wpisy wskazujące na nieistniejące pliki,
      - (opcjonalnie) może wykryć pliki bez wpisu (zwraca listę).
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

    # pliki bez wpisu
    files = {p.name for p in MODELS_DIR.glob("*.joblib")}
    paths = {str(_resolve_path(e.get("path")).name) for e in kept}
    orphans = sorted([f for f in files - paths])

    return {
        "removed_entries": len(removed),
        "orphan_files": orphans,
        "kept_entries": len(kept),
    }

# ----------------------------
# Wstecznie kompatybilne API
# ----------------------------
# (Twoje stare funkcje nadal działają)
def clear_registry_legacy():
    clear_registry()

# Utrzymanie starej sygnatury bez parametrów
def list_models_legacy():
    return list_models()
