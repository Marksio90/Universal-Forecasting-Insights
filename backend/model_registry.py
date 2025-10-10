from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple, Union
import os, io, json, time, sys, platform, hashlib
from datetime import datetime, timezone
from functools import lru_cache

import joblib
import fsspec  # lokalnie i s3://

# — logowanie „ciche” gdy brak loguru
try:
    from loguru import logger
except Exception:  # pragma: no cover
    class _L:
        def info(self, *a, **k): ...
        def warning(self, *a, **k): ...
        def error(self, *a, **k): ...
        def debug(self, *a, **k): ...
    logger = _L()  # type: ignore

# === META ===
@dataclass
class ModelMeta:
    name: str
    path: str
    created_ts: int
    created_iso: str
    size_bytes: int
    hash_sha256: str
    framework: str
    model_class: str
    python: str
    libs: Dict[str, str]
    tags: List[str]

# === POMOCNICZE ===
def _ts() -> int: return int(time.time())
def _iso(ts: int) -> str: return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name.strip())

def _join(fs, base: str, *parts: str) -> str:
    path = "/".join([base.rstrip("/")] + [p.strip("/") for p in parts])
    return path if fs.protocol in ("file", "local") else f"{fs.protocol}://{path}"

def _hash_file(fs, path: str, block: int = 2**20) -> str:
    h = hashlib.sha256()
    with fs.open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()

def _framework_of(model: Any) -> str:
    mod = model.__class__.__module__.lower()
    if "xgboost" in mod: return "xgboost"
    if "lightgbm" in mod or "lgbm" in mod: return "lightgbm"
    if "catboost" in mod: return "catboost"
    if "sklearn" in mod: return "sklearn"
    return mod.split(".")[0]

def _libs_versions() -> Dict[str, str]:
    out = {"python": platform.python_version()}
    for lib in ("numpy", "pandas", "scikit_learn", "xgboost", "lightgbm", "catboost", "joblib"):
        try:
            if lib == "scikit_learn":
                import sklearn  # type: ignore
                out["scikit_learn"] = sklearn.__version__
            else:
                mod = __import__(lib if lib != "scikit_learn" else "sklearn")
                out[lib] = getattr(mod, "__version__", "unknown")
        except Exception:
            pass
    return out

def _compression() -> Union[int, tuple]:
    # preferuj lz4 (szybkie), fallback gzip
    try:
        import lz4  # noqa: F401
        return ("lz4", 3)
    except Exception:
        return ("gzip", 3)

# === REGISTRY ===
class ModelRegistry:
    """
    PRO+++ Joblib Model Registry z wersjami, aliasem 'latest', metadanymi i retencją.
    - root: katalog lokalny lub prefix s3://bucket/models (ustaw przez ENV MODEL_STORE_URI)
    """
    def __init__(self, root: str = "models") -> None:
        self.root = root.rstrip("/")
        self.fs, self.base = fsspec.core.url_to_fs(self.root)
        if self.fs.protocol in ("file", "local"):
            self.fs.makedirs(self.base, exist_ok=True)

    # ścieżki
    def _paths(self, name: str, ts: int) -> Tuple[str, str]:
        base = f"{_safe_name(name)}_{ts}"
        return _join(self.fs, self.base, f"{base}.joblib"), _join(self.fs, self.base, f"{base}.meta.json")

    def _latest(self, name: str) -> Tuple[str, str]:
        b = _safe_name(name)
        return _join(self.fs, self.base, f"{b}_latest.joblib"), _join(self.fs, self.base, f"{b}_latest.meta.json")

    def _lock(self, name: str) -> str:
        return _join(self.fs, self.base, f".lock_{_safe_name(name)}")

    # prosty lock plikowy
    def _acquire(self, name: str, timeout: float = 10.0, interval: float = 0.1) -> None:
        p = self._lock(name); t0 = time.time()
        while True:
            try:
                with self.fs.open(p, "x") as f:
                    f.write(str(os.getpid()))
                return
            except Exception:
                if time.time()-t0 > timeout: raise TimeoutError(f"lock timeout for {name}")
                time.sleep(interval)

    def _release(self, name: str) -> None:
        p = self._lock(name)
        if self.fs.exists(p): self.fs.rm(p)

    # API
    def save(
        self, pipeline: Any, name: str, *, tags: Optional[List[str]] = None, keep_last: int = 10, write_card: bool = True
    ) -> str:
        """Zapisz wersję modelu + ustaw alias latest. Zwróć ścieżkę wersji."""
        self._acquire(name)
        try:
            ts = _ts()
            pjob, pmeta = self._paths(name, ts)
            platest, platest_meta = self._latest(name)

            # joblib tymczasowo + rename
            tmp = _join(self.fs, self.base, f".tmp_{_safe_name(name)}_{ts}.joblib")
            with self.fs.open(tmp, "wb") as f:
                joblib.dump(pipeline, f, compress=_compression(), protocol=4)
            if self.fs.exists(pjob): self.fs.rm(pjob)
            self.fs.rename(tmp, pjob)

            size = self.fs.size(pjob)
            sha = _hash_file(self.fs, pjob)

            meta = ModelMeta(
                name=_safe_name(name),
                path=pjob,
                created_ts=ts,
                created_iso=_iso(ts),
                size_bytes=int(size),
                hash_sha256=sha,
                framework=_framework_of(pipeline),
                model_class=pipeline.__class__.__name__,
                python=sys.version.split()[0],
                libs=_libs_versions(),
                tags=tags or [],
            )
            # meta -> wersja
            tmpm = _join(self.fs, self.base, f".tmp_{_safe_name(name)}_{ts}.meta.json")
            with self.fs.open(tmpm, "w") as f:
                json.dump(asdict(meta), f, ensure_ascii=False, indent=2)
            if self.fs.exists(pmeta): self.fs.rm(pmeta)
            self.fs.rename(tmpm, pmeta)

            # alias latest (kopiujemy, żeby latest był niezależny)
            if self.fs.exists(platest): self.fs.rm(platest)
            self.fs.copy(pjob, platest)

            latest_meta = asdict(meta).copy()
            latest_meta["path"] = pjob  # wskazuje na wersję
            tmpl = _join(self.fs, self.base, f".tmp_{_safe_name(name)}_latest.meta.json")
            with self.fs.open(tmpl, "w") as f:
                json.dump(latest_meta, f, ensure_ascii=False, indent=2)
            if self.fs.exists(platest_meta): self.fs.rm(platest_meta)
            self.fs.rename(tmpl, platest_meta)

            # model card
            if write_card:
                card = self._model_card(meta)
                pcard = _join(self.fs, self.base, f"{_safe_name(name)}_{ts}.model-card.md")
                with self.fs.open(pcard, "w") as f:
                    f.write(card)

            # retencja
            self._apply_retention(name, keep_last=keep_last)

            logger.info(f"Model saved: {pjob} ({size} bytes)")  # type: ignore[attr-defined]
            return pjob
        finally:
            self._release(name)

    @lru_cache(maxsize=16)
    def load(self, path_or_name: str, *, version_ts: Optional[int] = None) -> Any:
        """Załaduj model po pełnej ścieżce albo (name, ts|latest)."""
        if self.fs.exists(path_or_name):
            p = path_or_name
        else:
            p, _ = (self._paths(path_or_name, int(version_ts)) if version_ts is not None else self._latest(path_or_name))
        if not self.fs.exists(p):
            raise FileNotFoundError(f"Model not found: {p}")
        with self.fs.open(p, "rb") as f:
            return joblib.load(f)

    def list_versions(self, name: str) -> List[Tuple[int, str, str]]:
        """Lista (ts, path_joblib, path_meta) posortowana malejąco."""
        pattern = _join(self.fs, self.base, f"{_safe_name(name)}_*.joblib")
        files = sorted(self.fs.glob(pattern))
        out: List[Tuple[int, str, str]] = []
        for p in files:
            b = os.path.basename(p)
            if not b.startswith(f"{_safe_name(name)}_") or not b.endswith(".joblib"): continue
            try:
                ts = int(b[len(_safe_name(name))+1:-7])
            except Exception:
                continue
            out.append((ts, p, p.replace(".joblib", ".meta.json")))
        out.sort(key=lambda x: x[0], reverse=True)
        return out

    def latest_path(self, name: str) -> str:
        p, _ = self._latest(name)
        if not self.fs.exists(p): raise FileNotFoundError(f"No latest model for '{name}'")
        return p

    def verify(self, path_or_name: str, *, version_ts: Optional[int] = None) -> bool:
        """Sprawdź zgodność hash-a pliku z metadanymi."""
        if self.fs.exists(path_or_name):
            p = path_or_name; m = p.replace(".joblib", ".meta.json")
        else:
            p, m = (self._paths(path_or_name, int(version_ts)) if version_ts is not None else self._latest(path_or_name))
        if not (self.fs.exists(p) and self.fs.exists(m)): return False
        try:
            with self.fs.open(m, "r") as f: meta = json.load(f)
            return _hash_file(self.fs, p) == meta.get("hash_sha256")
        except Exception:
            return False

    def _apply_retention(self, name: str, keep_last: int = 10) -> None:
        if keep_last is None or keep_last < 0: return
        for ts, p, pm in self.list_versions(name)[keep_last:]:
            try:
                if self.fs.exists(p): self.fs.rm(p)
                if self.fs.exists(pm): self.fs.rm(pm)
            except Exception:
                logger.warning(f"Retention remove failed: {p}")  # type: ignore[attr-defined]

    def _model_card(self, meta: ModelMeta) -> str:
        libs = "\n".join([f"- **{k}**: {v}" for k, v in meta.libs.items()])
        tags = ", ".join(meta.tags) if meta.tags else "-"
        return f"""# Model Card — {meta.name}

**Created:** {meta.created_iso}  
**Path:** `{meta.path}`  
**Size:** {meta.size_bytes} bytes  
**Framework:** {meta.framework}  
**Class:** `{meta.model_class}`  

## Environment
- **Python:** {meta.python}
{libs}

## Integrity
- **SHA256:** `{meta.hash_sha256}`

## Tags
{tags}
"""

# === PROSTE FUNKCJE — kompatybilne z Twoim pierwotnym API ===
_default_registry: Optional[ModelRegistry] = None
def _reg() -> ModelRegistry:
    global _default_registry
    if _default_registry is None:
        base = os.getenv("MODEL_STORE_URI", "models")  # np. s3://bucket/models
        _default_registry = ModelRegistry(base)
    return _default_registry

def save_model(pipeline: Any, name: str) -> str:
    """Drop-in replacement: zapisze model i zwróci ścieżkę wersji."""
    return _reg().save(pipeline, name)

def load_model(path_or_name: str, *, version_ts: Optional[int] = None) -> Any:
    return _reg().load(path_or_name, version_ts=version_ts)

def list_model_versions(name: str) -> List[Tuple[int, str, str]]:
    return _reg().list_versions(name)

def verify_model(path_or_name: str, *, version_ts: Optional[int] = None) -> bool:
    return _reg().verify(path_or_name, version_ts=version_ts)

def latest_model_path(name: str) -> str:
    return _reg().latest_path(name)
