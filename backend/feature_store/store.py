# backend/feature_store/store.py
# === KONTEKST BIZNESOWY ===
# Lekki Feature Store dla DS/ML: wersjonowany zapis/odczyt cech jako Parquet z metadanymi.
# Obsługa lokalnie i na S3 (np. MinIO) przez fsspec. Przeznaczony do:
#  - spójnego przechowywania zestawów featurów,
#  - walidacji schematu między wersjami,
#  - bezpiecznych, atomowych zapisów w środowiskach wieloprocesowych (prosty lock),
#  - szybkiego listowania/odczytu i zarządzania retencją.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple, Iterable
import os, io, json, time, hashlib, math
import pandas as pd
from datetime import datetime, timedelta, timezone

# fsspec obsłuży zarówno lokalny FS, jak i s3:// (z s3fs w requirements)
import fsspec

# === DANE I META ===

@dataclass
class FeatureSetSchema:
    """Opis schematu: kolumny i dtypes w formacie pandas string."""
    columns: List[str]
    dtypes: Dict[str, str]

@dataclass
class FeatureSetMeta:
    """Metadane wersji: ścieżki, rozmiary, checksum, tagi i info o schemacie."""
    name: str
    path: str
    created_ts: int
    created_iso: str
    rows: int
    cols: int
    dtypes: Dict[str, str]
    checksum_sha256: str
    tags: List[str]

# === POMOCNICZE ===

def _utc_now_ts() -> int:
    return int(time.time())

def _to_iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

def _hash_df_sha256(df: pd.DataFrame) -> str:
    # Hashujemy deterministycznie – CSV do pamięci (szybkie dla ~MB; dla bardzo dużych: rozważ hash strumieniowy)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return hashlib.sha256(buf.getvalue()).hexdigest()

def _engine_kwargs() -> Dict[str, Any]:
    # Używamy pyarrow (masz w requirements)
    return {"engine": "pyarrow"}

# === GŁÓWNA KLASA ===

class FeatureStore:
    """
    PRO+++ Feature Store:
    - root: katalog lub prefix s3://bucket/prefix
    - atomic write: zapis tymczasowy + rename (na S3 to copy+delete, akceptowalne)
    - registry.json: indeks zarejestrowanych zbiorów i schematów
    """

    REGISTRY = "registry.json"

    def __init__(self, root: str = "feature_store") -> None:
        self.root = root.rstrip("/")
        self.fs, self._base = fsspec.core.url_to_fs(self.root)
        # utworzenie root jeśli lokalny FS
        if self._is_local():
            self.fs.makedirs(self._base, exist_ok=True)
        # ścieżka do rejestru
        self._registry_path = self._join(self.REGISTRY)
        # lazy registry
        self._registry_cache: Optional[Dict[str, Any]] = None

    # === ŚCIEŻKI I FS ===

    def _is_local(self) -> bool:
        return self.fs.protocol in ("file", "local")

    def _join(self, *parts: str) -> str:
        path = "/".join([self._base] + [p.strip("/") for p in parts])
        return f"{self.fs.protocol}://{path}" if not self._is_local() else path

    def _path_version(self, name: str, ts: int) -> Tuple[str, str]:
        base = f"{name}_{ts}"
        return self._join(f"{base}.parquet"), self._join(f"{base}.meta.json")

    def _path_latest(self, name: str) -> Tuple[str, str]:
        return self._join(f"{name}_latest.parquet"), self._join(f"{name}_latest.meta.json")

    def _path_lock(self, name: str) -> str:
        return self._join(f".lock_{name}")

    # === REGISTRY (schemat + opis + tagi domyślne) ===

    def _load_registry(self) -> Dict[str, Any]:
        if self._registry_cache is not None:
            return self._registry_cache
        if self.fs.exists(self._registry_path):
            with self.fs.open(self._registry_path, "r") as f:
                self._registry_cache = json.load(f)
        else:
            self._registry_cache = {}
        return self._registry_cache

    def _save_registry(self) -> None:
        reg = self._load_registry()
        tmp = self._join(f".tmp_registry_{_utc_now_ts()}.json")
        with self.fs.open(tmp, "w") as f:
            json.dump(reg, f, ensure_ascii=False, indent=2)
        # rename to final
        if self.fs.exists(self._registry_path):
            self.fs.rm(self._registry_path)
        self.fs.rename(tmp, self._registry_path)

    def register(
        self,
        name: str,
        df_example: Optional[pd.DataFrame] = None,
        description: Optional[str] = None,
        owner: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Rejestruje zbiór (i opcjonalnie zapisuje schemat z przykładowego df).
        """
        reg = self._load_registry()
        entry = reg.get(name, {})
        entry["name"] = name
        if description is not None:
            entry["description"] = description
        if owner is not None:
            entry["owner"] = owner
        if tags is not None:
            entry["tags"] = tags
        if df_example is not None and not df_example.empty:
            schema = FeatureSetSchema(
                columns=list(df_example.columns),
                dtypes={c: str(df_example[c].dtype) for c in df_example.columns},
            )
            entry["schema"] = asdict(schema)
        reg[name] = entry
        self._save_registry()

    def get_registered(self, name: str) -> Optional[Dict[str, Any]]:
        return self._load_registry().get(name)

    # === LOCK (prosty, bez zależności zewnętrznych) ===

    def _acquire_lock(self, name: str, timeout: float = 10.0, interval: float = 0.1) -> None:
        path = self._path_lock(name)
        t0 = time.time()
        while True:
            try:
                # 'x' -> wyłącznie utworzenie nowego pliku
                with self.fs.open(path, "x") as f:
                    f.write(str(os.getpid()))
                return
            except Exception:
                if time.time() - t0 > timeout:
                    raise TimeoutError(f"Lock timeout for {name}")
                time.sleep(interval)

    def _release_lock(self, name: str) -> None:
        path = self._path_lock(name)
        if self.fs.exists(path):
            self.fs.rm(path)

    # === RETENCJA ===

    def _apply_retention(
        self, name: str, keep_last: Optional[int] = None, max_age_days: Optional[int] = None
    ) -> None:
        if keep_last is None and max_age_days is None:
            return
        versions = self.list_versions(name)
        if not versions:
            return
        to_delete = []

        if keep_last is not None and keep_last >= 0:
            to_delete.extend(versions[keep_last:])  # reszta poza najnowszymi

        if max_age_days is not None and max_age_days > 0:
            cutoff = _utc_now_ts() - int(max_age_days * 24 * 3600)
            to_delete.extend([v for v in versions if v[0] < cutoff])

        # deduplicate i usuń
        seen = set()
        for ts, pparq, pmeta in to_delete:
            if ts in seen:
                continue
            seen.add(ts)
            if self.fs.exists(pparq):
                self.fs.rm(pparq)
            if self.fs.exists(pmeta):
                self.fs.rm(pmeta)

    # === PUBLIC API ===

    def write(
        self,
        df: pd.DataFrame,
        name: str,
        *,
        tags: Optional[List[str]] = None,
        keep_last: Optional[int] = 10,
        max_age_days: Optional[int] = None,
        validate_schema: bool = True,
    ) -> str:
        """
        Zapisuje nową wersję df jako Parquet + meta JSON.
        Utrzymuje alias 'latest'. Zwraca pełną ścieżkę do wersji.
        """
        if df is None or df.empty:
            raise ValueError("Empty DataFrame")

        self._acquire_lock(name)
        try:
            # (1) Walidacja schematu vs registry (jeśli istnieje)
            reg = self.get_registered(name)
            if validate_schema and reg and "schema" in reg:
                expected = reg["schema"]
                exp_cols = expected.get("columns", [])
                exp_dtypes = expected.get("dtypes", {})
                # columns check (kolejność ignorujemy, ale wymagamy nadzbiór)
                missing = [c for c in exp_cols if c not in df.columns]
                if missing:
                    raise ValueError(f"Schema mismatch: missing columns {missing}")
                # dtype check (tylko dla kolumn, które istnieją)
                mismatch = []
                for c in exp_cols:
                    if c in df.columns:
                        if str(df[c].dtype) != str(exp_dtypes.get(c, "")):
                            mismatch.append((c, str(df[c].dtype), str(exp_dtypes.get(c, ""))))
                if mismatch:
                    raise ValueError(f"Schema dtype mismatch: {mismatch}")

            # (2) Najnowszy timestamp i ścieżki
            ts = _utc_now_ts()
            pparq, pmeta = self._path_version(name, ts)
            platest_parq, platest_meta = self._path_latest(name)

            # (3) Hash DF (przed zapisem)
            checksum = _hash_df_sha256(df)

            # (4) Zapis tymczasowy i rename (atomic local / copy+delete s3)
            tmp_parq = self._join(f".tmp_{name}_{ts}.parquet")
            with self.fs.open(tmp_parq, "wb") as f:
                df.to_parquet(f, index=False, **_engine_kwargs())
            # zamiana na final
            if self.fs.exists(pparq):
                self.fs.rm(pparq)
            self.fs.rename(tmp_parq, pparq)

            # (5) Metadata JSON
            meta = FeatureSetMeta(
                name=name,
                path=pparq,
                created_ts=ts,
                created_iso=_to_iso(ts),
                rows=int(df.shape[0]),
                cols=int(df.shape[1]),
                dtypes={c: str(df[c].dtype) for c in df.columns},
                checksum_sha256=checksum,
                tags=tags or [],
            )
            tmp_meta = self._join(f".tmp_{name}_{ts}.meta.json")
            with self.fs.open(tmp_meta, "w") as f:
                json.dump(asdict(meta), f, ensure_ascii=False, indent=2)
            if self.fs.exists(pmeta):
                self.fs.rm(pmeta)
            self.fs.rename(tmp_meta, pmeta)

            # (6) Ustaw alias latest (parquet + meta)
            # uwaga: rename latest może wymagać rm->rename
            if self.fs.exists(platest_parq):
                self.fs.rm(platest_parq)
            self.fs.rename(pparq, platest_parq)
            # skopiuj meta latest (aby wskazywało faktycznie na pparq, nie latest)
            latest_meta_data = asdict(meta)
            latest_meta_data["path"] = pparq  # pointer do wersji, nie do latest
            tmp_latest_meta = self._join(f".tmp_{name}_latest.meta.json")
            with self.fs.open(tmp_latest_meta, "w") as f:
                json.dump(latest_meta_data, f, ensure_ascii=False, indent=2)
            if self.fs.exists(platest_meta):
                self.fs.rm(platest_meta)
            self.fs.rename(tmp_latest_meta, platest_meta)

            # (7) Retencja
            self._apply_retention(name, keep_last=keep_last, max_age_days=max_age_days)

            return pparq
        finally:
            self._release_lock(name)

    def list_versions(self, name: str) -> List[Tuple[int, str, str]]:
        """
        Zwraca listę wersji (ts, path_parquet, path_meta) posortowaną malejąco po ts.
        """
        pattern = self._join(f"{name}_*.parquet")
        files = sorted(self.fs.glob(pattern))
        out: List[Tuple[int, str, str]] = []
        for p in files:
            base = os.path.basename(p)
            # oczekujemy name_<ts>.parquet
            if not base.startswith(f"{name}_") or not base.endswith(".parquet"):
                continue
            try:
                ts = int(base[len(name)+1:-8])  # odetnij "name_" i ".parquet"
            except Exception:
                continue
            pmeta = p.replace(".parquet", ".meta.json")
            out.append((ts, p, pmeta))
        # sort malejąco
        out.sort(key=lambda x: x[0], reverse=True)
        return out

    def read_latest(self, name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        platest_parq, _ = self._path_latest(name)
        if not self.fs.exists(platest_parq):
            raise FileNotFoundError(f"No latest parquet for {name}")
        with self.fs.open(platest_parq, "rb") as f:
            return pd.read_parquet(f, columns=columns, **_engine_kwargs())

    def read_version(self, name: str, ts_or_path: int | str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        if isinstance(ts_or_path, int):
            pparq, _ = self._path_version(name, ts_or_path)
        else:
            pparq = ts_or_path
        if not self.fs.exists(pparq):
            raise FileNotFoundError(f"Version not found: {pparq}")
        with self.fs.open(pparq, "rb") as f:
            return pd.read_parquet(f, columns=columns, **_engine_kwargs())

    def head(self, name: str, n: int = 5) -> pd.DataFrame:
        df = self.read_latest(name)
        return df.head(n)

    def sample(self, name: str, n: int = 5, random_state: int = 42) -> pd.DataFrame:
        df = self.read_latest(name)
        n = min(n, len(df))
        return df.sample(n=n, random_state=random_state) if n > 0 else df

    def latest_meta(self, name: str) -> FeatureSetMeta:
        _, platest_meta = self._path_latest(name)
        if not self.fs.exists(platest_meta):
            raise FileNotFoundError(f"No latest meta for {name}")
        with self.fs.open(platest_meta, "r") as f:
            data = json.load(f)
        return FeatureSetMeta(**data)

    # === UTIL ===

    def exists(self, path: str) -> bool:
        return self.fs.exists(path)

    def uri(self) -> str:
        """Zwraca bazowy URI feature store (np. s3://bucket/prefix lub lokalny path)."""
        return self._join("")
